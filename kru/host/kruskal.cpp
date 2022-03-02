#include <vector>
#include <fstream>
#include <iostream>
#include <CL/opencl.h>
#include "AOCLUtils/aocl_utils.h"


using namespace aocl_utils;
class Graph
{
public:
    void insertEdge(int src, int dst, double weight){
        this->edgeSrc.emplace_back(src);
        this->edgeDst.emplace_back(dst);
        this->edgeWeight.emplace_back(weight);
    }

    void readFile2Graph(std::string fileName){
        std::cout<<fileName<<std::endl;
        std::ifstream Gin(fileName);
        if (!Gin.is_open()) { std::cout << "Error! Graph file not found!" << std::endl; exit(0); }
        Gin >> this->vCount >> this->eCount;
        this->vCount = ((this->vCount - 1) / 1024 + 1) *1024;
        this->vertexID.resize(vCount, 0);
        this->vertexActive.resize(vCount, 0);
        this->edgeSrc.reserve(this->eCount);
        this->edgeDst.reserve(this->eCount);
        this->edgeWeight.reserve(this->eCount);
        for (int i = 0; i < this->vCount; ++i)   this->vertexID[i] = i;
        for (int i = 0; i < eCount; i++)
        {
            int dst, src;
            Gin >>src >> dst ;
            this->insertEdge(src, dst, 1);
        }
        Gin.close();
    }

    std::vector<Graph> divideGraphByEdge(int partition){
        if (subGraph.size() == 0) {
            Graph g;
            subGraph.resize(partition, g);
            for (int i = 0; i < partition; ++i) {
                subGraph.at(i).vCount = this->vCount;
                subGraph.at(i).vertexID = this->vertexID;
                subGraph.at(i).distance = this->distance;
                subGraph.at(i).vertexActive = this->vertexActive;
                subGraph.at(i).activeNodeNum = this->activeNodeNum;
                for (int k = i * this->eCount / partition; k < (i + 1) * this->eCount / partition; k++)
                    subGraph.at(i).insertEdge(this->edgeSrc.at(k), this->edgeDst.at(k), this->edgeWeight.at(k));
                subGraph.at(i).eCount = subGraph.at(i).edgeSrc.size();
            }
        }
        else {
            for (int i = 0; i < partition; ++i) {
                subGraph.at(i).distance = this->distance;
                subGraph.at(i).vertexActive = this->vertexActive;
                subGraph.at(i).activeNodeNum = this->activeNodeNum;
            }
            
        }
        return this->subGraph;
    }

    int vCount;
    int eCount;
    int activeNodeNum = 0;

    std::vector<int> edgeSrc;
    std::vector<int> edgeDst;
    std::vector<int> edgeWeight;
    std::vector<int> vertexID;
    std::vector<int> vertexActive;
    std::vector<int> distance;
    std::vector<Graph> subGraph = std::vector<Graph>();
};

class Env
{
public:
	void setEnv(){
        cl_int iStatus = 0;
        this->platform = findPlatform("Intel");
        noPtrCheck(this->platform,"ERROR: Unable to find Intel(R) FPGA OpenCL platform.\n");
        
        cl_uint num_devices = 0;
        cl_device_id* devices =  getDevices(this->platform, CL_DEVICE_TYPE_ALL, &num_devices);
        this->device = devices[0];
        noPtrCheck(this->device,"ERROR: Unable to get device env.\n");

        this->context = clCreateContext(NULL, 1, &this->device, NULL, NULL,NULL);//changed
        noPtrCheck(this->context, "Can not create context");

        this->queue = clCreateCommandQueue(this->context, this->device, CL_QUEUE_PROFILING_ENABLE, NULL);
        noPtrCheck(this->queue, "Can not create CommandQueue");

        std::string binary_file = getBoardBinaryFile("/home/fpga/HGC/Kru/bin/kru", this->device);
        printf("Using AOCX: %s\n", binary_file.c_str());
        this->program = createProgramFromBinary(this->context, binary_file.c_str(), &this->device, 1);
        iStatus = clBuildProgram(this->program, 0, NULL, "", NULL, NULL);
        if (CL_SUCCESS != iStatus)
        {
            std::cout << "Error: Can not build program" << std::endl;
            char szBuildLog[16384];
            clGetProgramBuildInfo(this->program, this->device, CL_PROGRAM_BUILD_LOG, sizeof(szBuildLog), szBuildLog, NULL);
            std::cout << "Error in Kernel: " << std::endl << szBuildLog;
            clReleaseProgram(this->program);
            exit(0);
        }

        std::vector<std::string> KernelName{"GenMerge","Apply","MergeGraph","Gather"};
        for(int i = 0 ; i < 4 ;++i){
            this->kernels.push_back(clCreateKernel(this->program, KernelName[i].c_str(), NULL));
            noPtrCheck(this->kernels[i], "Failed to create kernel."+i);
        }
    }

    void errorCheck(cl_int iStatus, std::string errMsg) {
        if (CL_SUCCESS != iStatus) {
            std::cout <<"error:"<<iStatus<<"  :" <<errMsg << std::endl;
            exit(0);
        }
    }

    void noPtrCheck(void* ptr, std::string errMsg) {
        if (NULL == ptr) {
            std::cout << "error: " << errMsg << std::endl;
            exit(0);
        }
    }

	bool init = false;
	int memNum = 0;
	cl_platform_id platform = nullptr;
	cl_device_id device = nullptr;
	cl_program program = nullptr;
	cl_context context = nullptr;
	std::vector<cl_kernel> kernels;
	cl_command_queue queue = nullptr;
	std::vector<std::vector<cl_mem>> clMem;
};

class Kruskal
{
public:
	Kruskal();

	Kruskal(std::string GraphPath,int initNode,int partition,int iter){
        loadGraph(GraphPath);
        this->MemSpace = this->graph.vCount;
        this->graph.distance.resize(MemSpace, INT32_MAX);
        this->graph.vertexActive.reserve(this->graph.vCount);
        this->graph.vertexActive.assign(this->graph.vCount, 0);
        this->graph.distance[initNode] = 0;
        this->graph.vertexActive[initNode] = 1;
        this->graph.activeNodeNum = 1;

        env.setEnv();
        this->Engine_FPGA(partition,iter);
    }

	void loadGraph(std::string filePath){
        graph.readFile2Graph(filePath);
        std::cout<<"load graph success"<<std::endl;
    }

	void Engine_FPGA(int partition,int runiter){
        int iter = 0;
        std::vector<int> mValues(this->MemSpace);
        clock_t start, end, subStart, subEnd, subiter;
        start = clock();
        while (this->graph.activeNodeNum > 0) {
            std::cout << "----------------------" << std::endl;
            std::cout << "this is iter : " << iter++ << std::endl;
            //subStart = clock();
            //subiter = clock();
            std::vector<Graph> subGraph = graph.divideGraphByEdge(partition);
            //cout << "divide run time: " << (double)(clock() - subStart) << "ms" << endl;
            for (auto& g : subGraph) {
                mValues.assign(this->MemSpace, INT32_MAX);
                //subStart = clock();
                MSGGenMerge_FPGA(g, mValues);
                //cout << "Gen run time: " << (double)(clock() - subStart) << "ms" << endl;
                //subStart = clock();
                MSGApply_FPGA(g, mValues);
                //cout << "Apply run time: " << (double)(clock() - subStart) << "ms" << endl;
            }
            //subStart = clock();
            MergeGraph_FPGA(subGraph);
            //cout << "mergeGraph run time: " << (double)(clock() - subStart) << "ms" << endl;
            //subStart = clock();
            this->graph.activeNodeNum = GatherActiveNodeNum_FPGA(this->graph.vertexActive);
            //cout << "Gather run time: " << (double)(clock() - subStart) << "ms" << endl;
            //cout << "------------------------------" << endl;
            //cout << "iter run  time: " << (double)(clock() - subiter) << "ms" << endl;
            std::cout << "active node number" << this->graph.activeNodeNum << std::endl;
            std::cout << "------------------------------" << std::endl;
            if(iter > runiter)  break;
        }
        end = clock();
        std::cout << "Run time: " << (double)(end - start)/CLOCKS_PER_SEC << "s" << std::endl;
    }

	void MSGGenMerge_FPGA(Graph& g, std::vector<int>& mValue){
        if (g.vCount <= 0) return;
        size_t globalSize = g.eCount;
        cl_int iStatus = 0;
        size_t dim = 1;
        int index = -1,kernelID = 0;

        if(env.clMem.size() == 0){
            std::vector<cl_mem> tmp(6, nullptr);
            tmp[++index] = clCreateBuffer(env.context, CL_MEM_READ_WRITE, g.eCount * sizeof(int), nullptr, nullptr);//src
            tmp[++index] = clCreateBuffer(env.context, CL_MEM_READ_WRITE, g.eCount * sizeof(int), nullptr, nullptr);//dst
            tmp[++index] = clCreateBuffer(env.context, CL_MEM_READ_WRITE, g.eCount * sizeof(int), nullptr, nullptr);//weight
            tmp[++index] = clCreateBuffer(env.context, CL_MEM_READ_WRITE, g.vCount * sizeof(int), nullptr, nullptr);//vertex
            tmp[++index] = clCreateBuffer(env.context, CL_MEM_READ_WRITE, this->MemSpace * sizeof(int), nullptr, nullptr);//mValue
            tmp[++index] = clCreateBuffer(env.context, CL_MEM_READ_WRITE, this->MemSpace * sizeof(int), nullptr, nullptr);//vValue

            for (int i = 0; i <= index; i++) {
                if (tmp[i] == nullptr)
                    env.noPtrCheck(nullptr, "set mem error");
            }
            env.clMem.push_back(tmp);

            for (int i = 0; i <= index; i++) {
                iStatus |= clSetKernelArg(env.kernels[kernelID], i, sizeof(cl_mem), &env.clMem[kernelID][i]);
            }
            checkError(iStatus, "set kernel agrs fail!");
        }

        cl_event startEvt;
        index = -1;
        clEnqueueWriteBuffer(env.queue, env.clMem[kernelID][++index], CL_TRUE, 0, g.eCount * sizeof(int), &g.edgeSrc[0], 0, nullptr, nullptr);
        clEnqueueWriteBuffer(env.queue, env.clMem[kernelID][++index], CL_TRUE, 0, g.eCount * sizeof(int), &g.edgeDst[0], 0, nullptr, nullptr);
        clEnqueueWriteBuffer(env.queue, env.clMem[kernelID][++index], CL_TRUE, 0, g.eCount * sizeof(int), &g.edgeWeight[0], 0, nullptr, nullptr);
        clEnqueueWriteBuffer(env.queue, env.clMem[kernelID][++index], CL_TRUE, 0, g.vCount * sizeof(int), &g.vertexActive[0], 0, nullptr, nullptr);
        clEnqueueWriteBuffer(env.queue, env.clMem[kernelID][++index], CL_TRUE, 0, this->MemSpace * sizeof(int), &mValue[0],  0, nullptr, nullptr);
        clEnqueueWriteBuffer(env.queue, env.clMem[kernelID][++index], CL_TRUE, 0, this->MemSpace * sizeof(int), &g.distance[0], 0, nullptr, &startEvt);
        clWaitForEvents(1, &startEvt);

        iStatus = clEnqueueNDRangeKernel(env.queue, env.kernels[kernelID], dim, NULL, &globalSize, nullptr, 0, NULL, NULL);
        env.errorCheck(iStatus, "Can not run kernel");

        iStatus = clEnqueueReadBuffer(env.queue, env.clMem[kernelID][4], CL_TRUE, 0, this->MemSpace * sizeof(int), &mValue[0], 0, NULL, NULL);
        env.errorCheck(iStatus, "Can not reading result buffer");
    }

	void MSGApply_FPGA(Graph& g, std::vector<int>& mValue){
        g.activeNodeNum = 0;
	
        size_t globalSize = g.vCount;
        int kernelID = 1, index = 0;
        cl_int iStatus = 0;
        size_t dim = 1;

        if (env.clMem.size() == 1) {
            std::vector<cl_mem> tmp(3, nullptr);
            index = -1;

            tmp[++index] = clCreateBuffer(env.context, CL_MEM_READ_WRITE, g.vCount * sizeof(int), nullptr, nullptr);
            tmp[++index] = clCreateBuffer(env.context, CL_MEM_READ_WRITE, this->MemSpace * sizeof(int), nullptr, nullptr);
            tmp[++index] = clCreateBuffer(env.context, CL_MEM_READ_WRITE, this->MemSpace * sizeof(int), nullptr, nullptr);

            for (int i = 0; i <= index; i++) {
                if (tmp[i] == nullptr)
                    env.noPtrCheck(nullptr, "set mem error");
            }
            env.clMem.push_back(tmp);

            for (int i = 0; i <= index; i++) {
                iStatus |= clSetKernelArg(env.kernels[kernelID], i, sizeof(cl_mem), &env.clMem[kernelID][i]);
            }
            checkError(iStatus, "set kernel agrs fail!");
        }

        cl_event startEvt;
        index = -1;
        clEnqueueWriteBuffer(env.queue, env.clMem[kernelID][++index], CL_TRUE, 0, g.vCount * sizeof(int), &g.vertexActive[0], 0, nullptr, nullptr);
        clEnqueueWriteBuffer(env.queue, env.clMem[kernelID][++index], CL_TRUE, 0, this->MemSpace * sizeof(int), &mValue[0], 0, nullptr, nullptr);
        clEnqueueWriteBuffer(env.queue, env.clMem[kernelID][++index], CL_TRUE, 0, this->MemSpace * sizeof(int), &g.distance[0], 0, nullptr, &startEvt);
        clWaitForEvents(1, &startEvt);

        iStatus = clEnqueueNDRangeKernel(env.queue, env.kernels[kernelID], dim, NULL, &globalSize, nullptr, 0, NULL, NULL);
        checkError(iStatus, "Can not run Apply kernel");

        iStatus = clEnqueueReadBuffer(env.queue, env.clMem[kernelID][0], CL_TRUE, 0, g.vCount * sizeof(int), &g.vertexActive[0], 0, NULL, NULL);
        iStatus = clEnqueueReadBuffer(env.queue, env.clMem[kernelID][2], CL_TRUE, 0, this->MemSpace * sizeof(int), &g.distance[0], 0, NULL, NULL);
        checkError(iStatus, "Can not reading result buffer");
    }

    void MergeGraph_FPGA(std::vector<Graph>& subGraph){
        size_t globalSize = this->graph.vCount;
        cl_int iStatus = 0;
        size_t dim = 1;
        int index = -1;

        int kernelID = 2;
        if (env.clMem.size() == 2) {
            std::vector<cl_mem> tmp(4, nullptr);
            tmp[++index] = clCreateBuffer(env.context, CL_MEM_READ_WRITE, this->graph.vCount * sizeof(int), nullptr, nullptr);//src
            tmp[++index] = clCreateBuffer(env.context, CL_MEM_READ_WRITE, this->graph.vCount * sizeof(int), nullptr, nullptr);//dst
            tmp[++index] = clCreateBuffer(env.context, CL_MEM_READ_WRITE, this->MemSpace * sizeof(int), nullptr, nullptr);
            tmp[++index] = clCreateBuffer(env.context, CL_MEM_READ_WRITE, this->MemSpace * sizeof(int), nullptr, nullptr);

            for (int i = 0; i <= index; i++) {
                if (tmp[i] == nullptr)
                    env.noPtrCheck(nullptr, "set mem error");
            }
            env.clMem.push_back(tmp);

            for (int i = 0; i <= index; i++) {
                iStatus |= clSetKernelArg(env.kernels[kernelID], i, sizeof(cl_mem), &env.clMem[kernelID][i]);
            }
            env.errorCheck(iStatus, "set kernel agrs fail!");
        }
        
        cl_event startEvt;
        index = -1;
        clEnqueueWriteBuffer(env.queue, env.clMem[kernelID][++index], CL_TRUE, 0, this->graph.vCount * sizeof(int), &subGraph[0].vertexActive[0], 0, nullptr, nullptr);
        clEnqueueWriteBuffer(env.queue, env.clMem[kernelID][++index], CL_TRUE, 0, this->graph.vCount * sizeof(int), &subGraph[1].vertexActive[0], 0, nullptr, nullptr);
        clEnqueueWriteBuffer(env.queue, env.clMem[kernelID][++index], CL_TRUE, 0, this->MemSpace * sizeof(int), &subGraph[0].distance[0], 0, nullptr, nullptr);
        clEnqueueWriteBuffer(env.queue, env.clMem[kernelID][++index], CL_TRUE, 0, this->MemSpace * sizeof(int), &subGraph[1].distance[0], 0, nullptr, &startEvt);
        clWaitForEvents(1, &startEvt);

        iStatus = clEnqueueNDRangeKernel(env.queue, env.kernels[kernelID], dim, NULL, &globalSize, nullptr, 0, NULL, NULL);
        env.errorCheck(iStatus, "Can not run GenMerge kernel");

        iStatus = clEnqueueReadBuffer(env.queue, env.clMem[kernelID][0], CL_TRUE, 0, this->MemSpace * sizeof(int), &this->graph.vertexActive[0], 0, NULL, NULL);
        iStatus = clEnqueueReadBuffer(env.queue, env.clMem[kernelID][2], CL_TRUE, 0, this->MemSpace * sizeof(int), &this->graph.distance[0], 0, NULL, NULL);

        env.errorCheck(iStatus, "Can not reading result buffer");
    }

	int GatherActiveNodeNum_FPGA(std::vector<int>& activeNodes){
        int kernelID = 3, index = 0;
        const size_t localSize = 256;
        int len = activeNodes.size();
        int group = len/ localSize;
        const size_t globalSize = len;

        cl_int iStatus = 0;
        size_t dim = 1;
        std::vector<int> subSum(group, 0);
        if (env.clMem.size() == 3) {
            std::vector<cl_mem> tmp(2, nullptr);

            tmp[0] = clCreateBuffer(env.context, CL_MEM_READ_WRITE, globalSize * sizeof(int), nullptr, nullptr);
            tmp[1] = clCreateBuffer(env.context, CL_MEM_READ_WRITE, group * sizeof(int), nullptr, nullptr);
            if (tmp[0] == nullptr || tmp[1] == nullptr)
                env.noPtrCheck(nullptr, "set mem fail");

            env.clMem.push_back(tmp);
            env.errorCheck(clSetKernelArg(env.kernels[kernelID], 0, sizeof(cl_mem), &env.clMem[kernelID][0]), "set arg fail");
            env.errorCheck(clSetKernelArg(env.kernels[kernelID], 1, sizeof(cl_mem), &env.clMem[kernelID][1]), "set arg fail");
            env.errorCheck(clSetKernelArg(env.kernels[kernelID], 2, localSize * sizeof(int), nullptr), "set arg fail");
        }

        clEnqueueWriteBuffer(env.queue, env.clMem[kernelID][0], CL_TRUE, 0, globalSize * sizeof(int), &activeNodes[0], 0, nullptr, nullptr);
        clEnqueueWriteBuffer(env.queue, env.clMem[kernelID][1], CL_TRUE, 0, group * sizeof(int), &subSum[0], 0, nullptr, nullptr);

        env.errorCheck(clEnqueueNDRangeKernel(env.queue, env.kernels[kernelID], 1, NULL, &globalSize, &localSize, 0, NULL, NULL),
            "Can not run Gather kernel");

        env.errorCheck(clEnqueueReadBuffer(env.queue, env.clMem[kernelID][1], CL_TRUE, 0, group * sizeof(int), &subSum[0], 0, NULL, NULL),
            "Can not reading result buffer");

        int sum = 0;
        for (int i = 0; i < group; ++i) {
            sum += subSum[i];
        }
        return sum;
    }

	int MemSpace = 0;
	Graph graph;
	Env env;
};

int main(int argc, char **argv){
   std::vector<std::string> filePath{
    "/home/fpga/HGC/data/testGraph.txt",
    "/home/fpga/HGC/data/10kV_100kE.txt",
    "/home/fpga/HGC/data/soc-pokec-relationships.txt",
    "/home/fpga/HGC/data/web-BerkStan.txt",
    "/home/fpga/HGC/data/amazon0601.txt",
    "/home/fpga/HGC/data/roadNet-PA.txt",
    "/home/fpga/HGC/data/wiki-topcats.txt"};
    std::vector<int> initNode{0,0,46,254913,0,0,46};
    std::vector<int> iter{0,0,13,198,36,542,22};
    int ID = argv[1][0] - '0';
    Kruskal kruskal = Kruskal(filePath[ID],initNode[ID],2,iter[ID]);
    return 0;
}
