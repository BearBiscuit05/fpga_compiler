__kernel void GenMerge(__global int*restrict srcs, __global int*restrict dsts, __global int*restrict weights,
    __global int*restrict active, __global int*restrict mValue, __global int*restrict vValue)
{
    size_t index = get_global_id(0);
    if (active[srcs[index]] == 1) {
        atomic_min(&mValue[dsts[index]], vValue[srcs[index]]);
    }
    barrier(CLK_GLOBAL_MEM_FENCE);
}


__kernel void Apply(__global int*restrict active, __global int*restrict mValues, __global int*restrict vValues)
{
    size_t index = get_global_id(0);
    if (mValues[index] < vValues[index]) {
        vValues[index] = mValues[index];
        active[index] = 1;
    }
    else {
        active[index] = 0;
    }
}

__kernel void MergeGraph(__global int*restrict active_1, __global int*restrict active_2, __global int*restrict distance_1, __global int*restrict distance_2)
{
    size_t index = get_global_id(0);
    active_1[index] |= active_2[index];
    distance_1[index] = min(distance_1[index], distance_2[index]);
}

__kernel void Gather(__global int*restrict input, __global int*restrict output, __local int*restrict cache)
{
    int lid = get_local_id(0);
    int bid = get_group_id(0);
    int gid = get_global_id(0);
    int localSize = get_local_size(0);

    cache[lid] = input[gid];
    barrier(CLK_LOCAL_MEM_FENCE);

    for (int i = localSize >> 1; i > 0; i >>= 1)
    {
        if (lid < i)
        {
            cache[lid] += cache[lid + i];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (lid == 0) {
        output[bid] = cache[0];
    }
}

