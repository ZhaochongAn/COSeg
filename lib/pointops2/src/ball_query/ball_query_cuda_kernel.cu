#include "../cuda_utils.h"
#include "ball_query_cuda_kernel.h"


__device__ int get_bt_idx_in_ball_query(int idx, const int *offset)
{
    int i = 0;
    while (1)
    {
        if (idx < offset[i])
            break;
        else
            i++;
    }
    return i;
}


__global__ void ball_query_cuda_kernel(int m, float radius, int nsample, const float *__restrict__ xyz, const float *__restrict__ new_xyz, const int *__restrict__ offset, const int *__restrict__ new_offset, int *__restrict__ idx, float *__restrict__ dist2) {
    // input: xyz (n, 3) new_xyz (m, 3)
    // output: idx (m, nsample) dist2 (m, nsample)
    int pt_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (pt_idx >= m) return;

    new_xyz += pt_idx * 3;
    idx += pt_idx * nsample;
    dist2 += pt_idx * nsample;
    int bt_idx = get_bt_idx_in_ball_query(pt_idx, new_offset);
    int start;
    if (bt_idx == 0)
        start = 0;
    else
        start = offset[bt_idx - 1];
    int end = offset[bt_idx];

    float new_x = new_xyz[0];
    float new_y = new_xyz[1];
    float new_z = new_xyz[2];

    int count = 0;
    float radius2 = radius * radius;

    for(int i = 0; i < nsample; i++){
        idx[i] = -1;
        dist2[i] = 1e10;
    }
    for(int i = start; i < end; i++){
        float x = xyz[i * 3 + 0];
        float y = xyz[i * 3 + 1];
        float z = xyz[i * 3 + 2];
        float d2 = (new_x - x) * (new_x - x) + (new_y - y) * (new_y - y) + (new_z - z) * (new_z - z);
        if (d2 <= radius2){
            idx[count] = i;
            dist2[count] = d2;
            count++;
        }
        if (count >= nsample){
            break;
        }
    }
}


void ball_query_cuda_launcher(int m, float radius, int nsample, const float *xyz, const float *new_xyz, const int *offset, const int *new_offset, int *idx, float *dist2) {
    // input: new_xyz: (m, 3), xyz: (n, 3), idx: (m, nsample)
    dim3 blocks(DIVUP(m, THREADS_PER_BLOCK));
    dim3 threads(THREADS_PER_BLOCK);
    ball_query_cuda_kernel<<<blocks, threads, 0>>>(m, radius, nsample, xyz, new_xyz, offset, new_offset, idx, dist2);
}
