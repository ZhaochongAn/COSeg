#ifndef _BALL_QUERY_CUDA_KERNEL
#define _BALL_QUERY_CUDA_KERNEL
#include <vector>
#include <torch/serialize/tensor.h>
#include <ATen/cuda/CUDAContext.h>

void ball_query_cuda(int m, float radius, int nsample, at::Tensor xyz_tensor, at::Tensor new_xyz_tensor, at::Tensor offset_tensor, at::Tensor new_offset_tensor, at::Tensor idx_tensor, at::Tensor dist2_tensor);

#ifdef __cplusplus
extern "C" {
#endif

void ball_query_cuda_launcher(int m, float radius, int nsample, const float *xyz, const float *new_xyz, const int *offset, const int *new_offset, int *idx, float *dist2);

#ifdef __cplusplus
}
#endif
#endif
