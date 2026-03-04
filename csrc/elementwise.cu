#include "common.cuh"

// Element-wise add: out = a + b (BF16×4 vectorized)
__global__ void add_kernel(const __nv_bfloat16 *__restrict__ a,
                           const __nv_bfloat16 *__restrict__ b, __nv_bfloat16 *__restrict__ out,
                           int n) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int n4 = n / 4;

  if (tid < n4) {
    const uint2 *a_vec = reinterpret_cast<const uint2 *>(a);
    const uint2 *b_vec = reinterpret_cast<const uint2 *>(b);
    uint2 *out_vec = reinterpret_cast<uint2 *>(out);

    uint2 av = a_vec[tid];
    uint2 bv = b_vec[tid];
    __nv_bfloat162 a_lo = *reinterpret_cast<__nv_bfloat162 *>(&av.x);
    __nv_bfloat162 a_hi = *reinterpret_cast<__nv_bfloat162 *>(&av.y);
    __nv_bfloat162 b_lo = *reinterpret_cast<__nv_bfloat162 *>(&bv.x);
    __nv_bfloat162 b_hi = *reinterpret_cast<__nv_bfloat162 *>(&bv.y);

    __nv_bfloat162 r_lo, r_hi;
    r_lo.x = __float2bfloat16(__bfloat162float(a_lo.x) + __bfloat162float(b_lo.x));
    r_lo.y = __float2bfloat16(__bfloat162float(a_lo.y) + __bfloat162float(b_lo.y));
    r_hi.x = __float2bfloat16(__bfloat162float(a_hi.x) + __bfloat162float(b_hi.x));
    r_hi.y = __float2bfloat16(__bfloat162float(a_hi.y) + __bfloat162float(b_hi.y));

    uint2 result;
    result.x = *reinterpret_cast<unsigned int *>(&r_lo);
    result.y = *reinterpret_cast<unsigned int *>(&r_hi);
    out_vec[tid] = result;
  }

  // Scalar tail
  int scalar_idx = n4 * 4 + (blockIdx.x * blockDim.x + threadIdx.x);
  if (scalar_idx >= n4 * 4 && scalar_idx < n) {
    out[scalar_idx] = __float2bfloat16(__bfloat162float(a[scalar_idx]) + __bfloat162float(b[scalar_idx]));
  }
}

// Copy kernel (for slicing)
__global__ void copy_kernel(const __nv_bfloat16 *__restrict__ src,
                            __nv_bfloat16 *__restrict__ dst, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    dst[idx] = src[idx];
  }
}

// Softmax: out = softmax(x)
// Single block kernel for small vectors (attention scores)
__global__ void softmax_kernel(const __nv_bfloat16 *__restrict__ x,
                               __nv_bfloat16 *__restrict__ out, int n) {
  extern __shared__ float shared[];

  int tid = threadIdx.x;
  int stride = blockDim.x;

  // Find max (for numerical stability)
  float local_max = -INFINITY;
  for (int i = tid; i < n; i += stride) {
    local_max = fmaxf(local_max, __bfloat162float(x[i]));
  }
  shared[tid] = local_max;
  __syncthreads();

  // Reduce max
  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s) {
      shared[tid] = fmaxf(shared[tid], shared[tid + s]);
    }
    __syncthreads();
  }
  float max_val = shared[0];
  __syncthreads();

  // Compute exp(x - max) and sum
  float local_sum = 0.0f;
  for (int i = tid; i < n; i += stride) {
    float exp_val = expf(__bfloat162float(x[i]) - max_val);
    out[i] = __float2bfloat16(exp_val);
    local_sum += exp_val;
  }
  shared[tid] = local_sum;
  __syncthreads();

  // Reduce sum
  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s) {
      shared[tid] += shared[tid + s];
    }
    __syncthreads();
  }
  float sum = shared[0];
  __syncthreads();

  // Normalize
  float inv_sum = 1.0f / sum;
  for (int i = tid; i < n; i += stride) {
    out[i] = __float2bfloat16(__bfloat162float(out[i]) * inv_sum);
  }
}

extern "C" {
void add_cuda(const __nv_bfloat16 *a, const __nv_bfloat16 *b, __nv_bfloat16 *out, int n,
              cudaStream_t stream) {
  int block_size = 256;
  int n4 = n / 4;
  int num_blocks = (n4 + block_size - 1) / block_size;
  if (num_blocks == 0) num_blocks = 1;  // handle n < 4
  add_kernel<<<num_blocks, block_size, 0, stream>>>(a, b, out, n);
}

void copy_cuda(const __nv_bfloat16 *src, __nv_bfloat16 *dst, int n, cudaStream_t stream) {
  int block_size = 256;
  int num_blocks = (n + block_size - 1) / block_size;
  copy_kernel<<<num_blocks, block_size, 0, stream>>>(src, dst, n);
}

void softmax_cuda(const __nv_bfloat16 *x, __nv_bfloat16 *out, int n, cudaStream_t stream) {
  int block_size = 256;
  int shared_mem = block_size * sizeof(float);
  softmax_kernel<<<1, block_size, shared_mem, stream>>>(x, out, n);
}
}
