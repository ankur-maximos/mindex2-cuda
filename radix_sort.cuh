template<class T>
__device__ T plus_scan(T *x) {
  unsigned int i = threadIdx.x;
  unsigned int len = blockDim.x;
  unsigned int offset;
  for (offset = 1; offset < len; offset *= 2) {
    T t;
    if (i >= offset)
      t = x[i - offset];
    __syncthreads();
    if (i >= offset)
      x[i] = t + x[i];
    __syncthreads();
  }
  return x[i];
}

template<class T>
__device__ void partition_by_bit(unsigned *keys, T* values, unsigned bit) {
  unsigned int i = threadIdx.x;
  unsigned int size = blockDim.x;
  unsigned k_i = keys[i];
  T v_i = values[i];
  unsigned int p_i = (k_i >> bit) & 1;
  keys[i] = p_i;
  __syncthreads();
  unsigned int T_before = plus_scan(keys);
  unsigned int T_total  = keys[size-1];
  unsigned int F_total  = size - T_total;
  __syncthreads();
  if (p_i) {
    keys[T_before - 1 + F_total] = k_i;
    values[T_before - 1 + F_total] = v_i;
  } else {
    keys[i - T_before] = k_i;
    values[i - T_before] = v_i;
  }
}

template<class T>
__device__ unsigned partition_by_bound(unsigned *keys, T* values, unsigned bound) {
  unsigned i = threadIdx.x;
  unsigned size = blockDim.x;
  __syncthreads();
  unsigned k_i = keys[i];
  T v_i = values[i];
  unsigned int p_i = (k_i > bound);
  keys[i] = p_i;
  __syncthreads();
  unsigned T_before = plus_scan(keys);
  unsigned T_total  = keys[size-1];
  unsigned F_total  = size - T_total;
  __syncthreads();
  if (p_i) {
    keys[T_before - 1 + F_total] = k_i;
    values[T_before - 1 + F_total] = v_i;
  } else {
    keys[i - T_before] = k_i;
    values[i - T_before] = v_i;
  }
  return F_total;
}

/*__device__ void radix_sort(unsigned int *values) {
  int bit;
  for(bit = 0; bit < 32; ++bit) {
    partition_by_bit(values, bit);
    __syncthreads();
  }
}*/
