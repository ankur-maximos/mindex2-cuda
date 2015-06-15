#include "bitonic_sort.cuh"

// Trying out ESC algo for our
template <int BLOCK_THREADS>
__global__ void sgpu_CSR_IC_nnzC_mid4(const int IA[], const int JA[],
    const int IB[], const int JB[],
    const int drowIds[], const int gcount,
    const int m, const int n, int* IC) {
  
  __shared__ int   s_key[2 * BLOCK_THREADS];
  __shared__ short s_scan[2 * BLOCK_THREADS + 1];

  const int local_id = threadIdx.x;
  const int group_id = blockIdx.x;
  const int local_size = blockDim.x;
  const int local_id_halfwidth = local_id + local_size;

  int colID_A;
  int start_row_B;
  int end_row_B;  
  int strideB;  

  for(int q = group_id; q < gcount; q += gridDim.x) {
    int rowId = drowIds[q];
    s_key[local_id] = INT_MAX;
    s_key[local_id + BLOCK_THREADS] = INT_MAX;
    int local_counter = 0;
    for(int ap = IA[rowId]; ap < IA[rowId + 1]; ap++) {	
	colID_A = JA[ap];
	start_row_B = IB[colID_A];
	end_row_B = IB[colID_A + 1];
  	
	strideB = end_row_B - start_row_B;
	
	if(local_id < strideB) {
	     s_key[local_counter + local_id] = JB[start_row_B + local_id];
	}
	
	if(local_id_halfwidth < strideB) {
	     s_key[local_counter + local_id_halfwidth] = JB[start_row_B + local_id_halfwidth];
	}
	local_counter += strideB;	
    }
    __syncthreads();
    
    oddeven(s_key, blockDim.x * 2);
    __syncthreads();
    bool duplicate = 0;
    bool duplicate_halfwidth = 0;
    if (local_id < local_counter && local_id > 0) {
	duplicate = (s_key[local_id] != s_key[local_id-1]);	
    } 
    if (local_id_halfwidth < local_counter) {
	duplicate_halfwidth = (s_key[local_id_halfwidth] != s_key[local_id_halfwidth -1]);
    }
    s_scan[local_id] = duplicate;
    s_scan[local_id_halfwidth] = duplicate_halfwidth;
    __syncthreads();
    
    for(int offset = blockDim.x; offset > 0; offset >>= 1) {
	if(local_id < offset) {
	   s_scan[local_id] += s_scan[local_id + offset];
	}
	__syncthreads();
    } 
    if(threadIdx.x == 0) IC[rowId] = s_scan[0] + 1; 
  }
 }


template <int BLOCK_THREADS>
__global__ void sgpu_CSR_IC_nnzC_mid2(const int IA[], const int JA[],
    const int IB[], const int JB[],
    const int drowIds[], const int gcount,
    const int m, const int n, int* IC) {
  __shared__ int   s_key[2 * BLOCK_THREADS];
  __shared__ short s_scan[2 * BLOCK_THREADS + 1];
 
  __shared__ int as[BLOCK_THREADS];

  const int local_id = threadIdx.x;
  const int local_id_halfwidth = threadIdx.x + blockDim.x;

  for (int q = blockIdx.x; q < gcount; q += gridDim.x) {
    int rowId = drowIds[q];
    s_key[threadIdx.x] = INT_MAX;
    s_key[threadIdx.x + BLOCK_THREADS] = INT_MAX;
    int local_counter = 0;
    for (int ap = IA[rowId] + threadIdx.x; __syncthreads_or(ap < IA[rowId + 1]); ap += blockDim.x) {
      int predicate = (ap < IA[rowId + 1]);
      int a = predicate ? JA[ap] : -1; 
      as[threadIdx.x] = a;
      unsigned total = min(IA[rowId + 1] + threadIdx.x - ap, blockDim.x);
      __syncthreads();
      for (int ap = 0; ap < total; ++ap) {
        int a = as[ap];
        for (int bp = IB[a] + threadIdx.x; bp < IB[a + 1]; bp += blockDim.x) {
          int b = JB[bp];
          s_key[local_counter + bp - IB[a]] = b;
        }   
        const int strideB = IB[a + 1] - IB[a];
        local_counter += strideB;
        __syncthreads();
      }   
    }   
    //__syncthreads();
    oddeven(s_key, blockDim.x * 2); 
    __syncthreads();
    bool duplicate = 0;
    bool duplicate_halfwidth = 0;
    // generate bool value in registers
    if (local_id < local_counter && local_id > 0)
      duplicate = (s_key[local_id] != s_key[local_id - 1]);
    if (local_id_halfwidth < local_counter)
      duplicate_halfwidth = (s_key[local_id_halfwidth] != s_key[local_id_halfwidth - 1]);
    s_scan[local_id]                    = duplicate;
    s_scan[local_id_halfwidth]          = duplicate_halfwidth;
    __syncthreads();
    for (int offset = blockDim.x;
        offset > 0;
        offset >>= 1) {
      if(threadIdx.x < offset) {
        s_scan[threadIdx.x] += s_scan[threadIdx.x + offset];
      }   
      // wait until all threads in the block have
      // updated their partial sums
      __syncthreads();
    }   
    if (threadIdx.x == 0) IC[rowId] = s_scan[0] + 1;
  }
}

/*
template <int BLOCK_THREADS>
__global__ void sgpu_CSR_IC_nnzC_mid4(const int IA[], const int JA[],
    const int IB[], const int JB[],
    const int drowIds[], const int gcount,
    const int m, const int n, int* IC) {

  __shared__ int   s_key[2 * BLOCK_THREADS];
  __shared__ short s_scan[2 * BLOCK_THREADS + 1];

  __shared__ int as[BLOCK_THREADS];

  const int local_id = threadIdx.x;
  const int group_id = blockIdx.x;
  const int local_size = blockDim.x;
  const int local_id_halfwidth = local_id + local_size;

  int colID_A;
  int start_row_B;
  int end_row_B;
  int strideB;

  for(int q = group_id; q < gcount; q += gridDim.x) {
    int rowId = drowIds[q];
    s_key[local_id] = INT_MAX;
    s_key[local_id + BLOCK_THREADS] = INT_MAX;
    int local_counter = 0;
    for(int ap = IA[rowId + local_id]; __syncthreads_or(ap < IA[rowId + 1]); ap += blockDim.x) {
        int predicate = (ap < IA[rowId + 1]);
        colID_A = predicate ? JA[ap] : -1;
        as[local_id] = colID_A;

	unsigned total = min(IA[rowId + 1] + threadIdx.x - ap, blockDim.x);
	__syncthreads();

        for(int ap = 0; ap < total; ++ap) {
	int a = as[ap];
        start_row_B = IB[a];
        end_row_B = IB[a + 1];

        strideB = end_row_B - start_row_B;

        if(local_id < strideB) {
             s_key[local_counter + local_id] = JB[start_row_B + local_id];
        }

        if(local_id_halfwidth < strideB) {
             s_key[local_counter + local_id_halfwidth] = JB[start_row_B + local_id_halfwidth];
        }
        
        local_counter += strideB;
	__syncthreads();
	}
    }
    __syncthreads();

    oddeven(s_key, blockDim.x * 2);
    __syncthreads();
    bool duplicate = 0;
    bool duplicate_halfwidth = 0;
    if (local_id < local_counter && local_id > 0) {
        duplicate = (s_key[local_id] != s_key[local_id-1]);
    }
    if (local_id_halfwidth < local_counter) {
        duplicate_halfwidth = (s_key[local_id_halfwidth] != s_key[local_id_halfwidth -1]);
    }
    s_scan[local_id] = duplicate;
    s_scan[local_id_halfwidth] = duplicate_halfwidth;
    __syncthreads();

    for(int offset = blockDim.x; offset > 0; offset >>= 1) {
        if(local_id < offset) {
           s_scan[local_id] += s_scan[local_id + offset];
        }
        __syncthreads();
    }
    if(threadIdx.x == 0) IC[rowId] = s_scan[0] + 1;
  }
 } 
*/

