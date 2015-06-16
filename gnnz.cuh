#include "tryOutCompute.cu"

const int MAX_NBLOCKS = 2048;
const int GBs = 5.6;
const int MAX_SHARED_IC = 512;

int inline qmin(const int a, const int b) {
  if (a < b) return a;
  return b;
}

int inline qmin3(const int a, const int b, const int c) {
  return qmin(qmin(a, b), c);
}

__global__ void sgpu_CSR_IC_nnzC_a1(const int IA[], const int JA[],
    const int IB[], const int JB[], const int dgqueue[], const int gcount,
    const int m, const int n, int* IC) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  for (int q = tid; q < gcount; q += blockDim.x * gridDim.x) {
    int rowId = dgqueue[q];
    int ap = IA[rowId];
    int a = JA[ap];
    IC[rowId] = IB[a + 1] - IB[a];
  }
}

template <int BLOCK_THREADS>
__global__ void sgpu_CSR_IC_nnzC_fp1(const int IA[], const int JA[],
    const int IB[], const int JB[], const int dgqueue[], const int gcount,
    const int m, const int n, int* IC) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  __shared__ int iJCs[BLOCK_THREADS][2];
  int *iJC = iJCs[threadIdx.x];
  for (int q = tid; q < gcount; q += blockDim.x * gridDim.x) {
    int rowId = dgqueue[q];
    int count = 0;
    for (int ap = IA[rowId]; ap < IA[rowId + 1]; ++ap) {
      int a = JA[ap];
      for (int bp = IB[a]; bp < IB[a + 1]; ++bp) {
       int b = JB[bp];
       iJC[count++] = b;
      }
    }
    //count -= (iJC[0] == iJC[1]);
    IC[rowId] = count;
  }
}

template <int BLOCK_THREADS>
__global__ void sgpu_CSR_IC_nnzC_fp2(const int IA[], const int JA[],
    const int IB[], const int JB[], const int dgqueue[], const int gcount,
    const int m, const int n, int* IC) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  __shared__ int iJCs[BLOCK_THREADS][2];
  int *iJC = iJCs[threadIdx.x];
  for (int q = tid; q < gcount; q += blockDim.x * gridDim.x) {
    int rowId = dgqueue[q];
    int count = 0;
    for (int ap = IA[rowId]; ap < IA[rowId + 1]; ++ap) {
      int a = JA[ap];
      for (int bp = IB[a]; bp < IB[a + 1]; ++bp) {
       int b = JB[bp];
       iJC[count++] = b;
      }
    }
    count -= (iJC[0] == iJC[1]);
    IC[rowId] = count;
  }
}

template <int BLOCK_THREADS>
__global__ void sgpu_CSR_IC_nnzC_fpl4(const int IA[], const int JA[],
    const int IB[], const int JB[], const int dgqueue[], const int gcount,
    const int m, const int n, int* IC) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  __shared__ int iJCs[BLOCK_THREADS][4];
  int *iJC = iJCs[threadIdx.x];
  iJC[3] = INT_MAX;
  for (int q = tid; q < gcount; q += blockDim.x * gridDim.x) {
    int rowId = dgqueue[q];
    int count = 0;
    for (int ap = IA[rowId]; ap < IA[rowId + 1]; ++ap) {
      int a = JA[ap];
      for (int bp = IB[a]; bp < IB[a + 1]; ++bp) {
       int b = JB[bp];
       iJC[count++] = b;
      }
    }
    if (count == 3) Sort3(&iJC[0], &iJC[1], &iJC[2]);
    else Sort4(&iJC[0], &iJC[1], &iJC[2], &iJC[3]);
    count -= (iJC[0] == iJC[1]);
    count -= (iJC[1] == iJC[2]);
    count -= (iJC[2] == iJC[3]);
    IC[rowId] = count;
  }
}

#include "casHash.cuh"

template <int BLOCK_THREADS, int WARP_SIZE, int HPRIME>
__global__ void sgpu_CSR_IC_nnzC_mid(const int IA[], const int JA[],
    const int IB[], const int JB[],
    const int drowIds[], const int gcount,
    const int m, const int n, int* IC) {
  const int WARPS_PER_BlOCK = BLOCK_THREADS / WARP_SIZE;
  __shared__ int hbs[WARPS_PER_BlOCK][HPRIME];
  __shared__ int counts[WARPS_PER_BlOCK][WARP_SIZE];
  const int warpId = threadIdx.x / WARP_SIZE;
  const int gwarpId = warpId + blockIdx.x * WARPS_PER_BlOCK;
  const int laneId = threadIdx.x % WARP_SIZE;
  //int tid = threadIdx.x + blockIdx.x * blockDim.x;
  for (int q = gwarpId; q < gcount; q += WARPS_PER_BlOCK * gridDim.x) {
    int rowId = drowIds[q];
    int count = 0;
    for (int z = threadIdx.x % WARP_SIZE; z < HPRIME; z += WARP_SIZE) hbs[warpId][z] = -1;
    for (int ap = IA[rowId]; ap < IA[rowId + 1]; ++ap) {
      int a = JA[ap];
      for (int bp = IB[a] + laneId; bp < IB[a + 1]; bp += WARP_SIZE) {
        const int b = JB[bp];
        int index = hashCASAdd(hbs[warpId], HPRIME, b);
        if (index == -1) ++count;
      }
    }
    counts[warpId][laneId] = count;
    __syncthreads();
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
      if (laneId < offset)
        counts[warpId][laneId] += counts[warpId][laneId + offset];
      __syncthreads();
    }
    if (laneId == 0) IC[rowId] = counts[warpId][0];
    //atomicAdd(&IC[rowId], count);
    __syncthreads();
  }
}

/*

--- working code commenting to try something

template <int BLOCK_THREADS>
__global__ void sgpu_CSR_IC_nnzC_olarge(const int IA[], const int JA[],
    const int IB[], const int JB[],
    const int drowIds[], const int gcount,
    const int m, const int n, int* IC,
    int *xbs, int *iJCs) {
  __shared__ int as[BLOCK_THREADS];
  int *xb = xbs + blockIdx.x * n;
  int *iJC = iJCs + blockIdx.x * n;
  if (threadIdx.x == 0 && blockIdx.x ==0) {
    IC[m] = 0;
  }
  __shared__ int count;
  if (threadIdx.x == 0) {
    count = 0;
  }
  __syncthreads();
  for (int q = blockIdx.x; q < gcount; q += gridDim.x) {
    int rowId = drowIds[q];
    if (threadIdx.x == 0) IC[rowId] = 0;
    int end_Row = IA[rowId + 1];
    for (int ap = IA[rowId] + threadIdx.x; __syncthreads_or(ap < end_Row); ap += blockDim.x) {
      int predicate = (ap < end_Row);
      int a = predicate ? JA[ap] : -1;
      as[threadIdx.x] = a;
      unsigned total = min(end_Row + threadIdx.x - ap, blockDim.x);
      __syncthreads();
      for (int ap = 0; ap < total; ++ap) {
        int a = as[ap];
        for (int bp = IB[a] + threadIdx.x; bp < IB[a + 1]; bp += blockDim.x) {
          int b = JB[bp];
	  int xbB = xb[b];
          if (xbB == 0) {
            iJC[atomicAdd(&count, 1)] = b;
            xb[b] = true;
          }
        }
        __syncthreads();
      }
    }
    __syncthreads();
    for (int cp = threadIdx.x; cp < count; cp += blockDim.x) {
      int c = iJC[cp];
      xb[c] = 0;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
      IC[rowId] = count;
      count = 0;
    }
  }
}
*/

/*
__inline__ __device__
void binarysearch(int         *key,
                  int          key_input,
                  int          size,
                  bool        *is_new_col)
{
    int start = 0;
    int stop  = size - 1;
    int median;
    int key_median;

    while (stop >= start)
    {    
        median = (stop + start) / 2;
        key_median = key[median];

        if (key_input > key_median)
            start = median + 1;
        else if (key_input < key_median)
            stop = median - 1;
        else     
        { 
            // atomicAdd is not needed since duplicate is not existed in each input row
            //s_val[median] += val_input;
            *is_new_col = 0;
            break;
        }        
    }            
    //return start;
}

*/


template <int BLOCK_THREADS>
__global__ void sgpu_CSR_IC_nnzC_olarge(const int IA[], const int JA[],
    const int IB[], const int JB[],
    const int drowIds[], const int gcount,
    const int m, const int n, int* IC,
    int *xbs, int *iJCs) {
  __shared__ int as[BLOCK_THREADS];
  __shared__ int iJC_shared[MAX_SHARED_IC];

  int c;
  int *xb = xbs + blockIdx.x * n;
  int *iJC = iJCs + blockIdx.x * (n - MAX_SHARED_IC);
  if (threadIdx.x == 0 && blockIdx.x ==0) {
    IC[m] = 0;
  }
  __shared__ int count;
  if (threadIdx.x == 0) {
    count = 0;
  }
  __syncthreads();
  for (int q = blockIdx.x; q < gcount; q += gridDim.x) {
    int rowId = drowIds[q];
    if (threadIdx.x == 0) IC[rowId] = 0;
    int end_Row = IA[rowId + 1];
    for (int ap = IA[rowId] + threadIdx.x; __syncthreads_or(ap < end_Row); ap += blockDim.x) {
      int predicate = (ap < end_Row);
      int a = predicate ? JA[ap] : -1;
      as[threadIdx.x] = a;
      unsigned total = min(end_Row + threadIdx.x - ap, blockDim.x);
      __syncthreads();
      for (int ap = 0; ap < total; ++ap) {
        int a = as[ap];
        for (int bp = IB[a] + threadIdx.x; bp < IB[a + 1]; bp += blockDim.x) {
          int b = JB[bp];
          int xbB = xb[b];
          if (xbB == 0) {
	    int pos = atomicAdd(&count, 1);
	    if(pos < MAX_SHARED_IC) 
		iJC_shared[pos] = b;
	    else
		iJC[pos - MAX_SHARED_IC] = b;
            xb[b] = true;
          }
        }
        __syncthreads();
      }
    }
    __syncthreads();
    for (int cp = threadIdx.x; cp < count; cp += blockDim.x) {
      if(cp < MAX_SHARED_IC) {
      	 c = iJC_shared[cp];
      } else {
	 c = iJC[cp - MAX_SHARED_IC];
      }
      xb[c] = 0;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
      IC[rowId] = count;
      count = 0;
    }
  }
}

/*
 
 ---- continue updating this kernel

template <int BLOCK_THREADS>
__global__ void sgpu_CSR_IC_nnzC_olarge(const int IA[], const int JA[],
    const int IB[], const int JB[],
    const int drowIds[], const int gcount,
    const int m, const int n, int* IC,
    int *iJCs) {
  __shared__ int as[BLOCK_THREADS];
  __shared__ int iJC_shared[MAX_SHARED_IC];

  bool is_new_col;
  int *iJC = iJCs + blockIdx.x * (n - MAX_SHARED_IC);
  if (threadIdx.x == 0 && blockIdx.x ==0) {
    IC[m] = 0;
  }
  __shared__ int count;
  if (threadIdx.x == 0) {
    count = 0;
  }
  __syncthreads();
  for (int q = blockIdx.x; q < gcount; q += gridDim.x) {
    int rowId = drowIds[q];
    if (threadIdx.x == 0) IC[rowId] = 0;
    int end_Row = IA[rowId + 1];
    for (int ap = IA[rowId] + threadIdx.x; __syncthreads_or(ap < end_Row); ap += blockDim.x) {
      int predicate = (ap < end_Row);
      int a = predicate ? JA[ap] : -1;
      as[threadIdx.x] = a;
      unsigned total = min(end_Row + threadIdx.x - ap, blockDim.x);
      __syncthreads();
      for (int ap = 0; ap < total; ++ap) {
        int a = as[ap];
        for (int bp = IB[a] + threadIdx.x; bp < IB[a + 1]; bp += blockDim.x) {
          int b = JB[bp];
 	  is_new_col = 1;
	  if(count < MAX_SHARED_IC)
          	binarysearch(iJC_shared, b, count, &is_new_col);
          else 
 		binarysearch(iJC, b, (count-MAX_SHARED_IC), &is_new_col);
	  //__syncthreads();
	  if(is_new_col) {
       	    int pos = atomicAdd(&count, 1);
	    if(pos < MAX_SHARED_IC) 
		iJC_shared[pos] = b;
	    else
		iJC[pos - MAX_SHARED_IC] = b;
	  }
        }
        __syncthreads();
      }
    }
    __syncthreads();
    if (threadIdx.x == 0) {
      IC[rowId] = count;
      count = 0;
    }
  }
}

*/

/*
template <int BLOCK_THREADS>
__global__ void sgpu_CSR_IC_nnzC_olarge(const int IA[], const int JA[],
    const int IB[], const int JB[],
    const int drowIds[], const int gcount,
    const int m, const int n, int* IC, 
    int *xbs, int *iJCs) {
 // __shared__ int as[BLOCK_THREADS];
  int *xb = xbs + blockIdx.x * n;
  int *iJC = iJCs + blockIdx.x * n;
  if (threadIdx.x == 0 && blockIdx.x ==0) {
    IC[m] = 0;
  }
  __shared__ int count;
  if (threadIdx.x == 0) {
    count = 0;
  }
  __syncthreads();
  for (int q = blockIdx.x; q < gcount; q += gridDim.x) {
    int rowId = drowIds[q];
    if (threadIdx.x == 0) IC[rowId] = 0;
    for (int ap = IA[rowId]; ap < IA[rowId + 1]; ap++) {
      int a = JA[ap]; 
        for (int bp = IB[a] + threadIdx.x; bp < IB[a + 1]; bp += blockDim.x) {
          int b = JB[bp];
          if (xb[b] == 0) {
            iJC[atomicAdd(&count, 1)] = b;
            xb[b] = true;
          } 
 	}  
        __syncthreads();
    }   
    __syncthreads();
    for (int cp = threadIdx.x; cp < count; cp += blockDim.x) {
      int c = iJC[cp];
      xb[c] = 0;
    }   
    __syncthreads();
    if (threadIdx.x == 0) {
      IC[rowId] = count;
      count = 0;
    }   
  }
}
*/

template <int BLOCK_THREADS>
__global__ void sgpu_CSR_IC_nnzC_vlarge(const int IA[], const int JA[],
    const int IB[], const int JB[],
    const int drowIds[], const int gcount,
    const int m, const int n, int* IC,
    int *xbs, int *iJCs) {
  __shared__ unsigned keys[BLOCK_THREADS];
  __shared__ int as[BLOCK_THREADS];
  int *xb = xbs + blockIdx.x * n;
  int *iJC = iJCs + blockIdx.x * n;
  int a_0 = 0;
  int a_1 = 1;
  if (threadIdx.x == 0 && blockIdx.x ==0) {
    IC[m] = 0;
  }
  __shared__ int count;
  if (threadIdx.x == 0) {
    count = 0;
  }
  __syncthreads();
  for (int q = blockIdx.x; q < gcount; q += gridDim.x) {
    int rowId = drowIds[q];
    if (threadIdx.x == 0) IC[rowId] = 0;
    int end_Row = IA[rowId + 1];
    for (int ap = IA[rowId] + threadIdx.x; __syncthreads_or(ap < end_Row); ap += blockDim.x) {
      int predicate = (ap < end_Row);
      int a = predicate ? JA[ap] : -1;
      keys[threadIdx.x] = predicate ? (IB[a+1] - IB[a]) : -1;
      as[threadIdx.x] = a;
      unsigned le4 = partition_by_bound(keys, as, 4);
      unsigned le16 = partition_by_bound(keys, as, 32);
      unsigned le128 = partition_by_bound(keys, as, 64);
      unsigned total = min(IA[rowId + 1] + threadIdx.x - ap, blockDim.x);
      __syncthreads();
      for (int ap = threadIdx.x; ap < le4; ap += blockDim.x) {
        const int a = as[ap];
        for (unsigned bp = IB[a]; bp < IB[a + 1]; ++bp) {
          const int b = JB[bp];
          if (atomicCAS(xb + b, a_0, a_1) == 0) {
            iJC[atomicAdd(&count, 1)] = b;
          }
        }
      }
      __syncthreads();
      //half warp
      const int HWARPS_PER_BlOCK = BLOCK_THREADS / 16;
      const int hwarpId = threadIdx.x / 16;
      const int hlaneId = threadIdx.x % 16;
      for (unsigned atop = le4 + hwarpId; atop < le16; atop += HWARPS_PER_BlOCK) {
        const int a = as[atop];
        for (unsigned bp = IB[a] + hlaneId; bp < IB[a + 1]; bp += 16) {
          const int b = JB[bp];
          if (atomicCAS(xb + b, a_0, a_1) == 0) {
            iJC[atomicAdd(&count, 1)] = b;
          }
        }
      }
      __syncthreads();
      //warp
      const int WARPS_PER_BlOCK = BLOCK_THREADS / 32;
      const int warpId = threadIdx.x / 32;
      const int laneId = threadIdx.x % 32;
      for (unsigned atop = le16 + warpId; atop < le128; atop += WARPS_PER_BlOCK) {
        const int a = as[atop];
        for (unsigned bp = IB[a] + laneId; bp < IB[a + 1]; bp += 32) {
          const int b = JB[bp];
          if (atomicCAS(xb + b, a_0, a_1) == 0) {
            iJC[atomicAdd(&count, 1)] = b;
          }
        }
      }
      //block
      __syncthreads();
      for (int ap = le128; ap < total; ++ap) {
        int a = as[ap];
        for (int bp = IB[a] + threadIdx.x; bp < IB[a + 1]; bp += blockDim.x) {
          int b = JB[bp];
          if (xb[b] == 0) {
            iJC[atomicAdd(&count, 1)] = b;
            xb[b] = true;
          }
        }
        __syncthreads();
      }
    }
    __syncthreads();
    for (int cp = threadIdx.x; cp < count; cp += blockDim.x) {
      int c = iJC[cp];
      xb[c] = 0;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
      IC[rowId] = count;
      count = 0;
    }
  }
}

void gpu_compute_IC(const CSR &dA, const CSR &dB, int *drowIds, const vector<int> &hv, CSR &dC, int *temp_C_128_id, QValue *temp_C_128_val, int *temp_C_256_id, QValue *temp_C_256_val, int *temp_C_512_id, QValue *temp_C_512_val, int *temp_C_1024_id, QValue *temp_C_1024_val) {
  dC.rowPtr = NULL;
  int m = dA.rows;
  //int k = dA.cols;
  int n = dB.cols;

  HANDLE_ERROR(cudaMalloc((void**)&dC.rowPtr, (m + 1) * sizeof(int)));
  HANDLE_ERROR(cudaMemset(dC.rowPtr, 0, (m + 1) * sizeof(int)));
  if (hv.size() > 0 + 1 && hv[1] - hv[0] > 0) { // up to fp0
    const unsigned NTHREADS = 128; const unsigned WARP_SIZE = 1;
    const unsigned WARPS_PER_BLOCK = NTHREADS / WARP_SIZE;
    const unsigned NBLOCKS = qmin(65535, (hv[1] - hv[0] + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK);
    sgpu_CSR_IC_nnzC_a1<<<NBLOCKS, NTHREADS>>>(dA.rowPtr, dA.colInd, dB.rowPtr, dB.colInd, drowIds + hv[0], hv[1] - hv[0], m, n, dC.rowPtr);
    HANDLE_ERROR(cudaGetLastError());
  }

  if (hv.size() > 1 + 1 && hv[2] - hv[1] > 0) { // up to fp1
    const unsigned NTHREADS = 128; const unsigned WARP_SIZE = 1;
    const unsigned WARPS_PER_BLOCK = NTHREADS / WARP_SIZE;
    const unsigned NBLOCKS = qmin(65535, (hv[2] - hv[1] + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK);
    sgpu_CSR_IC_nnzC_fp1<NTHREADS><<<NBLOCKS, NTHREADS>>>(dA.rowPtr, dA.colInd, dB.rowPtr, dB.colInd, drowIds + hv[1], hv[2] - hv[1], m, n, dC.rowPtr);
    HANDLE_ERROR(cudaGetLastError());
  }
  if (hv.size() > 2 + 1 && hv[3] - hv[2] > 0) { // up to fp2
    const unsigned NTHREADS = 128; const unsigned WARP_SIZE = 1;
    const unsigned WARPS_PER_BLOCK = NTHREADS / WARP_SIZE;
    const unsigned NBLOCKS = qmin(65535, (hv[3] - hv[2] + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK);
    sgpu_CSR_IC_nnzC_fp2<NTHREADS><<<NBLOCKS, NTHREADS>>>(dA.rowPtr, dA.colInd, dB.rowPtr, dB.colInd, drowIds + hv[2], hv[3] - hv[2], m, n, dC.rowPtr);
    HANDLE_ERROR(cudaGetLastError());
  }
  if (hv.size() > 3 + 1 && hv[4] - hv[3] > 0) { // up to fp4
    const unsigned NTHREADS = 128; const unsigned WARP_SIZE = 1;
    const unsigned WARPS_PER_BLOCK = NTHREADS / WARP_SIZE;
    const unsigned NBLOCKS = qmin(65535, (hv[4] - hv[3] + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK);
    sgpu_CSR_IC_nnzC_fpl4<NTHREADS><<<NBLOCKS, NTHREADS>>>(dA.rowPtr, dA.colInd, dB.rowPtr, dB.colInd, drowIds + hv[3], hv[4] - hv[3], m, n, dC.rowPtr);
    HANDLE_ERROR(cudaGetLastError());
  }
  // 0 IA[i+1] - IA[i]=1   1 fp--1 2--fp2 3--fpl4 4--fpl8 5--fpl16 6--fpl32
  if (hv.size() > 4 + 1 && hv[5] - hv[4] > 0) { // up to fp8
    const unsigned NTHREADS = 128; const unsigned WARP_SIZE = 4;
    const unsigned WARPS_PER_BLOCK = NTHREADS / WARP_SIZE;
    const unsigned NBLOCKS = qmin(65535, (hv[5] - hv[4] + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK);
    sgpu_CSR_IC_nnzC_mid<NTHREADS, WARP_SIZE, 11><<<NBLOCKS, NTHREADS>>>(dA.rowPtr, dA.colInd, dB.rowPtr, dB.colInd, drowIds + hv[4], hv[5] - hv[4], m, n, dC.rowPtr);
    HANDLE_ERROR(cudaGetLastError());
  }
  if (hv.size() > 5 + 1 && hv[6] - hv[5] > 0) { // up to fp16
    const unsigned NTHREADS = 128; const unsigned WARP_SIZE = 8;
    const unsigned WARPS_PER_BLOCK = NTHREADS / WARP_SIZE;
    const unsigned NBLOCKS = qmin(65535, (hv[6] - hv[5] + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK);
    sgpu_CSR_IC_nnzC_mid<NTHREADS, WARP_SIZE, 23><<<NBLOCKS, NTHREADS>>>(dA.rowPtr, dA.colInd, dB.rowPtr, dB.colInd, drowIds + hv[5], hv[6] - hv[5], m, n, dC.rowPtr);
    HANDLE_ERROR(cudaGetLastError());
  }
  if (hv.size() > 6 + 1 && hv[7] - hv[6] > 0) { // up to fp32
    const unsigned NTHREADS = 128; const unsigned WARP_SIZE = 16;
    const unsigned WARPS_PER_BLOCK = NTHREADS / WARP_SIZE;
    const unsigned NBLOCKS = qmin(65535, (hv[7] - hv[6] + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK);
    sgpu_CSR_IC_nnzC_mid<NTHREADS, WARP_SIZE, 53><<<NBLOCKS, NTHREADS>>>(dA.rowPtr, dA.colInd, dB.rowPtr, dB.colInd, drowIds + hv[6], hv[7] - hv[6], m, n, dC.rowPtr);
    HANDLE_ERROR(cudaGetLastError());
  }
  if (hv.size() > 7 + 1 && hv[8] - hv[7] > 0) { // up to fp64
    const unsigned NTHREADS = 128; const unsigned WARP_SIZE = 32;
    const unsigned WARPS_PER_BLOCK = NTHREADS / WARP_SIZE;
    const unsigned NBLOCKS = qmin(65535, (hv[8] - hv[7] + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK);
    sgpu_CSR_IC_nnzC_mid<NTHREADS, WARP_SIZE, 111><<<NBLOCKS, NTHREADS>>>(dA.rowPtr, dA.colInd, dB.rowPtr, dB.colInd, drowIds + hv[7], hv[8] - hv[7], m, n, dC.rowPtr);
    HANDLE_ERROR(cudaGetLastError());
  }

 /*

---- commenting this to try combining compute and structure
  if (hv.size() > 8 + 1 && hv[9] - hv[8] > 0) { // up to fp128
    const unsigned NTHREADS = 128; const unsigned WARP_SIZE = 64;
    const unsigned WARPS_PER_BLOCK = NTHREADS / WARP_SIZE;
    const unsigned NBLOCKS = qmin(65535, (hv[9] - hv[8] + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK);
    sgpu_CSR_IC_nnzC_mid<NTHREADS, WARP_SIZE, 191><<<NBLOCKS, NTHREADS>>>(dA.rowPtr, dA.colInd, dB.rowPtr, dB.colInd, drowIds + hv[8], hv[9] - hv[8], m, n, dC.rowPtr);
    HANDLE_ERROR(cudaGetLastError());
  }
*/

/*

  ----- commenting this to try compute and structure phase together ----

  if (hv.size() > 9 + 1 && hv[10] - hv[9] > 0) { // up to fp256
    const unsigned NTHREADS = 128; 
    //const unsigned WARP_SIZE = 128;
    //const unsigned WARPS_PER_BLOCK = NTHREADS / WARP_SIZE;
    const unsigned NBLOCKS = qmin(65535, hv[10] - hv[9]);
    printf("NBLOCKS9=%u\n", NBLOCKS);
    sgpu_CSR_IC_nnzC_mid4<NTHREADS><<<NBLOCKS, NTHREADS>>>(dA.rowPtr, dA.colInd, dB.rowPtr, dB.colInd, drowIds + hv[9], hv[10] - hv[9], m, n, dC.rowPtr);
    HANDLE_ERROR(cudaGetLastError());
  }
*/
  if (hv.size() > 8 + 1 && hv[9] - hv[8] > 0) { // up to fp128
    const unsigned NTHREADS = 64; 
    // const unsigned WARP_SIZE = 64;
    // const unsigned WARPS_PER_BLOCK = NTHREADS / WARP_SIZE;
    const unsigned NBLOCKS = qmin(65535, hv[9] - hv[8]);
    sgpu_SpGEMM_mix_mid<NTHREADS><<<NBLOCKS, NTHREADS>>>(dA.rowPtr, dA.colInd, dA.values, dB.rowPtr, dB.colInd, dB.values, drowIds + hv[8], hv[9] - hv[8], m, n, dC.rowPtr, temp_C_128_id, temp_C_128_val);
    HANDLE_ERROR(cudaGetLastError());
  }

  if (hv.size() > 9 + 1 && hv[10] - hv[9] > 0) { // up to fp256
    const unsigned NTHREADS = 128; 
    //const unsigned WARP_SIZE = 128;
    //const unsigned WARPS_PER_BLOCK = NTHREADS / WARP_SIZE;
    const unsigned NBLOCKS = qmin(65535, hv[10] - hv[9]);
    printf("NBLOCKS9=%u\n", NBLOCKS);
    sgpu_SpGEMM_mix_mid<NTHREADS><<<NBLOCKS, NTHREADS>>>(dA.rowPtr, dA.colInd, dA.values, dB.rowPtr, dB.colInd, dB.values, drowIds + hv[9], hv[10] - hv[9], m, n, dC.rowPtr,temp_C_256_id,temp_C_256_val);
    HANDLE_ERROR(cudaGetLastError());
  }

/*  
   if (hv.size() > 10 + 1 && hv[11] - hv[10] > 0) { // up to fp512
    const unsigned NTHREADS = 256; 
    //const unsigned WARP_SIZE = 128;
    //const unsigned WARPS_PER_BLOCK = NTHREADS / WARP_SIZE;
    const unsigned NBLOCKS = qmin(65535, hv[11] - hv[10]);
    printf("NBLOCKS9=%u\n", NBLOCKS);
    sgpu_SpGEMM_mix_mid<NTHREADS><<<NBLOCKS, NTHREADS>>>(dA.rowPtr, dA.colInd, dA.values, dB.rowPtr, dB.colInd, dB.values, drowIds + hv[10], hv[11] - hv[10], m, n, dC.rowPtr,temp_C_512_id,temp_C_512_val);
    HANDLE_ERROR(cudaGetLastError());
  } */

  //assert (hv.size() == 65);
  //very large
  const int NTHREADS = 128;
  const double memoryMWordsAvail = GBs * 1000.0 / 4 - ((dA.nnz * 2 + dA.rows) + (dB.nnz * 2 + dB.rows) + dA.rows) / 1000.0 / 1000.0;
  const double blockAvail = memoryMWordsAvail * 1000 * 1000 / (n * 2);
  //printf("memoryMWordsAvail=%lf blockAvail = %lf\n", memoryMWordsAvail, blockAvail);
  const int NBLOCKS10 = qmin3(blockAvail, hv[11] - hv[10], MAX_NBLOCKS);
  const int NBLOCKS11 = qmin3(blockAvail, hv[12] - hv[11], MAX_NBLOCKS);
  const int NBLOCKS12 = qmin3(blockAvail, hv[64] - hv[63], MAX_NBLOCKS);
  int ret = std::max(NBLOCKS10,NBLOCKS11);
  int NBLOCKS = std::max(ret,NBLOCKS12); 
  int *xbs = NULL;
  int *iJCs = NULL;
  if(NBLOCKS > 0) {
     HANDLE_ERROR(cudaMalloc((void**)&xbs, NBLOCKS * n * sizeof(int)));
    HANDLE_ERROR(cudaMemset(xbs, -1, NBLOCKS * n * sizeof(int)));
    }
/*

---  changing this in order to check ----
  const int NBLOCKS = std::max(NBLOCKS10,NBLOCKS12);
  int *xbs = NULL;
  int *iJCs = NULL;
*/
  if (hv.size() > 10 + 1 && hv[11] - hv[10] > 0) { // up to fp512
    //const unsigned NTHREADS = 128;
     
    //const unsigned WARP_SIZE = 128;
    //const unsigned WARPS_PER_BLOCK = NTHREADS / WARP_SIZE;
    //const unsigned NBLOCKS = qmin(65535, hv[11] - hv[10]);
    printf("NBLOCKS10=%u\n", NBLOCKS);
    //int *xbs = NULL;
    
    sgpu_SpGEMM_mix_11<NTHREADS><<<NBLOCKS, NTHREADS>>>(dA.rowPtr, dA.colInd, dA.values, dB.rowPtr, dB.colInd, dB.values, drowIds + hv[10], hv[11] - hv[10], m, n, dC.rowPtr,temp_C_512_id,temp_C_512_val,xbs);
    HANDLE_ERROR(cudaGetLastError());
  }

if (hv.size() > 11 + 1 && hv[12] - hv[11] > 0) { // up to fp512
    const unsigned NTHREADS = 128;
     
    //const unsigned WARP_SIZE = 128;
    //const unsigned WARPS_PER_BLOCK = NTHREADS / WARP_SIZE;
    const unsigned NBLOCKS = NBLOCKS11;
    printf("NBLOCKS11=%u\n", NBLOCKS);
    sgpu_SpGEMM_mix_12<NTHREADS><<<NBLOCKS, NTHREADS>>>(dA.rowPtr, dA.colInd, dA.values, dB.rowPtr, dB.colInd, dB.values, drowIds + hv[11], hv[12] - hv[11], m, n, dC.rowPtr,temp_C_1024_id,temp_C_1024_val,xbs);
    HANDLE_ERROR(cudaGetLastError());
  }


  if (hv[64] - hv[63] > 0) {

    int count_ijc = n - MAX_SHARED_IC; 
     
     if (NBLOCKS > 0) {
        cudaMemset(xbs, 0, NBLOCKS * n * sizeof(int));
        HANDLE_ERROR(cudaMalloc((void**)&iJCs, NBLOCKS * count_ijc * sizeof(int)));
     }
 
    //const int NBLOCKS = NBLOCKS12;
    // const int NBLOCKS = 512;
    printf("nnz olarge memoryMWordsAvail=%lf blockAvail = %lf NBLOCKS=%d hsize=%d\n", memoryMWordsAvail, blockAvail, NBLOCKS, hv[64] - hv[63]);
    sgpu_CSR_IC_nnzC_olarge<NTHREADS><<<NBLOCKS, NTHREADS>>>(dA.rowPtr, dA.colInd, dB.rowPtr, dB.colInd, drowIds + hv[63], hv[64] - hv[63], m, n, dC.rowPtr, xbs, iJCs);
    HANDLE_ERROR(cudaGetLastError());
    //HANDLE_ERROR(cudaFree(iJCs));
    //HANDLE_ERROR(cudaFree(xbs));
  }
  if (NBLOCKS > 0) {
    HANDLE_ERROR(cudaFree(iJCs));
    HANDLE_ERROR(cudaFree(xbs));
  }
}

