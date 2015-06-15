template <int BLOCK_THREADS>
__global__ void sgpu_SpGEMM_fp1(const int IA[], const int JA[], const QValue A[],
    const int IB[], const int JB[], const QValue B[],
    const int dgqueue[], const int gcount,
    const int m, const int n,
    const int IC[], int JC[], QValue C[]) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  for (int q = tid; q < gcount; q += blockDim.x * gridDim.x) {
    int rowId = dgqueue[q];
    //if (rowId == 879) printf("row879 is fp1\n");
    int *iJC = JC + IC[rowId];
    QValue *iC = C + IC[rowId];
    for (int ap = IA[rowId]; ap < IA[rowId + 1]; ++ap) {
      int a = JA[ap];
      QValue Aap = A[ap];
      for (int bp = IB[a]; bp < IB[a + 1]; ++bp) {
       int b = JB[bp];
       iJC[0] = b;
       iC[0] = Aap * B[bp];
      }
    }
  }
}

template <int BLOCK_THREADS>
__global__ void sgpu_SpGEMM_fp2(const int IA[], const int JA[], const QValue A[],
    const int IB[], const int JB[], const QValue B[],
    const int dgqueue[], const int gcount,
    const int m, const int n,
    const int* IC, int JC[], QValue C[]) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  __shared__ int iJCs[BLOCK_THREADS][2];
  __shared__ QValue iCs[BLOCK_THREADS][2];
  int *iJC = iJCs[threadIdx.x];
  QValue *iC = iCs[threadIdx.x];
  for (int q = tid; q < gcount; q += blockDim.x * gridDim.x) {
    int rowId = dgqueue[q];
    //if (rowId == 879) printf("row879 is fp2\n");
    int count = 0;
    for (int ap = IA[rowId]; ap < IA[rowId + 1]; ++ap) {
      int a = JA[ap];
      QValue vA = A[ap];
      for (int bp = IB[a]; bp < IB[a + 1]; ++bp) {
       int b = JB[bp];
       iJC[count] = b;
       iC[count++] = vA * B[bp];
      }
    }
    //if (count != 2) printf("rowId %d in fp2 but count=%d\n", rowId, count);
    JC[IC[rowId]] = iJC[0];
    C[IC[rowId]] = iC[0];
    if (iJC[0] != iJC[1]) {
      JC[IC[rowId] + 1] = iJC[1];
      C[IC[rowId] + 1] = iC[1];
    } else {
      C[IC[rowId]] += iC[1];
    }
  }
}

template <int BLOCK_THREADS>
__global__ void sgpu_SpGEMM_a1(const int IA[], const int JA[], const QValue A[],
    const int IB[], const int JB[], const QValue B[],
    const int dgqueue[], const int gcount,
    const int m, const int n,
    const int IC[], int JC[], QValue C[]) {
  __shared__ unsigned keys[BLOCK_THREADS];
  __shared__ int rowIds[BLOCK_THREADS];
  //half warp
  const int HWARPS_PER_BlOCK = BLOCK_THREADS / 16;
  const int hwarpId = threadIdx.x / 16;
  const int hlaneId = threadIdx.x % 16;
  __shared__ int counts[HWARPS_PER_BlOCK];
  if (hlaneId == 0) {
    counts[hlaneId] = 0;
  }
  //warp
  const int WARPS_PER_BlOCK = BLOCK_THREADS / 32;
  const int warpId = threadIdx.x / 32;
  const int laneId = threadIdx.x % 32;
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  for (int q = tid; __syncthreads_or(q < gcount); q += blockDim.x * gridDim.x) {
    int predicate = (q < gcount);
    int rowId = predicate ? dgqueue[q] : -1;
    keys[threadIdx.x] = predicate? IA[rowId + 1] - IA[rowId] : -1;
    rowIds[threadIdx.x] = rowId;
    unsigned le4 = partition_by_bound(keys, rowIds, 4);
    unsigned le32 = partition_by_bound(keys, rowIds, 32);
    unsigned le128 = partition_by_bound(keys, rowIds, 128);
    unsigned total = min(gcount + threadIdx.x - q, blockDim.x);
    __syncthreads();
    for (unsigned t = threadIdx.x; t < le4; t += blockDim.x) {
      const int rowId = rowIds[t];
      int *iJC = JC + IC[rowId];
      QValue *iC = C + IC[rowId];
      int count = 0;
      for (int ap = IA[rowId]; ap < IA[rowId + 1]; ++ap) {
        int a = JA[ap];
        for (int bp = IB[a]; bp < IB[a + 1]; ++bp) {
          int b = JB[bp];
          iJC[count] = b;
          iC[count++] = A[ap] * B[bp];
        }
      }
    }
    __syncthreads();
    for (unsigned t = le4 + hwarpId; t < le32; t += HWARPS_PER_BlOCK) {
      const int rowId = rowIds[t];
      int *iJC = JC + IC[rowId];
      QValue *iC = C + IC[rowId];
      for (int ap = IA[rowId]; ap < IA[rowId + 1]; ++ap) {
        int a = JA[ap];
        for (int bp = IB[a] + hlaneId; bp < IB[a + 1]; bp += 16) {
          int b = JB[bp];
          int count = bp - IB[a] + counts[hwarpId];
          iJC[count] = b;
          iC[count] = A[ap] * B[bp];
        }
        __syncthreads();
        if (hwarpId == 0)
          counts[hwarpId] += IB[a + 1] - IB[a];
      }
      if (hlaneId == 0) {
        counts[hwarpId] = 0;
      }
      __syncthreads();
    }
    __syncthreads();
    for (unsigned t = le32 + warpId; t < le128; t += WARPS_PER_BlOCK) {
      const int rowId = rowIds[t];
      int *iJC = JC + IC[rowId];
      QValue *iC = C + IC[rowId];
      for (int ap = IA[rowId]; ap < IA[rowId + 1]; ++ap) {
        int a = JA[ap];
        for (int bp = IB[a] + laneId; bp < IB[a + 1]; bp += 32) {
          int b = JB[bp];
          int count = bp - IB[a] + counts[warpId];
          iJC[count] = b;
          iC[count] = A[ap] * B[bp];
          //__syncthreads();
        }
        __syncthreads();
        if (laneId == 0)
          counts[warpId] += IB[a + 1] - IB[a];
      }
      if (laneId == 0) {
        counts[warpId] = 0;
      }
      __syncthreads();
    }
    //block
    __syncthreads();
    for (unsigned t = le128; t < total; ++t) {
      const int rowId = rowIds[t];
      int *iJC = JC + IC[rowId];
      QValue *iC = C + IC[rowId];
      for (int ap = IA[rowId]; ap < IA[rowId + 1]; ++ap) {
        int a = JA[ap];
        for (int bp = IB[a] + threadIdx.x; bp < IB[a + 1]; bp += blockDim.x) {
          int b = JB[bp];
          int count = bp - IB[a] + counts[0];
          iJC[count] = b;
          iC[count] = A[ap] * B[bp];
        }
        __syncthreads();
        if (threadIdx.x == 0) counts[0] += IB[a + 1] + IB[a];
      }
    }
  }
}

template <int BLOCK_THREADS, int HPRIME>
__global__ void sgpu_SpGEMM_fpl4(const int IA[], const int JA[], const QValue A[],
    const int IB[], const int JB[], const QValue B[],
    const int dgqueue[], const int gcount,
    const int m, const int n,
    const int* IC, int JC[], QValue C[]) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  __shared__ int iJCs[BLOCK_THREADS][HPRIME];
  __shared__ QValue iCs[BLOCK_THREADS][HPRIME];
  int *iJC = iJCs[threadIdx.x];
  QValue *iC = iCs[threadIdx.x];
  iJC[3] = INT_MAX;
  for (int q = tid; q < gcount; q += blockDim.x * gridDim.x) {
    int rowId = dgqueue[q];
    int count = 0;
    for (int ap = IA[rowId]; ap < IA[rowId + 1]; ++ap) {
      int a = JA[ap];
      QValue vA = A[ap];
      for (int bp = IB[a]; bp < IB[a + 1]; ++bp) {
       int b = JB[bp];
       iJC[count] = b;
       iC[count++] = vA * B[bp];
      }
    }
    if (count == 3) kvSort3(&iJC[0], &iC[0], &iJC[1], &iC[1], &iJC[2], &iC[2]);
    else kvSort4(&iJC[0], &iC[0], &iJC[1], &iC[1], &iJC[2], &iC[2], &iJC[3], &iC[3]);
    int s = 0;
    for (int y = 1; y < count; ++y) {
      if (iJC[s] == iJC[y]) iC[s] += iC[y];
      else {
        ++s;
        iJC[s]  = iJC[y];
        iC[s] = iC[y];
      }
    }
    ++s;
    for (int x = 0, cp = IC[rowId]; x < s; ++x) {
      JC[cp] = iJC[x];
      C[cp] = iC[x];
    }
  }
}

#include "casHash.cuh"

template<class T, int WARP_SIZE>
__device__ T warp_plus_scan(T x[][WARP_SIZE]) {
  const unsigned warpId = threadIdx.x / WARP_SIZE;
  const unsigned laneId = threadIdx.x % WARP_SIZE;
  //const unsigned WARPS_PER_BlOCK = blockDim.x / WARP_SIZE;
  //unsigned int len = WARP_SIZE;
  for (unsigned offset = 1; offset < WARP_SIZE; offset *= 2) {
    T t;
    if (laneId >= offset)
      t = x[warpId][laneId - offset];
    __syncthreads();
    if (laneId >= offset)
      x[warpId][laneId] += t;
    __syncthreads();
  }
  return x[warpId][laneId];
}

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 200)
#undef assert
#define assert(arg)
#endif

template <int BLOCK_THREADS, int WARP_SIZE, int HPRIME>
__global__ void sgpu_SpGEMM_mid(const int IA[], const int JA[], const QValue A[],
    const int IB[], const int JB[], const QValue B[],
    const int drowIds[], const int gcount,
    const int m, const int n,
    const int* IC, int JC[], QValue C[]) {
  const int WARPS_PER_BlOCK = BLOCK_THREADS / WARP_SIZE;
  __shared__ int hbs[WARPS_PER_BlOCK][HPRIME];
  __shared__ QValue hxs[WARPS_PER_BlOCK][HPRIME];
  __shared__ int counts[WARPS_PER_BlOCK][WARP_SIZE];
  const int warpId = threadIdx.x / WARP_SIZE;
  const int gwarpId = warpId + blockIdx.x * WARPS_PER_BlOCK;
  const int laneId = threadIdx.x % WARP_SIZE;
  for (int q = gwarpId; q < gcount; q += WARPS_PER_BlOCK * gridDim.x) {
    int rowId = drowIds[q];
    for (int z = laneId; z < HPRIME; z += WARP_SIZE) {
      hbs[warpId][z] = -1;
      hxs[warpId][z] = 0.0;
    }
    __syncthreads();
    for (int ap = IA[rowId]; ap < IA[rowId + 1]; ++ap) {
      const int a = JA[ap];
      const QValue Aap = A[ap];
      for (int bp = IB[a] + laneId; bp < IB[a + 1]; bp += WARP_SIZE) {
        const int b = JB[bp];
        int pos = -1;
        int index = hashCASAdd2(hbs[warpId], HPRIME, b, &pos);
        if (index == -1) {
          hxs[warpId][pos] = Aap * B[bp];
        } else {
          hxs[warpId][pos] += Aap * B[bp];
        }
      }
      __syncthreads();
    }
    __syncthreads();
    counts[warpId][laneId] = 0;
    for (int z = laneId; z < HPRIME; z += WARP_SIZE) {
      if (hbs[warpId][z] != -1) {
        ++counts[warpId][laneId];
      }
    }
    __syncthreads();
    int cwl = warp_plus_scan<int, WARP_SIZE>(counts);
    int cp = IC[rowId];
    if (laneId != 0) cp += counts[warpId][laneId - 1];
    for (int z = laneId; z < HPRIME; z += WARP_SIZE) {
      if (hbs[warpId][z] != -1) {
        JC[cp] = hbs[warpId][z];
        C[cp++] = hxs[warpId][z];
      }
    }
  }
}


__global__ void gpuFlops(const int m, const int* dIA, const int *dJA, const int* dIB, long *dflops) {
  for (int i = threadIdx.x + blockIdx.x * blockDim.x; i < m; ++i) {
    long tmpRowFlops = 0;
    for (int jp = dIA[i]; jp < dIA[i + 1]; ++jp) {
      int j = dJA[jp];
      int BrowFlops = dIB[j + 1] - dIB[j];
      tmpRowFlops += BrowFlops;
    }
    dflops[i] = tmpRowFlops;
  }
}

__device__ inline int dqueueId(int x) {
  assert (x > 0);
  if (x == 0) return 0;
  else if (x == 1) return 1;
  int ret = 2;
  int up = 2;
  for (up = 2; ; up *= 2, ++ret) {
    if (x <= up) return ret;
  }
  //return -1;
}

template <int BLOCK_THREADS>
__global__ void gpuClassifyFlops(const int m, const long *dflops, const int* dIA, int *dqueue, int *dv) {
  __shared__ int counts[BLOCK_THREADS + 1];
  if (threadIdx.x < 64) counts[threadIdx.x + 1] = 0;
  __syncthreads();
  for (int i = threadIdx.x + blockIdx.x * blockDim.x; i < m; ++i) {
    if (dflops[i] == 0) continue;
    int q = 0;
    if (dIA[i + 1] - dIA[i] > 1) q = dqueueId(dflops[i]);
    atomicAdd(&counts[q], 1);
  }
  __syncthreads();
  plus_scan(counts);
}

void computeDv(const CSR &dA, const CSR &dB, int** dvp, int** dqueuep) {
  long *dflops = NULL;
  const int m = dA.rows;
  HANDLE_ERROR(cudaMalloc((void**)&dflops, m * sizeof(int)));
  const unsigned NBLOCKS = qmin(65535, m);
  gpuFlops<<<NBLOCKS, 512>>>(m, dA.rowPtr, dA.colInd, dB.rowPtr, dflops);
  HANDLE_ERROR(cudaMalloc((void**)dqueuep, m * sizeof(int)));
  HANDLE_ERROR(cudaMalloc((void**)dvp, 64 * sizeof(int)));
  HANDLE_ERROR(cudaMemset(*dvp, 0, 64 * sizeof(QValue)));
}

