#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/binary_search.h>
#include <thrust/device_vector.h>

#include "CSR.h"
#include "gpus/cuda_handle_error.h"

void outputDeviceLongArray(long *dflops, int m) {
  long *hflops = (long*)malloc(m * sizeof(long));
  HANDLE_ERROR(cudaMemcpy((void*)hflops, (void*) dflops, m * sizeof(long), cudaMemcpyDeviceToHost));
  printf("gpu hflops: ");
  for (int i = 0; i < m; ++i) {
    printf("%ld ", hflops[i]);
  }
  printf("\n");
  free(hflops);
}

void outputDeviceIntArray(const char *msg, int *dia, int m) {
  int *hia = (int*)malloc(m * sizeof(int));
  HANDLE_ERROR(cudaMemcpy((void*)hia, (void*)dia, m * sizeof(int), cudaMemcpyDeviceToHost));
  printf("%s", msg);
  for (int i = 0; i < m; ++i) {
    printf("%d ", hia[i]);
  }
  printf("\n");
  free(hia);
}

int inline qmin(const int a, const int b) {
  if (a < b) return a;
  return b;
}

__device__ inline int dqueueId(long x) {
  //assert (x > 0);
  if (x == 0) return 0;
  else if (x == 1) return 1;
  else if (x > 1024) return 63;
  else if (x > 256 && x <= 512) return 10;
  else if (x > 512 && x <= 1024) return 11;
  int ret = 2;
  int up = 2;
  for (up = 2; ; up *= 2, ++ret) {
    if (x <= up) return ret;
  }
  //return -1;
}

template <int BLOCK_THREADS>
__global__ void gcomputeBinId(const int m, const int* dIA, const int *dJA, const int* dIB,
     int *binIds, int *drowIds) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  for (int i = tid; i < m; i += blockDim.x * gridDim.x) {
    long tmpRowFlops = 0;
    for (int ap = dIA[i]; ap < dIA[i + 1]; ++ap) {
      int a = dJA[ap];
      int BrowFlops = dIB[a + 1] - dIB[a];
      tmpRowFlops += BrowFlops;
    }
    //dflops[i] = tmpRowFlops;
    int q = 0;
    q = dqueueId(tmpRowFlops);
    if (tmpRowFlops == 0) q = 64;
    else {
      int acount = dIA[i + 1] - dIA[i];
      // changing to 512 bin size
      if (tmpRowFlops > 1024  && tmpRowFlops > acount * 0) q = 63;
    }
    //if (q == 0) assert (dIA[i + 1] - dIA[i] == 1);
    __syncthreads();
    binIds[i] = q;
    drowIds[i] = i;
  }
}

thrust::device_vector<int> computeHistogram(int *sortedKeys, int m) {
  thrust::device_ptr<int> sortedKeys_dptr(sortedKeys);
  int num_bins = sortedKeys_dptr[m - 1] + 1;
  thrust::device_vector<int> histogram;
  histogram.resize(num_bins + 1);
  histogram[0] = 0;
  thrust::counting_iterator<int> search_begin(0);
  thrust::upper_bound(sortedKeys_dptr, sortedKeys_dptr + m,
                      search_begin, search_begin + num_bins,
                      histogram.begin() + 1);
  return histogram;
}

/* Classification of flops to bins*/
std::vector<int> gpuFlopsClassify(const CSR &dA, const CSR &dB, int **drowIdsp) {
  const int m = dA.rows;
  int *dbinIds = NULL, *drowIds = NULL;
  HANDLE_ERROR(cudaMalloc((void**)&dbinIds, m * sizeof(int)));
  HANDLE_ERROR(cudaMalloc((void**)&drowIds, m * sizeof(int)));
  //long *dflops = NULL;
  //HANDLE_ERROR(cudaMalloc((void**)&dflops, m * sizeof(long)));
  const unsigned BLOCK_THREADS = 256;
  const unsigned NBLOCKS = qmin(65535, (m + BLOCK_THREADS - 1) / BLOCK_THREADS);
  gcomputeBinId<BLOCK_THREADS><<<NBLOCKS, BLOCK_THREADS>>>(m, dA.rowPtr, dA.colInd, dB.rowPtr, dbinIds, drowIds);
  std::vector<int> v;
  thrust::device_ptr<int> dbinIds_dptr(dbinIds);
  thrust::device_ptr<int> drowIds_dptr(drowIds);
  thrust::stable_sort_by_key(dbinIds_dptr, dbinIds_dptr + m, drowIds_dptr);
  //outputDeviceIntArray("drowIds: ", drowIds, m);
  thrust::counting_iterator<int> search_begin(0);
  thrust::device_vector<int> dhist = computeHistogram(dbinIds, m);
  *drowIdsp = drowIds;
  v.resize(dhist.size());
  thrust::copy(dhist.begin(), dhist.end(), v.begin());
  //printf("vsize=%u\n", v.size());
  if (v.size() == 66) {
    v.pop_back();
  } else if (v.size() < 65) {
    int osize = v.size();
    int back = v.back();
    v.resize(65);
    for (int i = osize; i <= 65; ++i) {
      v[i] = back;
    }
  }
  return v;
}
