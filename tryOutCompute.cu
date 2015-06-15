#include "bitonic_sort.cuh"
#include "gpus/gpu_csr_kernel.h"

__inline__ __device__
void scan_128(volatile short *s_scan)
{
    int ai, bi;
    int baseai = 1 + 2 * threadIdx.x;
    int basebi = baseai + 1;
    short temp;

    if (threadIdx.x < 64) { ai = baseai - 1;     bi = basebi - 1;     s_scan[bi] += s_scan[ai]; }
    __syncthreads();
    if (threadIdx.x < 32) { ai =  2 * baseai - 1;  bi =  2 * basebi - 1;   s_scan[bi] += s_scan[ai]; }
    __syncthreads();
    if (threadIdx.x < 16) { ai =  4 * baseai - 1;  bi =  4 * basebi - 1;   s_scan[bi] += s_scan[ai]; }
    if (threadIdx.x < 8)  { ai = 8 * baseai - 1;  bi = 8 * basebi - 1;   s_scan[bi] += s_scan[ai]; }
    if (threadIdx.x < 4)  { ai = 16 * baseai - 1;  bi = 16 * basebi - 1;   s_scan[bi] += s_scan[ai]; }
    if (threadIdx.x < 2)  { ai = 32 * baseai - 1;  bi = 32 * basebi - 1;   s_scan[bi] += s_scan[ai]; }
    if (threadIdx.x == 0) { s_scan[127] += s_scan[63]; s_scan[128] = s_scan[127]; s_scan[127] = 0; temp = s_scan[63]; s_scan[63] = 0; s_scan[127] += temp; }
    if (threadIdx.x < 2)  { ai = 32 * baseai - 1;  bi = 32 * basebi - 1;   temp = s_scan[ai]; s_scan[ai] = s_scan[bi]; s_scan[bi] += temp;}
    if (threadIdx.x < 4)  { ai = 16 * baseai - 1;  bi = 16 * basebi - 1;   temp = s_scan[ai]; s_scan[ai] = s_scan[bi]; s_scan[bi] += temp;}
    if (threadIdx.x < 8)  { ai = 8 * baseai - 1;  bi = 8 * basebi - 1;   temp = s_scan[ai]; s_scan[ai] = s_scan[bi]; s_scan[bi] += temp;}
    if (threadIdx.x < 16) { ai =  4 * baseai - 1;  bi =  4 * basebi - 1;   temp = s_scan[ai]; s_scan[ai] = s_scan[bi]; s_scan[bi] += temp;}
    if (threadIdx.x < 32) { ai =  2 * baseai - 1;  bi =  2 * basebi - 1;   temp = s_scan[ai]; s_scan[ai] = s_scan[bi]; s_scan[bi] += temp;}
    __syncthreads();
    if (threadIdx.x < 64) { ai = baseai - 1;   bi = basebi - 1;   temp = s_scan[ai]; s_scan[ai] = s_scan[bi]; s_scan[bi] += temp; }
}

__inline__ __device__
void scan_512(volatile short *s_scan)
{
    int ai, bi;
    int baseai = 1 + 2 * threadIdx.x;
    int basebi = baseai + 1;
    short temp;

    if (threadIdx.x < 256) { ai = baseai - 1;     bi = basebi - 1;     s_scan[bi] += s_scan[ai]; }
    __syncthreads();
    if (threadIdx.x < 128) { ai =  2 * baseai - 1;  bi =  2 * basebi - 1;   s_scan[bi] += s_scan[ai]; }
    __syncthreads();
    if (threadIdx.x < 64)  { ai =  4 * baseai - 1;  bi =  4 * basebi - 1;   s_scan[bi] += s_scan[ai]; }
    __syncthreads();
    if (threadIdx.x < 32) { ai =  8 * baseai - 1;  bi =  8 * basebi - 1;   s_scan[bi] += s_scan[ai]; }
    __syncthreads();
    if (threadIdx.x < 16) { ai =  16 * baseai - 1;  bi =  16 * basebi - 1;   s_scan[bi] += s_scan[ai]; }
    if (threadIdx.x < 8)  { ai = 32 * baseai - 1;  bi = 32 * basebi - 1;   s_scan[bi] += s_scan[ai]; }
    if (threadIdx.x < 4)  { ai = 64 * baseai - 1;  bi = 64 * basebi - 1;   s_scan[bi] += s_scan[ai]; }
    if (threadIdx.x < 2)  { ai = 128 * baseai - 1;  bi = 128 * basebi - 1;   s_scan[bi] += s_scan[ai]; }
    if (threadIdx.x == 0) { s_scan[511] += s_scan[255]; s_scan[512] = s_scan[511]; s_scan[511] = 0; temp = s_scan[255]; s_scan[255] = 0; s_scan[511] += temp; }
    if (threadIdx.x < 2)  { ai = 128 * baseai - 1;  bi = 128 * basebi - 1;   temp = s_scan[ai]; s_scan[ai] = s_scan[bi]; s_scan[bi] += temp;}
    if (threadIdx.x < 4)  { ai = 64 * baseai - 1;  bi = 64 * basebi - 1;   temp = s_scan[ai]; s_scan[ai] = s_scan[bi]; s_scan[bi] += temp;}
    if (threadIdx.x < 8)  { ai = 32 * baseai - 1;  bi = 32 * basebi - 1;   temp = s_scan[ai]; s_scan[ai] = s_scan[bi]; s_scan[bi] += temp;}
    if (threadIdx.x < 16) { ai =  16 * baseai - 1;  bi =  16 * basebi - 1;   temp = s_scan[ai]; s_scan[ai] = s_scan[bi]; s_scan[bi] += temp;}
    if (threadIdx.x < 32) { ai =  8 * baseai - 1;  bi =  8 * basebi - 1;   temp = s_scan[ai]; s_scan[ai] = s_scan[bi]; s_scan[bi] += temp;}
    __syncthreads();
    if (threadIdx.x < 64) { ai =  4 * baseai - 1;  bi =  4 * basebi - 1;   temp = s_scan[ai]; s_scan[ai] = s_scan[bi]; s_scan[bi] += temp;}
    __syncthreads();
    if (threadIdx.x < 128) { ai =  2 * baseai - 1;  bi =  2 * basebi - 1;   temp = s_scan[ai]; s_scan[ai] = s_scan[bi]; s_scan[bi] += temp;}
    __syncthreads();
    if (threadIdx.x < 256) { ai = baseai - 1;   bi = basebi - 1;   temp = s_scan[ai]; s_scan[ai] = s_scan[bi]; s_scan[bi] += temp; }
}

__inline__ __device__
void scan_256(volatile short *s_scan)
{
    int ai, bi;
    int baseai = 1 + 2 * threadIdx.x;
    int basebi = baseai + 1;
    short temp;

    if (threadIdx.x < 128) { ai = baseai - 1;     bi = basebi - 1;     s_scan[bi] += s_scan[ai]; }
    __syncthreads();
    if (threadIdx.x < 64) { ai =  2 * baseai - 1;  bi =  2 * basebi - 1;   s_scan[bi] += s_scan[ai]; }
    __syncthreads();
    if (threadIdx.x < 32) { ai =  4 * baseai - 1;  bi =  4 * basebi - 1;   s_scan[bi] += s_scan[ai]; }
    __syncthreads();
    if (threadIdx.x < 16) { ai =  8 * baseai - 1;  bi =  8 * basebi - 1;   s_scan[bi] += s_scan[ai]; }
    if (threadIdx.x < 8)  { ai = 16 * baseai - 1;  bi = 16 * basebi - 1;   s_scan[bi] += s_scan[ai]; }
    if (threadIdx.x < 4)  { ai = 32 * baseai - 1;  bi = 32 * basebi - 1;   s_scan[bi] += s_scan[ai]; }
    if (threadIdx.x < 2)  { ai = 64 * baseai - 1;  bi = 64 * basebi - 1;   s_scan[bi] += s_scan[ai]; }
    if (threadIdx.x == 0) { s_scan[255] += s_scan[127]; s_scan[256] = s_scan[255]; s_scan[255] = 0; temp = s_scan[127]; s_scan[127] = 0; s_scan[255] += temp; }
    if (threadIdx.x < 2)  { ai = 64 * baseai - 1;  bi = 64 * basebi - 1;   temp = s_scan[ai]; s_scan[ai] = s_scan[bi]; s_scan[bi] += temp;}
    if (threadIdx.x < 4)  { ai = 32 * baseai - 1;  bi = 32 * basebi - 1;   temp = s_scan[ai]; s_scan[ai] = s_scan[bi]; s_scan[bi] += temp;}
    if (threadIdx.x < 8)  { ai = 16 * baseai - 1;  bi = 16 * basebi - 1;   temp = s_scan[ai]; s_scan[ai] = s_scan[bi]; s_scan[bi] += temp;}
    if (threadIdx.x < 16) { ai =  8 * baseai - 1;  bi =  8 * basebi - 1;   temp = s_scan[ai]; s_scan[ai] = s_scan[bi]; s_scan[bi] += temp;}
    if (threadIdx.x < 32) { ai =  4 * baseai - 1;  bi =  4 * basebi - 1;   temp = s_scan[ai]; s_scan[ai] = s_scan[bi]; s_scan[bi] += temp;}
    __syncthreads();
    if (threadIdx.x < 64) { ai =  2 * baseai - 1;  bi =  2 * basebi - 1;   temp = s_scan[ai]; s_scan[ai] = s_scan[bi]; s_scan[bi] += temp;}
    __syncthreads();
    if (threadIdx.x < 128) { ai = baseai - 1;   bi = basebi - 1;   temp = s_scan[ai]; s_scan[ai] = s_scan[bi]; s_scan[bi] += temp; }
}

/*
template<typename sT, typename T>
__inline__ __device__
void scan_double_width_plus1_shfl(volatile  sT *s_scan,
                                  volatile  T *s_scan_shfl,
                                  const int     local_id,
                                  T r_in,
                                  T r_in_halfwidth,
                                  const int seg_num)
{
    // 3-stage method. scan-scan-propogate

    // shfl version
    const int lane_id = local_id % 32;
    const int seg_id = local_id / 32;

    // stage 1. thread bunch scan
    T r_scan = scan_32_shfl<T>(r_in, lane_id);
    T r_scan_halfwidth = scan_32_shfl<T>(r_in_halfwidth, lane_id);

    if (lane_id == 32 - 1)
    {
        s_scan_shfl[seg_id] = r_scan;
        s_scan_shfl[seg_id + seg_num] = r_scan_halfwidth;
    }

    // inclusive to exclusive
    r_scan = __shfl_up(r_scan, 1);
    r_scan_halfwidth = __shfl_up(r_scan_halfwidth, 1);
    r_scan = lane_id ? r_scan : 0;
    r_scan_halfwidth = lane_id ? r_scan_halfwidth : 0;

    __syncthreads();

    // stage 2. one thread bunch scan
    r_in = (local_id < 2 * seg_num) ? s_scan_shfl[local_id] : 0;
    if (!seg_id)
        r_in = scan_32_shfl<T>(r_in, lane_id);

    if (local_id < 2 * seg_num)
        s_scan_shfl[local_id + 1] = r_in;

    // single thread in-place scan
    //scan_single<T>(s_scan_shfl, local_id, seg_num+1);

    __syncthreads();

    // stage 3. propogate (element-wise add) to all
    if (seg_id)
    {
        r_scan += s_scan_shfl[seg_id];
    }
    r_scan_halfwidth += s_scan_shfl[seg_id + seg_num];

    s_scan[local_id] = r_scan;
    s_scan[local_id + blockDim.x] = r_scan_halfwidth;
    if (!local_id)
        s_scan[2 * blockDim.x] = s_scan_shfl[2 * seg_num];

    return;
}
*/

/*  

 --------------- commenting this to try combinning structure and computation stage ------------------

// Trying out ESC algo for our
template <int BLOCK_THREADS>
__global__ void sgpu_SpGEMM_mid2(const int IA[], const int JA[],
    const QValue A[],const int IB[], const int JB[],const QValue B[],
    const int drowIds[], const int gcount,
    const int m, const int n, const int* IC, 
    int JC[], QValue C[]) {

  __shared__ int   s_key[2 * BLOCK_THREADS];
  __shared__ short s_scan[2 * BLOCK_THREADS + 1];
  __shared__ QValue s_val[2 * BLOCK_THREADS];

  //#if __CUDA_ARCH__ >= 300
//	volatile __shared__ int s_scan_shfl[2 * c_scansize / 32 +1];
 // #else
//	volatile __shared__ int *s_scan_shfl;
 // #endif

  const int local_id = threadIdx.x;
  const int group_id = blockIdx.x;
  const int local_size = blockDim.x;
  const int local_id_halfwidth = local_id + local_size;
  const int width = local_size * 2;

  int colID_A;
  int start_row_B;
  int end_row_B;
  int strideB;
  QValue val;
  int invalid_width;

  for(int q = group_id; q < gcount; q += gridDim.x) {
    int rowId = drowIds[q];
    //s_key[local_id] = INT_MAX;
    //s_key[local_id + BLOCK_THREADS] = INT_MAX;
    int local_counter = 0;
    const int ICi = IC[rowId];
    int *iJC = JC + ICi;
    float *iC = C + ICi;
    for(int ap = IA[rowId]; ap < IA[rowId + 1]; ap++) {
        colID_A = JA[ap];
	val = A[ap];  
	start_row_B = IB[colID_A];      
        end_row_B = IB[colID_A + 1];
	
        strideB = end_row_B - start_row_B;

        if(local_id < strideB) {
             s_key[local_counter + local_id] = JB[start_row_B + local_id];
	     s_val[local_counter + local_id] = B[start_row_B + local_id] * val;
        }

        if(local_id_halfwidth < strideB) {
             s_key[local_counter + local_id_halfwidth] = JB[start_row_B + local_id_halfwidth];
	     s_val[local_counter + local_id_halfwidth] = B[start_row_B + local_id_halfwidth] * val;
        }
        local_counter += strideB;
    }
    __syncthreads(); 
    invalid_width = width - local_counter;
    if (local_id < invalid_width) {
	s_key[local_counter + local_id] = n;
    }
    __syncthreads();
    oddeven(s_key, s_val, width);
    __syncthreads();
    bool duplicate = 1;
    bool duplicate_halfwidth = 1;
    if (local_id < local_counter && local_id > 0) {
	    duplicate = (s_key[local_id] != s_key[local_id-1]);
    }
    if (local_id_halfwidth < local_counter) {
        duplicate_halfwidth = (s_key[local_id_halfwidth] != s_key[local_id_halfwidth - 1]);
    }
    s_scan[local_id] = duplicate;
    s_scan[local_id_halfwidth] = duplicate_halfwidth;
    __syncthreads();
    
 //   for(int offset = blockDim.x; offset > 0; offset >>= 1) {
   //     if(local_id < offset) {
  //         s_scan[local_id] += s_scan[local_id + offset];
     //   }
  //  __syncthreads();
 //   if(threadIdx.x == 0) IC[rowId] = s_scan[0] + 1;
    switch(local_size)
    {
        case 64:
              scan_128(s_scan);
              break;
        case 128:
              scan_256(s_scan);
              break;
        case 256:
              scan_512(s_scan);
              break;
    }
    __syncthreads();
    int   move_pointer;
    short final_position, final_position_halfwidth;
    int   final_key,final_key_halfwidth;
    QValue final_value,final_value_halfwidth;
    if (local_id < local_counter && duplicate == 1)
    {
        final_position = s_scan[local_id];
        final_key = s_key[local_id];
        final_value = s_val[local_id];
        move_pointer = local_id + 1;
        while (s_scan[move_pointer] == s_scan[move_pointer + 1])
        {
            final_value += s_val[move_pointer];
            move_pointer++;
        }
    }
    if (local_id_halfwidth < local_counter && duplicate_halfwidth == 1)
    {
        final_position_halfwidth = s_scan[local_id_halfwidth];
        final_key_halfwidth = s_key[local_id_halfwidth];
        final_value_halfwidth = s_val[local_id_halfwidth];
        move_pointer = local_id_halfwidth + 1;

        while (s_scan[move_pointer] == s_scan[move_pointer + 1] && move_pointer < 2 * local_size)
        {
            final_value_halfwidth += s_val[move_pointer];
            move_pointer++;
	}
    }
    __syncthreads();
    // write final_positions and final_values to s_key and s_val
    if (local_id < local_counter && duplicate == 1)
    {
        s_key[final_position] = final_key;
        s_val[final_position] = final_value;
	//iJC[final_position] = final_key;
	//iC[final_position] = final_value;
    }
    if (local_id_halfwidth < local_counter && duplicate_halfwidth == 1)
    {
        s_key[final_position_halfwidth] = final_key_halfwidth;
        s_val[final_position_halfwidth] = final_value_halfwidth;
        //iJC[final_position_halfwidth] = final_key_halfwidth;
	//iC[final_position_halfwidth] = final_value_halfwidth;
    }
    // writing our results to global memory
    __syncthreads();

    local_counter = s_scan[width] - invalid_width;
    if(local_id < local_counter) {
	iJC[local_id] = s_key[local_id];
	iC[local_id] = s_val[local_id];
    }
    if(local_id_halfwidth < local_counter) {
	iJC[local_id_halfwidth] = s_key[local_id_halfwidth];
	iC[local_id_halfwidth] = s_val[local_id_halfwidth];
    }
  }
  }

*/

// My mid bin

template <int BLOCK_THREADS>
__global__ void sgpu_SpGEMM_mix_mid(const int IA[], const int JA[],
    const QValue A[],const int IB[], const int JB[],const QValue B[],
    const int drowIds[], const int gcount,
    const int m, const int n, int* IC, 
    int JC[], QValue C[]) {

  __shared__ int   s_key[2 * BLOCK_THREADS];
  __shared__ short s_scan[2 * BLOCK_THREADS + 1];
  __shared__ QValue s_val[2 * BLOCK_THREADS];
/*
   #if __CUDA_ARCH__ >= 300
	volatile __shared__ int s_scan_shfl[2 * c_scansize / 32 +1];
   #else
 	volatile __shared__ int *s_scan_shfl;
   #endif
*/
  const int local_id = threadIdx.x;
  const int group_id = blockIdx.x;
  const int local_size = blockDim.x;
  const int local_id_halfwidth = local_id + local_size;
  const int width = local_size * 2;

  int colID_A;
  int start_row_B;
  int end_row_B;
  int strideB;
  QValue val;
  int invalid_width;

  for(int q = group_id; q < gcount; q += gridDim.x) {
    int rowId = drowIds[q];
    //s_key[local_id] = INT_MAX;
    //s_key[local_id + BLOCK_THREADS] = INT_MAX;
    int local_counter = 0;
    const int ICi = q * width;
    int *iJC = JC + ICi;
    QValue *iC = C + ICi;
    for(int ap = IA[rowId]; ap < IA[rowId + 1]; ap++) {
        colID_A = JA[ap];
	val = A[ap];  
	start_row_B = IB[colID_A];      
        end_row_B = IB[colID_A + 1];
	
        strideB = end_row_B - start_row_B;

        if(local_id < strideB) {
             s_key[local_counter + local_id] = JB[start_row_B + local_id];
	     s_val[local_counter + local_id] = B[start_row_B + local_id] * val;
        }

        if(local_id_halfwidth < strideB) {
             s_key[local_counter + local_id_halfwidth] = JB[start_row_B + local_id_halfwidth];
	     s_val[local_counter + local_id_halfwidth] = B[start_row_B + local_id_halfwidth] * val;
        }
        local_counter += strideB;
    }
    __syncthreads(); 
    invalid_width = width - local_counter;
    if (local_id < invalid_width) {
	s_key[local_counter + local_id] = n;
    }
    __syncthreads();
    oddeven(s_key, s_val, width);
    __syncthreads();
    bool duplicate = 1;
    bool duplicate_halfwidth = 1;
    if (local_id < local_counter && local_id > 0) {
	    duplicate = (s_key[local_id] != s_key[local_id-1]);
    }
    if (local_id_halfwidth < local_counter) {
        duplicate_halfwidth = (s_key[local_id_halfwidth] != s_key[local_id_halfwidth - 1]);
    }
    s_scan[local_id] = duplicate;
    s_scan[local_id_halfwidth] = duplicate_halfwidth;
    __syncthreads();
    
   //   for(int offset = blockDim.x; offset > 0; offset >>= 1) {
   //     if(local_id < offset) {
   //           s_scan[local_id] += s_scan[local_id + offset];
   //   }
   //  __syncthreads();
   //   if(threadIdx.x == 0) IC[rowId] = s_scan[0] + 1;
/*
#if __CUDA_ARCH__ >= 300
    scan_double_width_plus1_shfl<short, int>(s_scan, s_scan_shfl, local_id,
                                             duplicate, duplicate_halfwidth, local_size/32);
#else*/
     switch(local_size)
    {
    // case 64 can be removed once its confirmed that its not needed
        case 64:
              scan_128(s_scan);
              break;
        case 128:
              scan_256(s_scan);
              break;
        case 256:
              scan_512(s_scan);
              break;
    }/*
#endif*/

    __syncthreads();
    int   move_pointer;
    short final_position, final_position_halfwidth;
    int   final_key,final_key_halfwidth;
    QValue final_value,final_value_halfwidth;
    if (local_id < local_counter && duplicate == 1)
    {
        final_position = s_scan[local_id];
        final_key = s_key[local_id];
        final_value = s_val[local_id];
        move_pointer = local_id + 1;
        while (s_scan[move_pointer] == s_scan[move_pointer + 1])
        {
            final_value += s_val[move_pointer];
            move_pointer++;
        }
    }
    if (local_id_halfwidth < local_counter && duplicate_halfwidth == 1)
    {
        final_position_halfwidth = s_scan[local_id_halfwidth];
        final_key_halfwidth = s_key[local_id_halfwidth];
        final_value_halfwidth = s_val[local_id_halfwidth];
        move_pointer = local_id_halfwidth + 1;

        while (s_scan[move_pointer] == s_scan[move_pointer + 1] && move_pointer < 2 * local_size)
        {
            final_value_halfwidth += s_val[move_pointer];
            move_pointer++;
	} 
    } 
    __syncthreads();
    // write final_positions and final_values to s_key and s_val
    if (local_id < local_counter && duplicate == 1)
    {
        s_key[final_position] = final_key;
        s_val[final_position] = final_value;
	//iJC[final_position] = final_key;
	//iC[final_position] = final_value;
    }
    if (local_id_halfwidth < local_counter && duplicate_halfwidth == 1)
    {
        s_key[final_position_halfwidth] = final_key_halfwidth;
        s_val[final_position_halfwidth] = final_value_halfwidth;
        //iJC[final_position_halfwidth] = final_key_halfwidth;
	//iC[final_position_halfwidth] = final_value_halfwidth;
    }
    // writing our results to global memory
    __syncthreads();

    //local_counter represents the total number of nnz in the row
    local_counter = s_scan[width] - invalid_width;
    if(local_id == 0) {
	IC[rowId] = local_counter;
    }
    if(local_id < local_counter) {
	iJC[local_id] = s_key[local_id];
	iC[local_id] = s_val[local_id];
    }
    if(local_id_halfwidth < local_counter) {
	iJC[local_id_halfwidth] = s_key[local_id_halfwidth];
	iC[local_id_halfwidth] = s_val[local_id_halfwidth];
    }  
  }
  
//   printf("exiting from kernel");
  }


template <int BLOCK_THREADS,int MUL>
__global__ void sgpu_SpGEMM_copy_mid(
    const int drowIds[], const int gcount,
    const int m, const int n, const int* IC, 
    int JC[], QValue C[], int tempJC[], QValue tempC[]) {

    const int group_id = blockIdx.x;
    const int block_size = blockDim.x;
    const int width = block_size * MUL;
    const int local_id = threadIdx.x;

    for(int q = group_id; q < gcount; q += gridDim.x) {
	int rowId = drowIds[q];
	const int distIC = IC[rowId];
        const int rowSize = IC[rowId + 1] - distIC;
	int tempCopyPos = q * width;
        int *iJC = JC + distIC;
	QValue *iC = C + distIC;
	for(int i = local_id; i < rowSize; i+=block_size) {
	    iJC[i] = tempJC[tempCopyPos + i];
	    iC[i] = tempC[tempCopyPos + i];
	} 
    }
}

template <int BLOCK_THREADS>
__global__ void sgpu_SpGEMM_copy_mid_11(
    const int drowIds[], const int gcount,
    const int m, const int n, const int* IC, 
    int JC[], QValue C[], int tempJC[], QValue tempC[]) {

    const int group_id = blockIdx.x;
    const int block_size = blockDim.x;
    const int width = block_size * 4;
    const int local_id = threadIdx.x;

    for(int q = group_id; q < gcount; q += gridDim.x) {
	int rowId = drowIds[q];
	const int distIC = IC[rowId];
        const int rowSize = IC[rowId + 1] - distIC;
	int tempCopyPos = q * width;
        int *iJC = JC + distIC;
	QValue *iC = C + distIC;
	for(int i = local_id; i < rowSize; i+=block_size) {
	    iJC[i] = tempJC[tempCopyPos + i];
	    iC[i] = tempC[tempCopyPos + i];
	} 
    }
}

template <int BLOCK_THREADS>
__global__ void sgpu_SpGEMM_mix_11(const int IA[], const int JA[],
    const QValue A[],const int IB[], const int JB[],const QValue B[],
    const int drowIds[], const int gcount,
    const int m, const int n, int* IC, 
    int JC[], QValue C[], int *xbs) {

  __shared__ int as[BLOCK_THREADS];
  __shared__ QValue Aaps[BLOCK_THREADS];
  __shared__ int count;

  const int local_id = threadIdx.x;
  const int group_id = blockIdx.x;
  const int local_size = blockDim.x;
  const int width = local_size * 4;

  if(local_id == 0) {
	count = 0;
  }
  int *xb = xbs + group_id * n;

  __syncthreads();

  for(int q = group_id; q < gcount; q += gridDim.x) {
    int rowId = drowIds[q];
    const int ICi = q * width;
    int *iJC = JC + ICi;
    QValue *iC = C + ICi;
    int end_Row = IA[rowId + 1];
    for(int ap = IA[rowId] + local_id; __syncthreads_or(ap < end_Row); ap+= local_size) {
	int predicate = (ap < end_Row);
	int a = predicate ? JA[ap] : -1;
 	QValue Aap = predicate ? A[ap] : 0.0;
	as[local_id] = a;
	Aaps[local_id] = Aap;
	unsigned total = min(end_Row + local_id - ap, local_size);
        __syncthreads();
	for (int ap = 0; ap < total; ++ap) {
        int a = as[ap];
        QValue Aap = Aaps[ap];
        for (int bp = IB[a] + local_id; bp < IB[a + 1]; bp += local_size) {
          int b = JB[bp];
          int xbB = xb[b];
          if (xbB == -1) {
            int pos = atomicAdd(&count, 1);
            iJC[pos] = b;
            iC[pos] = Aap * B[bp];
            xb[b] = pos;
          } else {
            iC[xbB] += Aap * B[bp];
          }
        }
        __syncthreads();
      }
    }
    for (int cp = threadIdx.x; cp < count; cp += blockDim.x) {
      int c = iJC[cp];
      xb[c] = -1;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
      IC[rowId] = count;
      count = 0;
    } 
  }
  }

template <int BLOCK_THREADS>
__global__ void sgpu_SpGEMM_mix_12(const int IA[], const int JA[],
    const QValue A[],const int IB[], const int JB[],const QValue B[],
    const int drowIds[], const int gcount,
    const int m, const int n, int* IC, 
    int JC[], QValue C[], int *xbs) {

  __shared__ int as[BLOCK_THREADS];
  __shared__ QValue Aaps[BLOCK_THREADS];
  __shared__ int count;

  const int local_id = threadIdx.x;
  const int group_id = blockIdx.x;
  const int local_size = blockDim.x;
  const int width = local_size * 8;

  if(local_id == 0) {
	count = 0;
  }
  int *xb = xbs + group_id * n;

  __syncthreads();

  for(int q = group_id; q < gcount; q += gridDim.x) {
    int rowId = drowIds[q];
    const int ICi = q * width;
    int *iJC = JC + ICi;
    QValue *iC = C + ICi;
    int end_Row = IA[rowId + 1];
    for(int ap = IA[rowId] + local_id; __syncthreads_or(ap < end_Row); ap+= local_size) {
	int predicate = (ap < end_Row);
	int a = predicate ? JA[ap] : -1;
 	QValue Aap = predicate ? A[ap] : 0.0;
	as[local_id] = a;
	Aaps[local_id] = Aap;
	unsigned total = min(end_Row + local_id - ap, local_size);
        __syncthreads();
	for (int ap = 0; ap < total; ++ap) {
        int a = as[ap];
        QValue Aap = Aaps[ap];
        for (int bp = IB[a] + local_id; bp < IB[a + 1]; bp += local_size) {
          int b = JB[bp];
          int xbB = xb[b];
          if (xbB == -1) {
            int pos = atomicAdd(&count, 1);
            iJC[pos] = b;
            iC[pos] = Aap * B[bp];
            xb[b] = pos;
          } else {
            iC[xbB] += Aap * B[bp];
          }
        }
        __syncthreads();
      }
    }
    for (int cp = threadIdx.x; cp < count; cp += blockDim.x) {
      int c = iJC[cp];
      xb[c] = -1;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
      IC[rowId] = count;
      count = 0;
    } 
  }
  }
