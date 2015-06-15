#ifndef CAS_HASH_CUH
#define CAS_HASH_CUH

__device__ inline uint32_t hash(uint32_t d) {
  d = (d ^ 61) ^ (d >> 16);
  d = d + (d << 3);
  d = d ^ (d >> 4);
  d = d * 0x27d4eb2d;
  d = d ^ (d >> 15);
  return d;
}


__device__ inline int hashCASAdd(int hb[], const int hprime, const int b) {
  int index = hash(b) % hprime;
  do {
    int old = atomicCAS(hb + index, -1, b);
    if (old == -1) return -1;
    int ob = atomicCAS(hb + index, b, b);
    if (ob == b) return index;
    index = (index+1) % hprime;
  } while (true);
} 

/*__device__ inline int bigHashCASAdd(int hb[], const int hprime, const int b) {

   int index = hash(b) % hprime;

   do {
	
   } while (true);
}*/

__device__ inline int hashCASAdd2(int hb[], const int hprime, const int b, int *pos) {
  *pos = hash(b) % hprime;
  do {
    int old = atomicCAS(hb + *pos, -1, b);
    if (old == -1) return -1;
    int ob = atomicCAS(hb + *pos, b, b);
    if (ob == b) return *pos;
    *pos = (*pos + 1) % hprime;
  } while (true);
}
#endif
