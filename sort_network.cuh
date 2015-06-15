
__device__ inline void Sort2(int *p0, int *p1) {
  if (*p0 > *p1) {
    int tmp = *p0;
    *p0 = *p1;
    *p1 = tmp;
  }
}

__device__ inline void Sort3(int *p0, int *p1, int *p2) {
  Sort2(p0, p1);
  Sort2(p1, p2);
  Sort2(p0, p1);
}

//[[0 1][2 3][0 2][1 3][1 2]]
__device__ inline void Sort4(int *p0, int *p1, int *p2, int *p3) {
  Sort2(p0, p1);
  Sort2(p2, p3);
  Sort2(p0, p2);
  Sort2(p1, p3);
  Sort2(p1, p2);
}

//[[0 1][2 3][0 2][1 4][0 1][2 3][1 2][3 4][2 3]]
__device__ inline void Sort5(int *p0, int *p1, int *p2, int *p3, int *p4) {
  Sort2(p0, p1);
  Sort2(p2, p3);
  Sort2(p0, p2);
  Sort2(p1, p4);
  Sort2(p0, p1);
  Sort2(p2, p3);
  Sort2(p1, p2);
  Sort2(p3, p4);
  Sort2(p2, p3);
}

__device__ inline void kvSort2(int *p0, QValue *f0,
    int *p1, QValue *f1) {
  if (*p0 > *p1) {
    int tmp = *p0; QValue ft = *f0;
    *p0 = *p1; *f0 = *f1;
    *p1 = tmp; *f1 = ft;
  }
}

__device__ inline void kvSort3(int *p0, QValue *f0,
    int *p1, QValue *f1,
    int *p2, QValue *f2) {
  kvSort2(p0, f0, p1, f1);
  kvSort2(p1, f1, p2, f2);
  kvSort2(p0, f0, p1, f1);
}

//[[0 1][2 3][0 2][1 3][1 2]]
__device__ inline void kvSort4(int *p0, QValue *f0,
    int *p1, QValue *f1,
    int *p2, QValue *f2,
    int *p3, QValue *f3) {
  kvSort2(p0, f0, p1, f1);
  kvSort2(p2, f2, p3, f3);
  kvSort2(p0, f0, p2, f2);
  kvSort2(p1, f1, p3, f3);
  kvSort2(p1, f1, p2, f2);
}

/*//[[0 1][2 3][4 5][0 2][1 4][3 5][0 1][2 3][4 5][1 2][3 4][2 3]]
__device inline void Sort6(int *p0, int *p1, int *p2, int *p3, int *p4, int *p5) {
  Sort2(p0, p1);
  Sort2(p2, p3);
  Sort2(p4, p5);
  Sort2(p0, p2);
  Sort2(p1, p4);
  Sort2(p3, p5);
  Sort2(p0, p1);
  Sort2(p2, p3);
  Sort2(p4, p5);
  Sort2(p1, p2);
  Sort2(p3, p4);
  Sort2(p2, p3);
}

__device inline void Sort8(int *p0, int *p1, int *p2, int *p3,
    int *p4, int *p5, int *p6, int *p7) {
  Sort4(p0, p1, p2, p3);
  Sort4(p4, p5, p6, p7);
  Sort2(p0, p4);
  Sort2(p1, p5);
  Sort2(p2, p6);
  Sort2(p3, p7);
  Sort2(p2, p4);
  Sort2(p3, p5);
  Sort2(p1, p2);
  Sort2(p3, p4);
  Sort2(p5, p6);
}

__device inline void Sort7(int *p0, int *p1, int *p2, int *p3,
    int *p4, int *p5, int *p6) {
  int e7 = INT_MAX;
  Sort8(p0, p1, p2, p3, p4, p5, p6, &e7);
}*/
