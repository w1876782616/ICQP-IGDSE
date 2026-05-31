
#include <stdio.h>
#include <stdlib.h>
#include "support.h"

#define TYPE double

#define row_size 64
#define col_size 64
#define N row_size*col_size

#define MIN 0.
#define MAX 1.0

#define MAX_ITERATION 1

void gemm(TYPE m1[N], TYPE m2[N], TYPE prod[N]);

struct bench_args_t {
  TYPE m1[N];
  TYPE m2[N];
  TYPE prod[N];
};
