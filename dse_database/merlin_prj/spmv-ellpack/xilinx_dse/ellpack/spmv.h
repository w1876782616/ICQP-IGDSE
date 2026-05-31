

#include <stdlib.h>
#include <stdio.h>
#include "support.h"

#define NNZ 1666
#define N 494
#define L 10

#define TYPE double

void ellpack(TYPE nzval[N*L], int32_t cols[N*L], TYPE vec[N], TYPE out[N]);

struct bench_args_t {
  TYPE nzval[N*L];
  int32_t cols[N*L];
  TYPE vec[N];
  TYPE out[N];
};
