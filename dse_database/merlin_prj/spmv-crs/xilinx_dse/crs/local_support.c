#include "spmv.h"
#include <string.h>

int INPUT_SIZE = sizeof(struct bench_args_t);

#define EPSILON 1.0e-6

void run_benchmark( void *vargs ) {
  struct bench_args_t *args = (struct bench_args_t *)vargs;
  spmv( args->val, args->cols, args->rowDelimiters, args->vec, args->out );
}

void input_to_data(int fd, void *vdata) {
  struct bench_args_t *data = (struct bench_args_t *)vdata;
  char *p, *s;

  memset(vdata,0,sizeof(struct bench_args_t));

  p = readfile(fd);

  s = find_section_start(p,1);
  STAC(parse_,TYPE,_array)(s, data->val, NNZ);

  s = find_section_start(p,2);
  parse_int32_t_array(s, data->cols, NNZ);

  s = find_section_start(p,3);
  parse_int32_t_array(s, data->rowDelimiters, N+1);

  s = find_section_start(p,4);
  STAC(parse_,TYPE,_array)(s, data->vec, N);
  free(p);
}

void data_to_input(int fd, void *vdata) {
  struct bench_args_t *data = (struct bench_args_t *)vdata;

  write_section_header(fd);
  STAC(write_,TYPE,_array)(fd, data->val, NNZ);

  write_section_header(fd);
  write_int32_t_array(fd, data->cols, NNZ);

  write_section_header(fd);
  write_int32_t_array(fd, data->rowDelimiters, N+1);

  write_section_header(fd);
  STAC(write_,TYPE,_array)(fd, data->vec, N);
}

void output_to_data(int fd, void *vdata) {
  struct bench_args_t *data = (struct bench_args_t *)vdata;
  char *p, *s;

  p = readfile(fd);

  s = find_section_start(p,1);
  STAC(parse_,TYPE,_array)(s, data->out, N);
  free(p);
}

void data_to_output(int fd, void *vdata) {
  struct bench_args_t *data = (struct bench_args_t *)vdata;

  write_section_header(fd);
  STAC(write_,TYPE,_array)(fd, data->out, N);
}

int check_data( void *vdata, void *vref ) {
  struct bench_args_t *data = (struct bench_args_t *)vdata;
  struct bench_args_t *ref = (struct bench_args_t *)vref;
  int has_errors = 0;
  int i;
  TYPE diff;

  for(i=0; i<N; i++) {
    diff = data->out[i] - ref->out[i];
    has_errors |= (diff<-EPSILON) || (EPSILON<diff);
  }

  return !has_errors;
}
