#include "nw.h"
#include <string.h>

int INPUT_SIZE = sizeof(struct bench_args_t);

void run_benchmark( void *vargs ) {
  struct bench_args_t *args = (struct bench_args_t *)vargs;
  needwun( args->seqA, args->seqB, args->alignedA, args->alignedB, args->M, args->ptr);
}

void input_to_data(int fd, void *vdata) {
  struct bench_args_t *data = (struct bench_args_t *)vdata;
  char *p, *s;

  memset(vdata,0,sizeof(struct bench_args_t));

  p = readfile(fd);

  s = find_section_start(p,1);
  parse_string(s, data->seqA, ALEN);

  s = find_section_start(p,2);
  parse_string(s, data->seqB, BLEN);
  free(p);

}

void data_to_input(int fd, void *vdata) {
  struct bench_args_t *data = (struct bench_args_t *)vdata;

  write_section_header(fd);
  write_string(fd, data->seqA, ALEN);

  write_section_header(fd);
  write_string(fd, data->seqB, BLEN);

  write_section_header(fd);
}

void output_to_data(int fd, void *vdata) {
  struct bench_args_t *data = (struct bench_args_t *)vdata;
  char *p, *s;

  memset(vdata,0,sizeof(struct bench_args_t));

  p = readfile(fd);

  s = find_section_start(p,1);
  parse_string(s, data->alignedA, ALEN+BLEN);

  s = find_section_start(p,2);
  parse_string(s, data->alignedB, ALEN+BLEN);
  free(p);
}

void data_to_output(int fd, void *vdata) {
  struct bench_args_t *data = (struct bench_args_t *)vdata;

  write_section_header(fd);
  write_string(fd, data->alignedA, ALEN+BLEN);

  write_section_header(fd);
  write_string(fd, data->alignedB, ALEN+BLEN);

  write_section_header(fd);
}

int check_data( void *vdata, void *vref ) {
  struct bench_args_t *data = (struct bench_args_t *)vdata;
  struct bench_args_t *ref = (struct bench_args_t *)vref;
  int has_errors = 0;

  has_errors |= memcmp(data->alignedA, ref->alignedA, ALEN+BLEN);
  has_errors |= memcmp(data->alignedB, ref->alignedB, ALEN+BLEN);

  return !has_errors;
}
