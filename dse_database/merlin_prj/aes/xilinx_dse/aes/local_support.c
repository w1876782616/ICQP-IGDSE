#include "aes.h"
#include <string.h>

int INPUT_SIZE = sizeof(struct bench_args_t);

void run_benchmark( void *vargs ) {
  struct bench_args_t *args = (struct bench_args_t *)vargs;
  aes256_encrypt_ecb( &(args->ctx), args->k, args->buf );
}

void input_to_data(int fd, void *vdata) {
  struct bench_args_t *data = (struct bench_args_t *)vdata;
  char *p, *s;

  memset(vdata,0,sizeof(struct bench_args_t));

  p = readfile(fd);

  s = find_section_start(p,1);
  parse_uint8_t_array(s, data->k, 32);

  s = find_section_start(p,2);
  parse_uint8_t_array(s, data->buf, 16);
  free(p);
}

void data_to_input(int fd, void *vdata) {
  struct bench_args_t *data = (struct bench_args_t *)vdata;

  write_section_header(fd);
  write_uint8_t_array(fd, data->k, 32);

  write_section_header(fd);
  write_uint8_t_array(fd, data->buf, 16);
}

void output_to_data(int fd, void *vdata) {
  struct bench_args_t *data = (struct bench_args_t *)vdata;

  char *p, *s;

  memset(vdata,0,sizeof(struct bench_args_t));

  p = readfile(fd);

  s = find_section_start(p,1);
  parse_uint8_t_array(s, data->buf, 16);
  free(p);
}

void data_to_output(int fd, void *vdata) {
  struct bench_args_t *data = (struct bench_args_t *)vdata;

  write_section_header(fd);
  write_uint8_t_array(fd, data->buf, 16);
}

int check_data( void *vdata, void *vref ) {
  struct bench_args_t *data = (struct bench_args_t *)vdata;
  struct bench_args_t *ref = (struct bench_args_t *)vref;
  int has_errors = 0;

  has_errors |= memcmp(&data->buf, &ref->buf, 16*sizeof(uint8_t));

  return !has_errors;
}
