#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <assert.h>

#include "nw.h"
int main(int argc, char **argv)
{
  struct bench_args_t data;

  char seqA[ALEN+1] = "tcgacgaaataggatgacagcacgttctcgtattagagggccgcggtacaaaccaaatgctgcggcgtacagggcacggggcgctgttcgggagatcgggggaatcgtggcgtgggtgattcgccggc";
  char seqB[BLEN+1] = "ttcgagggcgcgtgtcgcggtccatcgacatgcccggtcggtgggacgtgggcgcctgatatagaggaatgcgattggaaggtcggacgggtcggcgagttgggcccggtgaatctgccatggtcgat";
  int fd;

  assert( ALEN==strlen(seqA) && "String initializers must be exact length");
  assert( BLEN==strlen(seqB) && "String initializers must be exact length");

  memcpy(data.seqA, seqA, ALEN);
  memcpy(data.seqB, seqB, BLEN);

  fd = open("input.data", O_WRONLY|O_CREAT|O_TRUNC, S_IRUSR|S_IWUSR|S_IRGRP|S_IWGRP|S_IROTH|S_IWOTH);
  assert( fd>0 && "Couldn't open input data file" );
  data_to_input(fd, (void *)(&data));

  return 0;
}
