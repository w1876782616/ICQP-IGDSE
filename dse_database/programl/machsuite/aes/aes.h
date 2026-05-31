

#include "support.h"

typedef struct {
  uint8_t key[32];
  uint8_t enckey[32];
  uint8_t deckey[32];
} aes256_context;

void aes256_encrypt_ecb(aes256_context *ctx, uint8_t k[32], uint8_t buf[16]);

struct bench_args_t {
  aes256_context ctx;
  uint8_t k[32];
  uint8_t buf[16];
};

