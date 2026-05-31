

#include <stdlib.h>
#include <stdio.h>
#include "aes.h"

#define DUMP(s, i, buf, sz)  {printf(s);                   \
                              for (i = 0; i < (sz);i++)    \
                                  printf("%02x ", buf[i]); \
                              printf("\n");}

int main (int argc, char *argv[])
{
    aes256_context ctx;
    uint8_t key[32];
    uint8_t buf[16], i;

    for (i = 0; i < sizeof(buf);i++){
        buf[i] = i * 16 + i;
    }

    for (i = 0; i < sizeof(key);i++){
        key[i] = i;
    }

    DUMP("txt: ", i, buf, sizeof(buf));
    DUMP("key: ", i, key, sizeof(key));

    printf("---\n");

    aes256_encrypt_ecb(&ctx,key, buf);

    DUMP("enc: ", i, buf, sizeof(buf));
    printf("tst: 8e a2 b7 ca 51 67 45 bf ea fc 49 90 4b 49 60 89\n");

    return 0;
}

