#include "nw.h"

int main(){
    int i;
    char allignedA[N+M];
    char allignedB[M+M];

char seqA[N] = "tcgacgaaataggatgacagcacgttctcgtattagagggccgcggtacaaaccaaatgctgcggcgtacagggcacggggcgctgttcgggagatcgggggaatcgtggcgtgggtgattcgccggc";

  char seqB[M] = "ttcgagggcgcgtgtcgcggtccatcgacatgcccggtcggtgggacgtgggcgcctgatatagaggaatgcgattggaaggtcggacgggtcggcgagttgggcccggtgaatctgccatggtcgat";
    char sA[N];
    char sB[M];
    for(i=0;i<M;i++){
        sA[i] = seqA[i];
        sB[i] = seqB[i];
    }

    needwun(sA, sB, allignedA, allignedB);

    return 0;
}
