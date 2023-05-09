#include <stdio.h>
#include <stdlib.h>
#include <limits.h>
#include <omp.h>
int main(int argc, char** argv){
    if(argc != 3){
	    fprintf(stderr, "run as ./binauto n nt\n");
        exit(EXIT_FAILURE);
    }
    int N = atoi(argv[1]);
    int nt = atoi(argv[2]);
    omp_set_num_threads(nt);
    printf("lab04 reduction -- automatic, n=%i  nt=%i\n", N, nt);
    double sum = 0.0, *x;
    x = (double*)malloc(sizeof(double)*N);
    for(int i = 0; i < N; ++i){
        x[i] = (double)rand()/(double)RAND_MAX;
    }

    // reduction
    for(int i = 0; i < N; ++i){
        sum += x[i];
    }
    free(x);
    printf("sum = %f\nDONE\n", sum);
}
