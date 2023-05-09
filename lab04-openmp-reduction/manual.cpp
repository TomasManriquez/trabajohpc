#include <stdio.h>
#include <stdlib.h>
#include <limits.h>
#include <omp.h>

double parallel_reduction(double *x, long n, int nt){
    // programar en paralelo reduccion (OpenMP)
    double *results = new double[nt];
    #pragma omp parallel shared(results)
    {
        // FASE 1 ---> de n valores a nt resultados.
        int tid = omp_get_thread_num();
        results[tid] = 0;
        int segment = (n + nt -1)/nt;
        int start = segment*tid;
        int end = start + segment;
        // sumando segmentos en paralelo
		double lsum = 0.0f;
        for(int i=start; i<end && i<n; ++i){
            //results[tid] += x[i];
            lsum += x[i];
        }
		results[tid] = lsum;
        #pragma omp barrier
        // terminamos con sumas parciales en "results", nos olvidamos de x
        // FASE 2 ---> el proceso O(log n) --> terminamos 1 valor
        // de nt --> a 1 resultado gradualmente
        int workers = nt/2;
        while(workers > 0){
            if(tid < workers){
                results[tid] += results[tid + workers];
            }
            workers = workers/2;
            #pragma omp barrier
        }
        // resultado queda en results[0]
    }
    return results[0];
}

int main(int argc, char** argv){
    if(argc != 3){
	    fprintf(stderr, "run as ./binauto n nt\n");
	    exit(EXIT_FAILURE);
    }
    int N = atoi(argv[1]);
    int nt = atoi(argv[2]);
    printf("lab04 reduction -- manual, n=%i, nt=%i\n", N, nt);
    omp_set_num_threads(nt);
    double *x;

    // malloc e inicializacion
    printf("reservando e inicializando memoria....."); fflush(stdout);
    x = (double*)malloc(sizeof(double)*N);
    for(int i = 0; i < N; ++i){
        x[i] = (double)rand()/(double)RAND_MAX;
    }
    printf("done\n"); fflush(stdout);

    // calculo paralelo
    printf("manual parallel reduction.............."); fflush(stdout);
    double t1 = omp_get_wtime();
    double psum = parallel_reduction(x, N, nt);
    double t2 = omp_get_wtime();
    printf("done: %f secs, result: %f\n", t2-t1, psum); fflush(stdout);

    // calculo secuencial
    printf("sequential reduction..................."); fflush(stdout);
    t1 = omp_get_wtime();
    double sum=0.0;
    for(int i = 0; i < N; ++i){
        sum += x[i];
    }
    t2 = omp_get_wtime();
    printf("done: %f secs, result: %f\n", t2-t1, sum); fflush(stdout);
    free(x);
}
