#ifndef _H_INSERTS
#define _H_INSERTS

#include <assert.h>

#define msg(format, ...) do { fprintf(stderr, format, ##__VA_ARGS__); } while (0)
#define err(format, ...) do { fprintf(stderr, format, ##__VA_ARGS__); exit(1); } while (0)



#ifdef __CUDACC__
inline void checkCuda(cudaError_t e) {
    if (e != cudaSuccess) {
        err("CUDA Error: %s\n", cudaGetErrorString(e));
    }
}

inline void checkLastCudaError() {
    checkCuda(cudaGetLastError());
}
#endif

int***  cuda_computation(int***, int, int, int, float);
int*** cuda_computation2 ( int *** ,int ,int ,int );
int*** file_read(char*, int*, int*, int*);
int     file_write(char*, int, int, int, int***);


double  wtime(void);

extern int _debug;

#endif
