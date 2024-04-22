#ifndef GLOBAL_CUH
#define GLOBAL_CUH

__global__ void vecminus(double *vec, double *minvec, int size);
__global__ void concatenateMatricesKernel(double *sol_con, double *P, double *A, int m, int n, double rho, double sigma);

#endif