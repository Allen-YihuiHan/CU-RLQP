#include <stdio.h>
#include "global.cuh"

__global__ void vecminus(double *vec, double *minvec, int size)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    if (idx < size)
        vec[idx] = vec[idx] - minvec[idx];

    __syncthreads();
}

__global__ void concatenateMatricesKernel(double *sol_con, double *P, double *A, int m, int n, double rho, double sigma)
{
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    int idy = threadIdx.y + blockDim.y * blockIdx.y;

    if (idx < m + n && idy < m + n)
    {
        // Concatenate P + rho*I
        if (idx < n && idy < n)
            sol_con[idy * (n + m) + idx] = P[idy * n + idx] + ((idx == idy) ? sigma : 0.0);
        else if (n <= idy && idy < m + n && idx < n) // Concatenate AˆT
            sol_con[idy * (n + m) + idx] = A[idx * m + (idy - n)];
        else if (n <= idx && idx < n + m && idy < n) // Concatenate A
            sol_con[idy * (n + m) + idx] = A[idy * m + (idx - n)];
        else if (n <= idx && n <= idy && idx < n + m && idy < n + m) // Concatenate -(1/rho)*I
            sol_con[idy * (n + m) + idx] = ((idx == idy) ? -1.0 / rho : 0.0);
    }
    __syncthreads();
}