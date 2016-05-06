#include <cstdlib>
#include <cstdio>
#include <cstring>

////////////////////////////////////////////////////////////////////////////////
// Common host and device functions
////////////////////////////////////////////////////////////////////////////////
//Round a / b to nearest higher integer value
int iDivUp(int a, int b){
    return (a % b != 0) ? (a / b + 1) : (a / b);
}

//Round a / b to nearest lower integer value
int iDivDown(int a, int b){
    return a / b;
}

//Align a to nearest higher multiple of b
int iAlignUp(int a, int b){
    return (a % b != 0) ?  (a - a % b + b) : a;
}

//Align a to nearest lower multiple of b
int iAlignDown(int a, int b){
    return a - a % b;
}

////////////////////////////////////////////////////////////////////////////////
// GPU convolution
////////////////////////////////////////////////////////////////////////////////
//kernels
#include "kern.cu"

////////////////////////////////////////////////////////////////////////////////
// Data configuration
////////////////////////////////////////////////////////////////////////////////
//Image width should be aligned to maximum coalesced read/write size
//for best global memory performance in both row and column filter.

const int      DATA_W = iAlignUp(256, 16);
const int      DATA_H = 256;

const int   DATA_SIZE = DATA_W * DATA_H * sizeof(float);
const int KERNEL_SIZE = KERNEL_W * sizeof(float);

//Carry out dummy calculations before main computation loop
//in order to "warm up" the hardware/driver
#define WARMUP
////////////////////////////////////////////////////////////////////////////////
// Main program
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char **argv){
    float
        *h_Kernel,
        *h_DataA,
        *h_DataB,
        *h_ResultGPU;

    float
        *d_DataA,
        *d_DataB;

    double
        sum_delta, sum_ref, L1norm, gpuTime;

    int i;

    //unsigned int hTimer;

    //CUT_SAFE_CALL(cutCreateTimer(&hTimer));

    printf("%i x %i\n", DATA_W, DATA_H);
    printf("Initializing data...\n");
        h_Kernel    = (float *)malloc(KERNEL_SIZE);
        h_DataA     = (float *)malloc(DATA_SIZE);
        h_DataB     = (float *)malloc(DATA_SIZE);
        h_ResultGPU = (float *)malloc(DATA_SIZE);
        cudaMalloc( (void **)&d_DataA, DATA_SIZE);
        cudaMalloc( (void **)&d_DataB, DATA_SIZE);

        float kernelSum = 0;
        for(i = 0; i < KERNEL_W; i++){
            float dist = (float)(i - KERNEL_RADIUS) / (float)KERNEL_RADIUS;
            h_Kernel[i] = expf(- dist * dist / 2);
            kernelSum += h_Kernel[i];
        }
        for(i = 0; i < KERNEL_W; i++)
            h_Kernel[i] /= kernelSum;

        srand((int)time(NULL));
        for(i = 0; i < DATA_W * DATA_H; i++)
            h_DataA[i] = (float)rand() / (float)RAND_MAX;

        cudaMemcpyToSymbol(d_Kernel, h_Kernel, KERNEL_SIZE);
        cudaMemcpy(d_DataA, h_DataA, DATA_SIZE, cudaMemcpyHostToDevice);


    dim3 blockGridRows(iDivUp(DATA_W, ROW_TILE_W), DATA_H);
    dim3 blockGridColumns(iDivUp(DATA_W, COLUMN_TILE_W), iDivUp(DATA_H, COLUMN_TILE_H));
    dim3 threadBlockRows(KERNEL_RADIUS_ALIGNED + ROW_TILE_W + KERNEL_RADIUS);
    dim3 threadBlockColumns(COLUMN_TILE_W, 8);

    printf("GPU convolution...\n");

        cudaThreadSynchronize();
        //CUT_SAFE_CALL( cutResetTimer(hTimer) );
        //CUT_SAFE_CALL( cutStartTimer(hTimer) );
        convolutionRowGPU<<<blockGridRows, threadBlockRows>>>(
            d_DataB,
            d_DataA,
            DATA_W,
            DATA_H
        );

        convolutionColumnGPU<<<blockGridColumns, threadBlockColumns>>>(
            d_DataA,
            d_DataB,
            DATA_W,
            DATA_H,
            COLUMN_TILE_W * threadBlockColumns.y,
            DATA_W * threadBlockColumns.y
        );

    cudaThreadSynchronize();
    //CUT_SAFE_CALL(cutStopTimer(hTimer));
    //gpuTime = cutGetTimerValue(hTimer);
    //printf("GPU convolution time : %f msec //%f Mpixels/sec\n", gpuTime, 1e-6 * DATA_W * DATA_H / (gpuTime * 0.001));

    printf("Reading back GPU results...\n");
        cudaMemcpy(h_ResultGPU, d_DataA, DATA_SIZE, cudaMemcpyDeviceToHost);

    printf("Shutting down...\n");
        cudaFree(d_DataB);
        cudaFree(d_DataA);
        free(h_ResultGPU);
        free(h_DataB);
        free(h_DataA);
        free(h_Kernel);
}
