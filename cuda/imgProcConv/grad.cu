#include <cstdlib>
#include <cstdio>
#include <cstring>

////////////////////////////////////////////////////////////////////////////////
// Tracker state variables and pointers
////////////////////////////////////////////////////////////////////////////////
//Have the pointers been initialized?
bool trackerPointers = false;



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
#include "gradkern.cu"
#include "convkern.cu"

////////////////////////////////////////////////////////////////////////////////
// Data configuration
////////////////////////////////////////////////////////////////////////////////
//Image width should be aligned to maximum coalesced read/write size
//for best global memory performance in both row and column filter.

//const int      DATA_W = iAlignUp(256, 16);
//const int      DATA_H = 256;

//const int   DATA_SIZE = DATA_W * DATA_H * sizeof(float);
//const int GRADIENT_SIZE = GRADIENT_W * sizeof(float);

////////////////////////////////////////////////////////////////////////////////
// Main program
////////////////////////////////////////////////////////////////////////////////
void calculateGradient(float* h_ResultGPU, float* img, int imgWidth, int imgHeight){

    const int      DATA_W = imgWidth;
    const int      DATA_H = imgHeight;

    const int   DATA_SIZE = DATA_W * DATA_H * sizeof(float);
    const int GRADIENT_SIZE = GRADIENT_W * sizeof(float);

    float
        *h_Gradient,
        *h_DataA;

    float
        *d_DataA,
        *d_DataIx,
        *d_DataIy,
        *d_DataTemp;

    double
        sum_delta, sum_ref, L1norm, gpuTime;

    int i;

    printf("%i x %i\n", DATA_W, DATA_H);
    printf("Initializing data...\n");
        h_Gradient  = (float *)malloc(GRADIENT_SIZE);
        h_DataA     = (float *)malloc(DATA_SIZE);
        cudaMalloc( (void **)&d_DataA, DATA_SIZE);
        cudaMalloc( (void **)&d_DataIx, DATA_SIZE);
        cudaMalloc( (void **)&d_DataIy, DATA_SIZE);
        cudaMalloc( (void **)&d_DataTemp, DATA_SIZE);

        float kernelSum = 0;
        for(i = -GRADIENT_RADIUS; i < GRADIENT_RADIUS+1; i++){
            h_Gradient[i+GRADIENT_RADIUS] = 1.0*i;
        }
        
        for(i = 0; i < DATA_W * DATA_H; i++)
            h_DataA[i] = img[i];

        cudaMemcpyToSymbol(d_Gradient, h_Gradient, GRADIENT_SIZE);
        cudaMemcpy(d_DataA, h_DataA, DATA_SIZE, cudaMemcpyHostToDevice);


    dim3 gradBlockGridRows(iDivUp(DATA_W, GRAD_ROW_TILE_W), DATA_H);
    dim3 gradBlockGridColumns(iDivUp(DATA_W, GRAD_COLUMN_TILE_W), iDivUp(DATA_H, GRAD_COLUMN_TILE_H));
    dim3 gradThreadBlockRows(GRADIENT_RADIUS_ALIGNED + GRAD_ROW_TILE_W + GRADIENT_RADIUS);
    dim3 gradThreadBlockColumns(GRAD_COLUMN_TILE_W, 8);

/**/
    printf("GPU gradient calculation...\n");

        cudaThreadSynchronize();
        gradientRowGPU<<<gradBlockGridRows, gradThreadBlockRows>>>(
            d_DataIx,
            d_DataA,
            DATA_W,
            DATA_H
        );

        gradientColumnGPU<<<gradBlockGridColumns, gradThreadBlockColumns>>>(
            d_DataIy,
            d_DataA,
            DATA_W,
            DATA_H,
            GRAD_COLUMN_TILE_W * gradThreadBlockColumns.y,
            DATA_W * gradThreadBlockColumns.y
        );

    cudaThreadSynchronize();
/**/
    printf("Reading back GPU results...\n");
        cudaMemcpy(h_ResultGPU, d_DataIx, DATA_SIZE, cudaMemcpyDeviceToHost);

    printf("Shutting down...\n");
        cudaFree(d_DataIx);
        cudaFree(d_DataIy);
        cudaFree(d_DataA);
        free(h_DataA);
        free(h_Gradient);

    return;
}
