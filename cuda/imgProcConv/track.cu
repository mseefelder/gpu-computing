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
    const int KERNEL_SIZE = KERNEL_W * sizeof(float);

    float
        *h_Gradient,
        *h_Kernel;

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
        h_Kernel    = (float *)malloc(KERNEL_SIZE);
        cudaMalloc( (void **)&d_DataA, DATA_SIZE);
        cudaMalloc( (void **)&d_DataIx, DATA_SIZE);
        cudaMalloc( (void **)&d_DataIy, DATA_SIZE);
        cudaMalloc( (void **)&d_DataTemp, DATA_SIZE);

        //Initializing gradient computation filter
        for(i = -GRADIENT_RADIUS; i < GRADIENT_RADIUS+1; i++){
            h_Gradient[i+GRADIENT_RADIUS] = 1.0*i;
        }

        //Initializing smoothing convolution kernel
        float kernelSum = 0;
        for(i = 0; i < KERNEL_W; i++){
            //float dist = (float)(i - KERNEL_RADIUS) / (float)KERNEL_RADIUS;
            h_Kernel[i] = 1.0;//expf(- dist * dist / 2);
            kernelSum += h_Kernel[i];
        }
        for(i = 0; i < KERNEL_W; i++)
            h_Kernel[i] /= kernelSum;

        //Copying to device
        cudaMemcpyToSymbol(d_Gradient, h_Gradient, GRADIENT_SIZE);
        cudaMemcpyToSymbol(d_Kernel, h_Kernel, KERNEL_SIZE);
        cudaMemcpy(d_DataA, img, DATA_SIZE, cudaMemcpyHostToDevice);

    //Gradient filter dimensions
    dim3 gradBlockGridRows(iDivUp(DATA_W, GRAD_ROW_TILE_W), DATA_H);
    dim3 gradBlockGridColumns(iDivUp(DATA_W, GRAD_COLUMN_TILE_W), iDivUp(DATA_H, GRAD_COLUMN_TILE_H));
    dim3 gradThreadBlockRows(GRADIENT_RADIUS_ALIGNED + GRAD_ROW_TILE_W + GRADIENT_RADIUS);
    dim3 gradThreadBlockColumns(GRAD_COLUMN_TILE_W, 8);

    //Smoothing convolution filter dimensions
    dim3 convBlockGridRows(iDivUp(DATA_W, ROW_TILE_W), DATA_H);
    dim3 convBlockGridColumns(iDivUp(DATA_W, COLUMN_TILE_W), iDivUp(DATA_H, COLUMN_TILE_H));
    dim3 convThreadBlockRows(KERNEL_RADIUS_ALIGNED + ROW_TILE_W + KERNEL_RADIUS);
    dim3 convThreadBlockColumns(COLUMN_TILE_W, 8);

/**/
    printf("GPU gradient calculation...\n");

        cudaThreadSynchronize();
        //Compute Ix and Iy (gradient)
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

        //Compute Ix*Iy smoothed
        iXiYRowGPU<<<convBlockGridRows, convThreadBlockRows>>>(
            d_DataTemp,
            d_DataIx,
            d_DataIy,
            DATA_W,
            DATA_H
        );

        convolutionColumnGPU<<<convBlockGridColumns, convThreadBlockColumns>>>(
            d_DataA,
            d_DataTemp,
            DATA_W,
            DATA_H,
            COLUMN_TILE_W * convThreadBlockColumns.y,
            DATA_W * convThreadBlockColumns.y
        );

        //Compute Ix^2 smoothed
        convolutionSquaredRowGPU<<<convBlockGridRows, convThreadBlockRows>>>(
            d_DataTemp,
            d_DataIx,
            DATA_W,
            DATA_H
        );

        convolutionColumnGPU<<<convBlockGridColumns, convThreadBlockColumns>>>(
            d_DataIx,
            d_DataTemp,
            DATA_W,
            DATA_H,
            COLUMN_TILE_W * convThreadBlockColumns.y,
            DATA_W * convThreadBlockColumns.y
        );

        //Compute Iy^2 smoothed
        convolutionSquaredRowGPU<<<convBlockGridRows, convThreadBlockRows>>>(
            d_DataTemp,
            d_DataIy,
            DATA_W,
            DATA_H
        );

        convolutionColumnGPU<<<convBlockGridColumns, convThreadBlockColumns>>>(
            d_DataIy,
            d_DataTemp,
            DATA_W,
            DATA_H,
            COLUMN_TILE_W * convThreadBlockColumns.y,
            DATA_W * convThreadBlockColumns.y
        );        

    cudaThreadSynchronize();
/**/
    printf("Reading back GPU results...\n");
        cudaMemcpy(h_ResultGPU, d_DataIx, DATA_SIZE, cudaMemcpyDeviceToHost);

    printf("Shutting down...\n");
        cudaFree(d_DataIx);
        cudaFree(d_DataIy);
        cudaFree(d_DataA);
        free(h_Gradient);
        free(h_Kernel);

    return;
}
