////////////////////////////////////////////////////////////////////////////////
// Kernel configuration
////////////////////////////////////////////////////////////////////////////////
/*
#define KERNEL_RADIUS 8
#define      KERNEL_W (2 * KERNEL_RADIUS + 1)
__device__ __constant__ float d_Kernel[KERNEL_W];

// Assuming ROW_TILE_W, KERNEL_RADIUS_ALIGNED and dataW 
// are multiples of maximum coalescable read/write size,
// all global memory operations are coalesced in convolutionRowGPU()
#define            ROW_TILE_W 128
#define KERNEL_RADIUS_ALIGNED 16

// Assuming COLUMN_TILE_W and dataW are multiples
// of maximum coalescable read/write size, all global memory operations 
// are coalesced in convolutionColumnGPU()
#define COLUMN_TILE_W 16
#define COLUMN_TILE_H 48
*/
#define KERNEL_RADIUS 1
#define      KERNEL_W (2 * KERNEL_RADIUS + 1)
__device__ __constant__ float d_Kernel[KERNEL_W];

// Assuming ROW_TILE_W, KERNEL_RADIUS_ALIGNED and dataW 
// are multiples of maximum coalescable read/write size,
// all global memory operations are coalesced in convolutionRowGPU()
#define            ROW_TILE_W 128
#define KERNEL_RADIUS_ALIGNED 16

// Assuming COLUMN_TILE_W and dataW are multiples
// of maximum coalescable read/write size, all global memory operations 
// are coalesced in convolutionColumnGPU()
#define COLUMN_TILE_W 16
#define COLUMN_TILE_H 48


////////////////////////////////////////////////////////////////////////////////
// Loop unrolling templates, needed for best performance
////////////////////////////////////////////////////////////////////////////////
template<int i> __device__ float convolutionRow(float *data){
    return
        data[KERNEL_RADIUS - i] * d_Kernel[i]
        + convolutionRow<i - 1>(data);
}

template<> __device__ float convolutionRow<-1>(float *data){
    return 0;
}

template<int i> __device__ float convolutionColumn(float *data){
    return 
        data[(KERNEL_RADIUS - i) * COLUMN_TILE_W] * d_Kernel[i]
        + convolutionColumn<i - 1>(data);
}

template<> __device__ float convolutionColumn<-1>(float *data){
    return 0;
}



////////////////////////////////////////////////////////////////////////////////
// Row convolution filter
////////////////////////////////////////////////////////////////////////////////
__global__ void convolutionRowGPU(
    float *d_Result,
    float *d_Data,
    int dataW,
    int dataH
){
    //Data cache
    __shared__ float data[KERNEL_RADIUS + ROW_TILE_W + KERNEL_RADIUS];

    //Current tile and apron limits, relative to row start
    const int         tileStart = blockIdx.x*ROW_TILE_W;
    const int           tileEnd = tileStart + ROW_TILE_W - 1;
    const int        apronStart = tileStart - KERNEL_RADIUS;
    const int          apronEnd = tileEnd   + KERNEL_RADIUS;

    //Clamp tile and apron limits by image borders
    const int    tileEndClamped = min(tileEnd, dataW - 1);
    const int apronStartClamped = max(apronStart, 0);
    const int   apronEndClamped = min(apronEnd, dataW - 1);

    //Row start index in d_Data[]
    const int          rowStart = blockIdx.y*dataW;

    //Aligned apron start. Assuming dataW and ROW_TILE_W are multiples 
    //of half-warp size, rowStart + apronStartAligned is also a 
    //multiple of half-warp size, thus having proper alignment 
    //for coalesced d_Data[] read.
    const int apronStartAligned = tileStart - KERNEL_RADIUS_ALIGNED;

    const int loadPos = apronStartAligned + threadIdx.x;
    //Set the entire data cache contents
    //Load global memory values, if indices are within the image borders,
    //or initialize with zeroes otherwise
    if(loadPos >= apronStart){
        const int smemPos = loadPos - apronStart;

        data[smemPos] = 
            ((loadPos >= apronStartClamped) && (loadPos <= apronEndClamped)) ?
            d_Data[rowStart + loadPos] : 0;
    }

    //Ensure the completness of the loading stage
    //because results, emitted by each thread depend on the data,
    //loaded by another threads
    __syncthreads();

    const int writePos = tileStart + threadIdx.x;
    //Assuming dataW and ROW_TILE_W are multiples of half-warp size,
    //rowStart + tileStart is also a multiple of half-warp size,
    //thus having proper alignment for coalesced d_Result[] write.
    if(writePos <= tileEndClamped){
        const int smemPos = writePos - apronStart;
        float sum = 0;

        sum = convolutionRow<2 * KERNEL_RADIUS>(data + smemPos);
        //for(int k = -KERNEL_RADIUS; k <= KERNEL_RADIUS; k++)
        //    sum += data[smemPos + k] * d_Kernel[KERNEL_RADIUS - k];

        d_Result[rowStart + writePos] = sum;
    }
}



////////////////////////////////////////////////////////////////////////////////
// Column convolution filter
////////////////////////////////////////////////////////////////////////////////
__global__ void convolutionColumnGPU(
    float *d_Result,
    float *d_Data,
    int dataW,
    int dataH,
    int smemStride,
    int gmemStride
){
    //Data cache
    __shared__ float data[COLUMN_TILE_W * (KERNEL_RADIUS + COLUMN_TILE_H + KERNEL_RADIUS)];

    //Current tile and apron limits, in rows
    const int         tileStart = blockIdx.y*COLUMN_TILE_H;
    const int           tileEnd = tileStart + COLUMN_TILE_H - 1;
    const int        apronStart = tileStart - KERNEL_RADIUS;
    const int          apronEnd = tileEnd   + KERNEL_RADIUS;

    //Clamp tile and apron limits by image borders
    const int    tileEndClamped = min(tileEnd, dataH - 1);
    const int apronStartClamped = max(apronStart, 0);
    const int   apronEndClamped = min(apronEnd, dataH - 1);

    //Current column index
    const int       columnStart = (blockIdx.x*COLUMN_TILE_W) + threadIdx.x;

    //Shared and global memory indices for current column
    int smemPos = (threadIdx.y*COLUMN_TILE_W) + threadIdx.x;
    int gmemPos = ((apronStart + threadIdx.y)*dataW) + columnStart;
    //Cycle through the entire data cache
    //Load global memory values, if indices are within the image borders,
    //or initialize with zero otherwise
    for(int y = apronStart + threadIdx.y; y <= apronEnd; y += blockDim.y){
        data[smemPos] = 
        ((y >= apronStartClamped) && (y <= apronEndClamped)) ? 
        d_Data[gmemPos] : 0;
        smemPos += smemStride;
        gmemPos += gmemStride;
    }

    //Ensure the completness of the loading stage
    //because results, emitted by each thread depend on the data, 
    //loaded by another threads
    __syncthreads();
    
    //Shared and global memory indices for current column
    smemPos = ((threadIdx.y + KERNEL_RADIUS)*COLUMN_TILE_W) + threadIdx.x;
    gmemPos = ((tileStart + threadIdx.y)*dataW) + columnStart;
    //Cycle through the tile body, clamped by image borders
    //Calculate and output the results
    for(int y = tileStart + threadIdx.y; y <= tileEndClamped; y += blockDim.y){
        float sum = 0;

        sum = convolutionColumn<2 * KERNEL_RADIUS>(data + smemPos);
        //for(int k = -KERNEL_RADIUS; k <= KERNEL_RADIUS; k++)
        //    sum += 
        //        data[smemPos + IMUL(k, COLUMN_TILE_W)] *
        //        d_Kernel[KERNEL_RADIUS - k];

        d_Result[gmemPos] = sum;
        smemPos += smemStride;
        gmemPos += gmemStride;
    }
}
