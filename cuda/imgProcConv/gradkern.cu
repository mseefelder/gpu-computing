////////////////////////////////////////////////////////////////////////////////
// Kernel configuration
////////////////////////////////////////////////////////////////////////////////
#define GRADIENT_RADIUS 1
#define      GRADIENT_W (2 * GRADIENT_RADIUS + 1)
__device__ __constant__ float d_Gradient[GRADIENT_W];

// Assuming ROW_TILE_W, GRADIENT_RADIUS_ALIGNED and dataW 
// are multiples of maximum coalescable read/write size,
// all global memory operations are coalesced in convolutionRowGPU()
#define            GRAD_ROW_TILE_W 128
#define GRADIENT_RADIUS_ALIGNED 16

// Assuming COLUMN_TILE_W and dataW are multiples
// of maximum coalescable read/write size, all global memory operations 
// are coalesced in convolutionColumnGPU()
#define GRAD_COLUMN_TILE_W 16
#define GRAD_COLUMN_TILE_H 48


////////////////////////////////////////////////////////////////////////////////
// Loop unrolling templates, needed for best performance
////////////////////////////////////////////////////////////////////////////////
template<int i> __device__ float gradientRow(float *data){
    return
        data[GRADIENT_RADIUS - i] * d_Gradient[i]
        + gradientRow<i - 1>(data);
}

template<> __device__ float gradientRow<-1>(float *data){
    return 0;
}

template<int i> __device__ float gradientColumn(float *data){
    return 
        data[(GRADIENT_RADIUS - i) * GRAD_COLUMN_TILE_W] * d_Gradient[i]
        + gradientColumn<i - 1>(data);
}

template<> __device__ float gradientColumn<-1>(float *data){
    return 0;
}



////////////////////////////////////////////////////////////////////////////////
// Row convolution filter
////////////////////////////////////////////////////////////////////////////////
__global__ void gradientRowGPU(
    float *d_Result,
    float *d_Data,
    int dataW,
    int dataH
){
    //Data cache
    __shared__ float data[GRADIENT_RADIUS + GRAD_ROW_TILE_W + GRADIENT_RADIUS];

    //Current tile and apron limits, relative to row start
    const int         tileStart = blockIdx.x*GRAD_ROW_TILE_W;
    const int           tileEnd = tileStart + GRAD_ROW_TILE_W - 1;
    const int        apronStart = tileStart - GRADIENT_RADIUS;
    const int          apronEnd = tileEnd   + GRADIENT_RADIUS;

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
    const int apronStartAligned = tileStart - GRADIENT_RADIUS_ALIGNED;

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

        sum = gradientRow<2 * GRADIENT_RADIUS>(data + smemPos);
        //for(int k = -GRADIENT_RADIUS; k <= GRADIENT_RADIUS; k++)
        //    sum += data[smemPos + k] * d_Kernel[GRADIENT_RADIUS - k];

        d_Result[rowStart + writePos] = sum;
    }
}



////////////////////////////////////////////////////////////////////////////////
// Column convolution filter
////////////////////////////////////////////////////////////////////////////////
__global__ void gradientColumnGPU(
    float *d_Result,
    float *d_Data,
    int dataW,
    int dataH,
    int smemStride,
    int gmemStride
){
    //Data cache
    __shared__ float data[GRAD_COLUMN_TILE_W * (GRADIENT_RADIUS + GRAD_COLUMN_TILE_H + GRADIENT_RADIUS)];

    //Current tile and apron limits, in rows
    const int         tileStart = blockIdx.y*GRAD_COLUMN_TILE_H;
    const int           tileEnd = tileStart + GRAD_COLUMN_TILE_H - 1;
    const int        apronStart = tileStart - GRADIENT_RADIUS;
    const int          apronEnd = tileEnd   + GRADIENT_RADIUS;

    //Clamp tile and apron limits by image borders
    const int    tileEndClamped = min(tileEnd, dataH - 1);
    const int apronStartClamped = max(apronStart, 0);
    const int   apronEndClamped = min(apronEnd, dataH - 1);

    //Current column index
    const int       columnStart = (blockIdx.x*GRAD_COLUMN_TILE_W) + threadIdx.x;

    //Shared and global memory indices for current column
    int smemPos = (threadIdx.y*GRAD_COLUMN_TILE_W) + threadIdx.x;
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
    smemPos = ((threadIdx.y + GRADIENT_RADIUS)*GRAD_COLUMN_TILE_W) + threadIdx.x;
    gmemPos = ((tileStart + threadIdx.y)*dataW) + columnStart;
    //Cycle through the tile body, clamped by image borders
    //Calculate and output the results
    for(int y = tileStart + threadIdx.y; y <= tileEndClamped; y += blockDim.y){
        float sum = 0;

        sum = gradientColumn<2 * GRADIENT_RADIUS>(data + smemPos);
        //for(int k = -GRADIENT_RADIUS; k <= GRADIENT_RADIUS; k++)
        //    sum += 
        //        data[smemPos + IMUL(k, COLUMN_TILE_W)] *
        //        d_Kernel[GRADIENT_RADIUS - k];

        d_Result[gmemPos] = sum;
        smemPos += smemStride;
        gmemPos += gmemStride;
    }
}
