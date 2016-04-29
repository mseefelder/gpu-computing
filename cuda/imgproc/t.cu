//Download CImg.h and put it on the same folder.
//in Linux, compile with:
// g++ main.cpp -o imageViewer -lX11 -pthreads

#include <iostream>

__global__
void kernel(int x, int y, int s, float *img)
{
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  if (i < (x*y*s)) img[i] = (img[i] > 125) ? 255 : 0;
}

float* t (float *f, int x, int y, int s) {
  //for error checking
  cudaError_t lerror = cudaSuccess;

  //Number of pixels
  int imgSize = x*y*s;
  int imgSizeOnMem = imgSize*sizeof(float);
  
  //Image on device
  float *d_img;
  //Alloc memory on device
  lerror = cudaMalloc(&d_img, imgSizeOnMem);

  //transfer from host to device:
  lerror = cudaMemcpy(d_img, f, imgSizeOnMem, cudaMemcpyHostToDevice);

  kernel<<<(imgSize+255)/256, 256>>>(x, y, s, d_img);

  //Image on host
  float *img = new float[imgSize];
  //Alloc memory on host
  //img = (float*)malloc(imgSizeOnMem);

  //transfer from device to host
  lerror = cudaMemcpy(img, d_img, imgSizeOnMem, cudaMemcpyDeviceToHost);

  return img;
}