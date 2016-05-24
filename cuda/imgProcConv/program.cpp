#include <iostream>
#include <omp.h>
#include "CImg.h"

//float* t (float *f, int x, int y, int s);
void calculateGradient(float* h_ResultGPU, float* img, int imgWidth, int imgHeight);

int main(int argc, char* argv[]) {

  cimg_library::CImg<float> img1;

  if (argc < 2) 
  { // Check the value of argc. If not enough parameters have been passed, inform user and exit.
    std::cout << "Usage is: \n <image path>\n"<<
      "Press any key to close... \n";
      std::cin.get();
      return 0;
  } 
  else 
  {
    //try and open file
    try {
      img1.assign(argv[1]);
    } catch (cimg_library::CImgException &e) {
      std::cout << "Unable to open files. Use -h for help\n";
      std::cin.get();
      return 0;
    }
  }

  int size = img1.width()*img1.height()*img1.spectrum();

  int sizeBW = img1.width()*img1.height();
  float bW[sizeBW];

  int s = img1.spectrum();
  float *fPointer = img1.data();
  for (int i = 0; i < sizeBW; ++i)
  {
    float pix = .0;
    for (int j = 0; j < s; ++j)
    {
      pix += fPointer[i+(sizeBW*j)];
    }
    pix /= s*1.0;
    bW[i] = pix;
  }

  //float *data = new float[size];
  //for (int i = 0; i < size; ++i)
  //{
  //  data[i] = img1.data()[i];
  //}
  
  float result[sizeBW];
  calculateGradient(result, bW, img1.width(), img1.height());

  //float reorganizedResult[sizeBW];
  //for (int i = 0; i < sizeBW; i++)
  //{
  //  reorganizedResult[i]        = result[i*2];
  //  reorganizedResult[i+sizeBW] = result[(i*2)+1];
  //}

  cimg_library::CImg<float> imgFinal(result, img1.width(), img1.height(), 1, 1);

  //display final image
  /**/
  cimg_library::CImgDisplay img1_disp(imgFinal, "Image 1");
  while (!img1_disp.is_closed()) { 
  }
  /**/

  return 0;
}