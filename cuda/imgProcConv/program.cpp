#include <iostream>
#include "CImg.h"

float* t (float *f, int x, int y, int s);
int convolve(/*params*/);

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
  //float *data = new float[size];
  //for (int i = 0; i < size; ++i)
  //{
  //  data[i] = img1.data()[i];
  //}
  float *result = t(img1.data(), img1.width(), img1.height(), img1.spectrum());

  cimg_library::CImg<float> imgFinal(result, img1.width(), img1.height(), 1, img1.spectrum());

  //display final image
  /**/
  cimg_library::CImgDisplay img1_disp(imgFinal, "Image 1");
  while (!img1_disp.is_closed()) { 
  }
  /**/

  return 0;
}