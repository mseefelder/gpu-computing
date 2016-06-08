//#define THREAD 60

#include <iostream>
#include <tucano.hpp>
#include <omp.h>
#include "GLFW/glfw3.h"
#include "CImg.h"
#include "glwindow.hpp"

#ifdef THREAD
#include <thread>
#endif

int WINDOW_WIDTH = 100;
int WINDOW_HEIGHT = 100;
GLWindow* glWindow;

void initialize (void)
{
	Tucano::Misc::initGlew();
	glWindow = new GLWindow();	
	glWindow->initialize(WINDOW_WIDTH, WINDOW_HEIGHT);
	cout << "initialized" << endl;
}

/*
void keyCallback(GLFWwindow* window, int key, int scancode, int action, int mods)
{
	if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS)
		glfwSetWindowShouldClose(window, 1);
}

static void mouseButtonCallback (GLFWwindow* window, int button, int action, int mods)
{
    if (button == GLFW_MOUSE_BUTTON_LEFT && action == GLFW_PRESS)
	{
		double xpos, ypos;
		glfwGetCursorPos(window, &xpos, &ypos);
		Eigen::Vector2i mouse(xpos, abs(ypos-WINDOW_HEIGHT));
	}
	if (button == GLFW_MOUSE_BUTTON_LEFT && action == GLFW_RELEASE)
	{
		double xpos, ypos;
		glfwGetCursorPos(window, &xpos, &ypos);
		Eigen::Vector2i mouse(xpos, abs(ypos-WINDOW_HEIGHT));
	}
}
static void cursorPosCallback(GLFWwindow* window, double xpos, double ypos)
{
	if (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_LEFT) == GLFW_PRESS)
	{
		Eigen::Vector2i mouse(xpos, abs(ypos-WINDOW_HEIGHT));
	}
}
*/

int main(int argc, char *argv[])
{
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

	std::cout<<"Opened file"<<std::endl;

	long long sizeBW = img1.width()*img1.height();
	float *bW = new float[sizeBW];

	int s = img1.spectrum();
	float *fPointer = img1.data();
	
	#pragma omp parallel for
	for (int i = 0; i < sizeBW; ++i)
	{
		float pix = .0;
		for (int j = 0; j < s; ++j)
		{
		  pix += fPointer[i+(sizeBW*j)];
		}
		pix /= s*1.0;
		bW[sizeBW-1-i] = pix/255.;
	}

	GLFWwindow* main_window;

	if (!glfwInit()) 
	{
    	std::cerr << "Failed to init glfw" << std::endl;
		return 1;
	}

	main_window = glfwCreateWindow(WINDOW_WIDTH, WINDOW_HEIGHT, "Tracker", NULL, NULL);
	if (!main_window)
	{
		std::cerr << "Failed to create the GLFW window" << std::endl;
		glfwTerminate();
		return -1;
	}

	glfwMakeContextCurrent(main_window);
	/*
	glfwSetKeyCallback(main_window, keyCallback);
	glfwSetMouseButtonCallback(main_window, mouseButtonCallback);
	glfwSetCursorPosCallback(main_window, cursorPosCallback);
	*/
	glfwSetInputMode(main_window, GLFW_STICKY_KEYS, true);

	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
   	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
   	glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

	WINDOW_WIDTH = img1.width();
	WINDOW_HEIGHT = img1.height();

   	glfwSetWindowSize(main_window, WINDOW_WIDTH, WINDOW_HEIGHT);

	initialize();

	//glWindow->setTexture(bW, 1, img1.width(), img1.height());
	glWindow->gradient(bW, 1, img1.width(), img1.height());

	while (!glfwWindowShouldClose(main_window))
	{
		glfwMakeContextCurrent(main_window);
		glfwSwapBuffers(main_window);
		glWindow->paint();
		glfwPollEvents();
	}
	
	glfwDestroyWindow(main_window);
	glfwTerminate();

	return 0;
}
