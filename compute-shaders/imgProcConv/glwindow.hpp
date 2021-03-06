#ifndef __GLWINDOW__
#define __GLWINDOW__

#include <GL/glew.h>

#include <rendertexture.hpp>
#include "kernel.hpp"

//#include <opencv/cv.hpp>
//#include <highgui.h>

using namespace std;

class GLWindow 
{

public:

    explicit GLWindow(void)
    {
    	frameTexture = NULL;
    	initd = false;
    	hastexture = false;
   	}

    ~GLWindow()
    {
    	delete frameTexture;
    }
    
    /**
     * @brief Initializes the window
	 * @param width Window width in pixels
	 * @param height Window height in pixels
     */
    void initialize(int width, int height)
    {
    	// the default is /shaders relative to your running dir
	    string shaders_dir("../shaders/");

	    //Set viewportSize vector
	    viewportSize = Eigen::Vector2i(width,height);
	    channels = 0;

	    // set effects
	    rendertexture.setShadersDir(shaders_dir);
	    rendertexture.initialize();
	    k.setShadersDir(shaders_dir);
	    k.initialize();

	    /// set this widget as initialized
    	initd = true;
    }

    virtual void setTexture(float* frame, int spectrum, int width = 0, int height = 0, int depth = 0)
    {
    	frameTexture = new Tucano::Texture;
    	channels = spectrum;
		
	    try
	    {
	    	switch (channels)
	    	{
	    		case 1:
	    			frameTexture->create(GL_TEXTURE_2D, GL_RED, width, height, GL_RED, GL_FLOAT, frame);
	    			break;
	    		case 3:
	    			frameTexture->create(GL_TEXTURE_2D, GL_RGB, width, height, GL_RGB, GL_FLOAT, frame);
	    			break;
	    	}
	    }
	    catch( exception& e)
	    {
	    	throw;
	    	std::cout<<"ERROR in setTexture: "<<e.what()<<std::endl;
	    	Tucano::Misc::errorCheckFunc(__FILE__, __LINE__);
	    }

	    viewportSize << width, height;

	    hastexture = true;
    }

    virtual void directUpdateFrame(float* frame)
    {
    	frameTexture->update(frame);
    }

    void gradient(float* frame, int spectrum, int width = 0, int height = 0, int depth = 0)
    {
    	if(!initd) {
    		std::cout<<"GL Windows has not been initialized yet!"<<std::endl;
	        return;
    	}

    	frameTexture = new Tucano::Texture;
    	tempTexture = new Tucano::Texture;
    	channels = spectrum;

    	try
	    {
	    	switch (channels)
	    	{
	    		case 1:
	    			frameTexture->create(GL_TEXTURE_2D, GL_RED, width, height, GL_RED, GL_FLOAT, 0);
	    			tempTexture->create(GL_TEXTURE_2D, GL_RED, width, height, GL_RED, GL_FLOAT, frame);
	    			break;
	    		case 3:
	    			frameTexture->create(GL_TEXTURE_2D, GL_RGB, width, height, GL_RGB, GL_FLOAT, 0);
	    			tempTexture->create(GL_TEXTURE_2D, GL_RGB, width, height, GL_RGB, GL_FLOAT, frame);
	    			break;
	    	}
	    }
	    catch( exception& e)
	    {
	    	throw;
	    	std::cout<<"ERROR in setTexture: "<<e.what()<<std::endl;
	    	Tucano::Misc::errorCheckFunc(__FILE__, __LINE__);
	    }

	    viewportSize << width, height;

	    k.gradient(tempTexture, frameTexture, viewportSize);

	    hastexture = true;
    }

    /**
     * Repaints screen buffer.
     **/
    virtual void paint()
    {
		if(!initd)
	        return;

		glClearColor(1.0, 1.0, 1.0, 0.0);
	    glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT);		

		// renders the given image, not that we are setting a fixed viewport that follows the widgets size
	    // so it may not be scaled correctly with the image's size (just to keep the example simple)
	    //Eigen::Vector2i viewport (viewportSize[0], viewportSize[1]);
	    if (hastexture)
	    {
	    	rendertexture.renderTexture(*frameTexture, viewportSize);
	    }
    }

	Tucano::Texture* texPointer()
	{
		return frameTexture;
	}

private:

	// A simple phong shader for rendering meshes
    //Effects::GL GL;

	/// Render image effect (simply renders a texture)
    Effects::RenderTexture rendertexture;

    /// 
    Effects::Kernels k;

    /// Texture to hold input image
    Tucano::Texture* frameTexture;
    Tucano::Texture* tempTexture;

    /// Path where shaders are stored
    string shaders_dir;

    /// Viewport size
    Eigen::Vector2i viewportSize;
    int channels;

    /// has this been initialized?
	bool initd;

	/// has texture
	bool hastexture;
};

#endif // GLWINDOW
