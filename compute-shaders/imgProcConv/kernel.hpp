#ifndef __KERNELS__
#define __KERNELS__

#include <tucano.hpp>

//Also needs to be defined on shader!
#define GRAD_ROW_TILE_W 128

using namespace std;

using namespace Tucano;

namespace Effects
{

/**
 * @brief A simple effect to render a texture.
 **/
class Kernels : public Effect
{
public:
    /**
     * @brief Default Constructor.
     */
    Kernels (void)
    {
    }

    /**
     * @brief Deafult empty destructor.
     */
    ~Kernels (void) {}

    /**
     * @brief Initializes the effect, creating and loading the shader.
     */
    virtual void initialize()
    {
		//loadShader(gradientX, "gradientcolumn");
        loadShader(gradientY, "gradientrow");
        //quad.createQuad();
    }

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

    /**
     * @brief Renders the given texture.
     *
     * Renders the given texture using a proxy geometry, a quad the size of the viewport
     * to hold the texture.
     */
    void gradient (Tucano::Texture *input, Tucano::Texture *output, Eigen::Vector2i imageDimensions)
    {
        gradientY.bind();

        //Input texture:
        GLint inputUnit = input->bind();

        //Outpu texture:
        GLuint outputUnit = output->bind();
        glBindImageTexture(outputUnit, output->texID(), 0, GL_TRUE, 0, GL_READ_WRITE,
                           GL_R32F);

        //set uniforms:
        gradientY.setUniform("inputTexture", inputUnit);
        gradientY.setUniform("outputTexture", (GLint)outputUnit);
        gradientY.setUniform("dataW", imageDimensions[0]);
        gradientY.setUniform("dataH", imageDimensions[1]);

        Tucano::Misc::errorCheckFunc(__FILE__, __LINE__);

        //run smoothing:
        glDispatchCompute(iDivUp(imageDimensions[0], GRAD_ROW_TILE_W), imageDimensions[1], 1);

        glBindImageTexture(0, output->texID(), 0, GL_FALSE, 0, GL_WRITE_ONLY, GL_R32F);
        output->unbind();
        input->unbind();

        gradientY.unbind();
        
        return;
    }

private:

    /// The square rendering shader.
    Shader gradientX;
    Shader gradientY;

};

}

#endif
