#ifndef __KERNELS__
#define __KERNELS__

#include <tucano.hpp>

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
		loadShader(gradientX, "gradientX");
        loadShader(gradientY, "gradientY");
        quad.createQuad();
    }

    /**
     * @brief Renders the given texture.
     *
     * Renders the given texture using a proxy geometry, a quad the size of the viewport
     * to hold the texture.
     */
    void gradient (Eigen::Vector2i viewport, Eigen::Vector2i firstCorner, Eigen::Vector2i spread)
    {
        gradientX.bind();
        
        
    }

private:

    /// The square rendering shader.
    Shader gradientX;
    Shader gradientY;

};

}

#endif
