#version 430

uniform sampler2D imageTexture;
uniform ivec2 viewportSize;

out vec4 out_Color;

void main()
{
  vec2 texCoord = vec2(gl_FragCoord.xy/vec2(viewportSize.xy));
  vec4 tex = texture(imageTexture, texCoord+vec2(0.01,0));
  out_Color = vec4(vec3(tex.r),1.0);
}
