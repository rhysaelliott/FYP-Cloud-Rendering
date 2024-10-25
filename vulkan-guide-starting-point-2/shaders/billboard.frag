#version 450
#extension GL_GOOGLE_include_directive : require
#include "input_structures.glsl"
#define PI 3.1415926

layout (location =0) in vec3 inNormal;
layout(location=1) in vec3 inColor;
layout(location =2) in vec2 inUV;
layout(location =3) in vec3 inPos;


layout(location=0) out vec4 outFragColor;



float saturate(in float num)
{
	return clamp(num, 0.0,1.0);
}




void main()
{

	vec3 base = texture(colorTex, inUV).xyz;
	base = inColor;



	outFragColor =vec4(base, 1.0f);
}
