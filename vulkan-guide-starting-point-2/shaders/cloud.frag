#version 450
#extension GL_GOOGLE_include_directive : require
#include "input_structures.glsl"
#define PI 3.1415926

layout(set =1, binding =10) uniform sampler3D voxelBuffer;

layout(location =0) in vec3 inPos;

layout(location=0) out vec4 outFragColor;


float saturate(in float num)
{
	return clamp(num, 0.0,1.0);
}

float HenyeyGreenstein(float cos_angle, float g)
{
	float g2 = g*g;
	float val = ((1.0-g2)/pow(1.0+g2-2.0*g*cos_angle,1.5))/4 * PI;
	return val;
}



void main()
{
	vec3 backgroundColor = vec3(1.0,0.0,0.0);

	float sigma_a =0.1; //absorbtion 
	float distance =10;
	float T = exp(-distance*sigma_a);
	vec3 volumeColor = vec3(0,0,1);
	vec3 backgroundColorThroughVolume = T * backgroundColor + (1-T)*volumeColor;

	backgroundColor =vec3(texture(voxelBuffer, inPos));


	outFragColor =vec4(backgroundColorThroughVolume , 1.0f);
}
