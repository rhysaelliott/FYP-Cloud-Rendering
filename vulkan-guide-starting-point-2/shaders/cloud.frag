#version 450
#extension GL_GOOGLE_include_directive : require
#include "input_structures.glsl"
#define PI 3.1415926

layout(set =1, binding =10) uniform sampler3D voxelBuffer;
layout(set=2, binding=0) uniform VoxelInfo
{
	vec4 centrePos;
	vec4 bounds;
} voxelInfo;

layout(location =0) in vec3 inPos;
layout (location =1) in vec3 rayOrigin;




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
	float stepSize =0.1;
	float tMin=0;
	float tMax=100;

	
	vec3 voxelGridCentre = voxelInfo.centrePos.xyz;
	vec3 voxelDimension = voxelInfo.bounds.xyz;

	vec3 voxelGridMin = voxelGridCentre - voxelDimension*0.5;
	vec3 voxelGridMax = voxelGridCentre + voxelDimension*0.5;

	vec3 rayDir = normalize(inPos -rayOrigin);

	vec3 backgroundColor = vec3(0.53, 0.81, 0.98);

	float accumulatedDensity =0.0;
	float T =1.0;
	float sigma_a =0.05; //absorbtion 
	float sigma_s =0.05; //scattering
	while(tMin<tMax && accumulatedDensity<1.0)
	{
		vec3 samplePos = rayOrigin+ (rayDir*tMin); // worldspace
		tMin+=stepSize;
		if (!(samplePos.x >= voxelGridMin.x && samplePos.x <= voxelGridMax.x &&
            samplePos.y >= voxelGridMin.y && samplePos.y <= voxelGridMax.y &&
            samplePos.z >= voxelGridMin.z && samplePos.z <= voxelGridMax.z)) continue;
        
		vec3 uvw = (samplePos - voxelGridMin) / (voxelGridMax - voxelGridMin);
		float density =vec3(texture(voxelBuffer, (uvw))).r;
		T *= exp(-stepSize * density*(sigma_a+sigma_s));

		accumulatedDensity+=density;
	}


	vec3 volumeColor = vec3(0.1,0.1,0.1);
	vec3 backgroundColorThroughVolume =  T * backgroundColor + (1-T)*volumeColor;

	backgroundColorThroughVolume = vec3(accumulatedDensity);
	

	outFragColor =vec4(backgroundColorThroughVolume , 1.0);
}
