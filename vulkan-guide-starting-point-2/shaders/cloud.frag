#version 450
#extension GL_GOOGLE_include_directive : require
#include "input_structures.glsl"
#define PI 3.1415926

layout(set =1, binding =10) uniform sampler3D voxelBuffer;
layout(set =1, binding =11) uniform sampler2D backgroundTex;
layout(set=2, binding=0) uniform VoxelInfo
{
	vec4 centrePos;
	vec4 bounds;

	vec2 screenRes;
	float outScatterMultiplier;


} voxelInfo;

layout(location =0) in vec3 inPos;
layout (location =1) in vec3 rayOrigin;

layout(location=0) out vec4 outFragColor;


float saturate(in float num)
{
	return clamp(num, 0.0,1.0);
}

float beer(float d)
{
   return exp(-d);
}

float powder(float d)
{
	return 1.0 - exp(-d*2);
}

float random(vec2 uv) {
    return fract(sin(dot(uv, vec2(12.9898, 78.233))) * 43758.5453123);
}

float HenyeyGreenstein(float angle, float g)
{
	return (1.0f - pow(g,2)) / (4.0f * 3.14159 * pow(1 + pow(g, 2) - 2.0f * g * angle, 1.5f));
}

void main()
{
	float tMin=0;
	float sunTMin =0;
	float sunAccumulatedDensity=0;

	vec3 voxelGridCentre = voxelInfo.centrePos.xyz;
	vec3 voxelDimension = voxelInfo.bounds.xyz;

	vec3 voxelGridMin = voxelGridCentre - voxelDimension*0.5;
	vec3 voxelGridMax = voxelGridCentre + voxelDimension*0.5;

	vec3 rayDir = normalize(inPos -rayOrigin);

	vec2 backgroundUV = (gl_FragCoord).xy / voxelInfo.screenRes;
	vec3 backgroundColor = texture(backgroundTex,backgroundUV).xyz;

	vec3 sunlightColor = sceneData.sunlightColor.xyz;
	vec3 sunlightDir = normalize(sceneData.sunlightDirection.xyz);
	vec3 toSun = -sunlightDir;

	float cosAngle = dot(rayDir,toSun);
	float phase = HenyeyGreenstein(1, cosAngle) + HenyeyGreenstein(-1,cosAngle)/2.0 ;


	float I =0.0; //illumination
	float transmit = 1.0;
	float sunTransmit =1.0;
	
	float stepSize = 2.0;
	float stepMax = 128.0;
	float sunStepSize = 5.0;
	float sunStepMax = 0.0;


	while(tMin<=stepMax &&I<0.8&&transmit>0.1)
	{
		tMin+=stepSize;
		float jitter =(random(gl_FragCoord.xy) - 0.5) * stepSize;
		tMin += jitter;
		vec3 samplePos = rayOrigin+ (rayDir*tMin);


		if (!(samplePos.x >= voxelGridMin.x && samplePos.x <= voxelGridMax.x &&
            samplePos.y >= voxelGridMin.y && samplePos.y <= voxelGridMax.y &&
            samplePos.z >= voxelGridMin.z && samplePos.z <= voxelGridMax.z)) continue;
        
		vec3 uvw = (samplePos - voxelGridMin) / (voxelGridMax - voxelGridMin);
		uvw = clamp(uvw, vec3(0),vec3(1));
		float density =vec3(texture(voxelBuffer, uvw)).r * stepSize;

		if(density>0.0)
		{

			while(sunTMin<sunStepMax && sunAccumulatedDensity<0.8)
			{
				vec3 sunSamplePos = samplePos+ (toSun*sunTMin);
				sunTMin+=sunStepSize + jitter;

						if (!(sunSamplePos.x >= voxelGridMin.x && sunSamplePos.x <= voxelGridMax.x &&
            			sunSamplePos.y >= voxelGridMin.y && sunSamplePos.y <= voxelGridMax.y &&
            			sunSamplePos.z >= voxelGridMin.z && sunSamplePos.z <= voxelGridMax.z)) continue;

						uvw = (sunSamplePos-voxelGridMin) / (voxelGridMax - voxelGridMin);
						uvw=clamp(uvw, vec3(0),vec3(1));

						float sunDensity = vec3(texture(voxelBuffer, uvw)).r*sunStepSize;

						sunAccumulatedDensity+=sunDensity;
						sunTransmit *= beer(sunDensity * (1-voxelInfo.outScatterMultiplier));
			}

			I+= density * transmit * phase * sunTransmit ;
			transmit*= (beer(density)+powder(density)) * (1- voxelInfo.outScatterMultiplier);
		}
	}



	vec3 finalColor = (sunlightColor * I) + (backgroundColor * transmit) ;



	outFragColor =vec4(finalColor , 1.0);
}
