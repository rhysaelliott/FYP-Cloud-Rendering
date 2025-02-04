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
	float time;

	float silverIntensity;
	float silverSpread;

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
	return ((1.0-g)/ pow((1.0+g*g-2.0*g*angle),3.0/2.0))/4.0*3.1459;
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

	float cosAngle = cos(dot(rayDir,toSun));
	float eccentricity=0.99;

	float phase = max(HenyeyGreenstein(eccentricity, cosAngle), voxelInfo.silverIntensity*HenyeyGreenstein(cosAngle,0.99-voxelInfo.silverSpread)) ;
	
	float stepSize = 1.0;
	float stepMax = 512.0;
	float sunStepSize = 1.0;
	float sunStepMax = 512.0;


	float I =0.0; //illumination
	float sunI =0.0;
	float transmit = 1.0;
	float sunTransmit =1.0;

	while(tMin<=stepMax &&I<0.7 && transmit>0.0)
	{
		tMin+=stepSize;
		float jitter =(random((gl_FragCoord.xy)*voxelInfo.time - 0.5)) * stepSize;
		tMin += jitter;
		vec3 samplePos = rayOrigin+ (rayDir*tMin);

		if (!(samplePos.x >= voxelGridMin.x && samplePos.x <= voxelGridMax.x &&
            samplePos.y >= voxelGridMin.y && samplePos.y <= voxelGridMax.y &&
            samplePos.z >= voxelGridMin.z && samplePos.z <= voxelGridMax.z)) continue; 
        
		vec3 uvw = (samplePos - voxelGridMin) / (voxelGridMax - voxelGridMin);
		uvw = clamp(uvw, vec3(0),vec3(1));
		float density =vec3(texture(voxelBuffer, uvw)).r * 2.0;

		if(density>0.0)
		{
			while(sunTMin<sunStepMax && sunTransmit >0.0  )
			{
				sunTMin+=sunStepSize;
				jitter =(random((gl_FragCoord.xy)*voxelInfo.time - 0.5)) * stepSize;
				sunTMin+=jitter;
				vec3 sunSamplePos = (samplePos) + (toSun*sunTMin);

						if (!(sunSamplePos.x >= voxelGridMin.x && sunSamplePos.x <= voxelGridMax.x &&
            			sunSamplePos.y >= voxelGridMin.y && sunSamplePos.y <= voxelGridMax.y &&
            			sunSamplePos.z >= voxelGridMin.z && sunSamplePos.z <= voxelGridMax.z)) continue;

						uvw = (sunSamplePos-voxelGridMin) / (voxelGridMax - voxelGridMin);


						float sunDensity = vec3(texture(voxelBuffer, uvw)).r*sunStepSize;

						sunTransmit *= beer(sunDensity) ;
						sunI+= sunTransmit * powder(sunDensity) * phase;
			}

			I+=  transmit* phase * powder(density);
			I+=  max((sunI*0.05), 0.001);
			transmit*= (max((beer(density) + powder(density)), beer(density*0.25)*0.7) * (1- voxelInfo.outScatterMultiplier));
			transmit*=(sunTransmit);

		}
	}

	vec3 finalColor = (sunlightColor * I) + (backgroundColor * transmit) ;


	outFragColor =vec4(finalColor , 1.0);
}
