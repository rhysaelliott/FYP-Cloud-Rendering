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
} voxelInfo;

layout(location =0) in vec3 inPos;
layout (location =1) in vec3 rayOrigin;

layout(location=0) out vec4 outFragColor;


float saturate(in float num)
{
	return clamp(num, 0.0,1.0);
}

float random(vec2 uv) {
    return fract(sin(dot(uv, vec2(12.9898, 78.233))) * 43758.5453123);
}

float HenyeyGreenstein(float cos_angle, float g)
{
	float g2 = g*g;
	float val = ((1.0-g2)/pow(1.0+g2-2.0*g*cos_angle,1.5))/4 * PI;
	return val;
}

void main()
{
	float stepSize =0.5;
	float tMin=0;
	float tMax=128;

	vec3 voxelGridCentre = voxelInfo.centrePos.xyz;
	vec3 voxelDimension = voxelInfo.bounds.xyz;

	vec3 voxelGridMin = voxelGridCentre - voxelDimension*0.5;
	vec3 voxelGridMax = voxelGridCentre + voxelDimension*0.5;

	vec3 rayDir = normalize(inPos -rayOrigin);

	vec2 backgroundUV = (gl_FragCoord).xy / voxelInfo.screenRes;
	vec3 backgroundColor = texture(backgroundTex,backgroundUV).xyz;

	vec3 sunlightColor = sceneData.sunlightColor.xyz;
	vec3 sunlightDir = sceneData.sunlightDirection.xyz;

	vec3 lightDir = normalize(sceneData.sunlightDirection).xyz;
	float cosAngle = dot(rayDir,sunlightDir);
	float phase = mix(HenyeyGreenstein(cosAngle,-0.3), HenyeyGreenstein(cosAngle,0.3),0.7);


	float mu = 0.5+0.5*dot(rayDir, sunlightDir);

	vec3 T =vec3(1.0); //total transmittance
	float sigma_a =0.05; //absorbtion 
	float sigma_s =0.05; //scattering
	float sigma_t = sigma_a+sigma_s;

	vec3 color = vec3(0.3); 

	float I =0.0; //illumination
	float transmit = 1.0;

	float accumulatedDensity=0.0;

	while(tMin<=tMax &&accumulatedDensity<1.0 )
	{
		float jitter =(random(gl_FragCoord.xy) - 0.5) * stepSize;
		tMin += jitter;
		vec3 samplePos = rayOrigin+ (rayDir*tMin);
		tMin+=stepSize;

		if (!(samplePos.x >= voxelGridMin.x && samplePos.x <= voxelGridMax.x &&
            samplePos.y >= voxelGridMin.y && samplePos.y <= voxelGridMax.y &&
            samplePos.z >= voxelGridMin.z && samplePos.z <= voxelGridMax.z)) continue;
        
		vec3 uvw = (samplePos - voxelGridMin) / (voxelGridMax - voxelGridMin);
		uvw = clamp(uvw, vec3(0),vec3(1));
		float density =vec3(texture(voxelBuffer, (uvw))).r * stepSize;

		if(density>0.0)
		{
			accumulatedDensity+=density;

			I+= density * transmit * phase;
			transmit*= exp(-(density * (1- 0.05)));
		}



	}


	vec3 finalColor = (sunlightColor * I) + backgroundColor * transmit;
	//finalColor = vec3(accumulatedDensity);
	//todo film mapping and gamma correction


	outFragColor =vec4(finalColor , 1.0);
}
