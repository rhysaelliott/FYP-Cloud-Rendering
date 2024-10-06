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

float phongDiffuse()
{
	return (1.0/PI);
}

float distributuionGGX(vec3 N, vec3 H, float roughness)
{
	float a = roughness*roughness;
	float a2 = a*a;
	float NdotH = max(dot(N,H), 0.0);
	float NdotH2 = NdotH * NdotH;

	float denom = (NdotH2 * (a2 - 1.0) +1.0);
	denom = PI* denom*denom;

	return a2/denom;
}

float geometrySchlickGGX(float NdotV, float roughness)
{
	float r = (roughness +1.0);
	float k = (r*r)/8.0;

	float denom = NdotV * (1.0-k) + k;

	return NdotV / denom;
}

float geometrySmith(vec3 N, vec3 V, vec3 L, float roughness)
{
	float NdotV = max(dot(N,V), 0.0);
	float NdotL = max(dot(N,L), 0.0);
	float ggx1 = geometrySchlickGGX(NdotV, roughness);
	float ggx2 = geometrySchlickGGX(NdotL, roughness);

	return ggx1*ggx2;
}



void main()
{
	float lightValue = max(dot(inNormal, sceneData.sunlightDirection.xyz), 0.1f);

	vec3 base = texture(colorTex, inUV).xyz;
	vec3 color = inColor * base;
	vec3 ambient = color * sceneData.ambientColor.xyz;




	vec3 lightColor = vec3(0.0);
	for(int i =0; i<lightData.numLights;i++)
	{
	LightStruct light = lightData.lights[i];
	float distance = length(light.position - inPos);
		if(light.range>=distance )
		{
			//get light position in local space
			//calculate light attenuation
			float A = light.constant + light.linear * distance + light.quadratic * (distance*distance);
			A = max(A, 0.001);
			A= 1.0/A;
			//calculate Light direction, View direction and Halfway vector vectors
			vec3 localLightPos = vec3(sceneData.view * vec4(light.position,1.0)).xyz;
			vec3 L = normalize(localLightPos - inPos);
			vec3 V = normalize(-inPos);
			vec3 H = normalize(L+V);
			vec3 nn = normalize(inNormal);

			//get roughness and metallic
			float metal = texture(metalRoughTex, inUV).x;
			float rough = texture(metalRoughTex, inUV).y;

			//mix between base and metallic for constant base specular value (0.04)
			vec3 baseSpec = mix(vec3(0.04), base , metal); 
			//compute material reflectance (frensel-schlik)
			vec3 F0 = vec3(baseSpec);
			vec3 F = F0 + (1.0-F0) * pow(1.0-dot(H,V),5.0);

			//do cook-torrance brdf
			float NDF = distributuionGGX(nn, H, rough);
			float G = geometrySmith(nn, V, L, rough);
			vec3 numerator = NDF * G * F;
			float denom = 4.0 *max(dot(nn, V), 0.0)* max(dot(nn, L), 0.0);
			vec3 spec = numerator / max(denom, 0.001);

			//do diffuse component (lambertian)
			vec3 kD = vec3(1.0) - F;
			kD *= 1.0-metal;

			//do diffuse 
			vec3 diffuse = kD * (color/PI);

			//calculate radiance
			vec3 radiance = light.color * A * light.intensity;

			//accumulate light contribution

			lightColor += (diffuse+spec) * radiance * max(dot(nn,L),0.0);	
		}
	}



	outFragColor =vec4(color*lightValue*sceneData.sunlightColor.w + ambient + lightColor, 1.0f);
}
