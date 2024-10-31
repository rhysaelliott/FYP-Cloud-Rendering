#version 450
#extension GL_GOOGLE_include_directive : require
#include "input_structures.glsl"
#define PI 3.1415926

layout (location =0) in vec3 inNormal;
layout(location=1) in vec3 inColor;
layout(location =2) in vec2 inUV;
layout(location =3) in vec3 inPos;
layout(location=4) in flat int inInstanceIndex;


layout(location=0) out vec4 outFragColor;



float saturate(in float num)
{
	return clamp(num, 0.0,1.0);
}


void main()
{
	vec4 base;
	switch(inInstanceIndex)
	{
	case 0:
		base = texture(colorTex,inUV);
	break;
	case 1:
		base = texture(metalRoughTex,inUV);
	break; 
		case 2:
		base = texture(cloudTex3,inUV);
	break; 
		case 3:
		base = texture(cloudTex4,inUV);
	break; 
		case 4:
		base = texture(cloudTex5,inUV);
	break; 
	case 5:
		base = texture(cloudTex6,inUV);
	break; 
		case 6:
		base = texture(cloudTex7,inUV);
	break; 
		case 7:
		base = texture(cloudTex8,inUV);
	break; 
		case 8:
		base = texture(cloudTex9,inUV);
	break; 
	default:
	base = texture(metalRoughTex, inUV);
	break;
	}
vec4 threshold = vec4(0.1);
 if((base.r < threshold.x && base.g < threshold.y && base.b < threshold.z) || base.a < threshold.w) {
     discard;
 }

	float lightValue = max(dot(inNormal, sceneData.sunlightDirection.xyz), 0.1f);
	lightValue = mix(0.1f, lightValue, lightValue);

	vec3 color = base.xyz;
	vec3 ambient = color * sceneData.ambientColor.xyz * 0.5; //0.5 as stand in for density

	float rim = pow(1.0-lightValue,3.0);
	vec3 rimColor = sceneData.sunlightColor.xyz * rim * 0.2;



	outFragColor =vec4(color, base.a);
}
