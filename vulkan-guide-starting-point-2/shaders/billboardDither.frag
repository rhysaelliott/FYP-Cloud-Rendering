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

const float dither[4][4] = {
    {0.0625, 0.5625, 0.1875, 0.6875},
    {0.8125, 0.3125, 0.9375, 0.4375},
    {0.25,   0.75,   0.125,  0.625},
    {1.0,    0.5,    0.875,  0.375}
};

void main()
{

	vec4 base;
	int b = int(billboardData.texIndex[inInstanceIndex/4][inInstanceIndex%4]);
	switch(b)
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

	float ditherValue = dither[int(mod(gl_FragCoord.y, 4.0))][int(mod(gl_FragCoord.x, 4.0))];


	float lightValue = max(dot(inNormal, sceneData.sunlightDirection.xyz), 0.1f);
	lightValue = mix(0.1f, lightValue, lightValue);

	vec3 color = base.xyz;
	vec3 ambient = color * sceneData.ambientColor.xyz * ((base.r+base.g+base.b)/3.0);

	float rim = pow(1.0-lightValue,3.0);
	vec3 rimColor = sceneData.sunlightColor.xyz * rim * 0.2;

	float distanceToCamera =length(inPos.xyz - inverse(sceneData.view)[3].xyz);

	float fadeStart =10.0 * billboardData.scale[inInstanceIndex/4][inInstanceIndex%4];
	float fadeEnd = 5.0;

	float fadeFactor = saturate((distanceToCamera-fadeEnd)/(fadeStart-fadeEnd));

	if( pow( base.a,0.6) * fadeFactor<ditherValue)discard;

	outFragColor =vec4(color+(lightValue*sceneData.sunlightColor.xyz)+ambient + rimColor, 1.0);
}