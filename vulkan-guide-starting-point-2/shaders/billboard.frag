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
	float b = billboardData.texIndex[inInstanceIndex/4][inInstanceIndex%4];
if(b == 0)
{
		base = texture(colorTex,inUV);
}
else if(b==1)
{
		base = texture(metalRoughTex,inUV);
}
else if(b==2)
{
		base = texture(cloudTex3,inUV);
}
else if(b==3)
{
		base = texture(cloudTex4,inUV);
}
else if(b==4)
{

		base = texture(cloudTex5,inUV);
}
else if(b==5)
{

		base = texture(cloudTex6,inUV);
}
else if(b==6)
{
		base = texture(cloudTex7,inUV);
}
else if(b==7)
{
		base = texture(cloudTex8,inUV);
}else if(b==8)
{

		base = texture(cloudTex9,inUV);
}else 
{
		base = texture(metalRoughTex,inUV);
}

//base = texture(cloudTex7,inUV);


	float lightValue = max(dot(inNormal, sceneData.sunlightDirection.xyz), 0.1f);
	lightValue = mix(0.1f, lightValue, lightValue);

	vec3 color = base.xyz;
	vec3 ambient = color * sceneData.ambientColor.xyz * 0.5; //0.5 as stand in for density

	float rim = pow(1.0-lightValue,3.0);
	vec3 rimColor = sceneData.sunlightColor.xyz * rim * 0.2;



	outFragColor =vec4(color+(lightValue*sceneData.sunlightColor.xyz)+ambient + rimColor, base.a);
}
