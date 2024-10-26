#version 450
#extension GL_GOOGLE_include_directive : require
#extension GL_EXT_buffer_reference : require

#include "input_structures.glsl"

layout(location =0) out vec3 outNormal;
layout(location =1) out vec3 outColor;
layout(location = 2) out vec2 outUV;
layout(location = 3) out vec3 outPos;

struct Vertex
{
	vec3 position;
	float uv_x;
	vec3 normal;
	float uv_y;
	vec4 color;
};

layout(buffer_reference, std430) readonly buffer VertexBuffer
{
	Vertex vertices[];
};

layout(push_constant) uniform constants
{
	mat4 render_matrix;
	VertexBuffer vertexBuffer;
} PushConstants;

layout(set=2, binding= 0) uniform BillboardData
{
	vec4 position[10];

} billboardData;


void main()
{
	Vertex v = PushConstants.vertexBuffer.vertices[gl_VertexIndex];
	
	vec4 position = vec4(v.position + billboardData.position[gl_InstanceIndex].xyz,1.0);

	mat4 translationMatrix = mat4(1.0);
	translationMatrix[3] = vec4(billboardData.position[gl_InstanceIndex].xyz,1.0);


	mat4 modelView = sceneData.view *translationMatrix *PushConstants.render_matrix;
	modelView[0][0] = 1;
	modelView[0][1] = 0;
	modelView[0][2] = 0;

	modelView[1][0] = 0;
	modelView[1][1] = 1;
	modelView[1][2] = 0;

	modelView[2][0] = 0;
	modelView[2][1] = 0;
	modelView[2][2] = 1;

	gl_Position = sceneData.proj * modelView*position;
	outPos= (modelView*position).xyz;

	outNormal =	(PushConstants.render_matrix * vec4(v.normal,0.f)).xyz;
	outColor = v.color.xyz;

	outColor=vec3(1,0,0);

	outUV.x = v.uv_x;
	outUV.y = v.uv_y;
}