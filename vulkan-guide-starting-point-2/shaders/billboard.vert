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
	vec4 position[128];

} billboardData;


void main()
{
    Vertex v = PushConstants.vertexBuffer.vertices[gl_VertexIndex];
    
    vec4 vertexPosition = vec4(v.position, 1.0);

    mat4 translationMatrix = mat4(1.0);
    translationMatrix[3] = vec4(billboardData.position[gl_InstanceIndex].xyz, 1.0);


    vec3 cameraPosition = inverse(sceneData.view)[3].xyz;
    vec3 billboardPosition = billboardData.position[gl_InstanceIndex].xyz;
    vec3 cameraDirection = normalize(billboardPosition - cameraPosition);

    vec3 upVector = vec3(0.0, 1.0, 0.0);
    vec3 rightVector = normalize(cross(upVector, cameraDirection));


    mat4 rotationMatrix = mat4(
        vec4(rightVector, 0.0),
        vec4(upVector, 0.0),
        vec4(-cameraDirection, 0.0),
        vec4(0.0, 0.0, 0.0, 1.0)
    );

    mat4 modelView = sceneData.view * (translationMatrix * rotationMatrix * PushConstants.render_matrix);

    gl_Position = sceneData.proj * modelView * vertexPosition;
    outPos = (modelView * vertexPosition).xyz;

    outNormal = (PushConstants.render_matrix * vec4(v.normal, 0.0)).xyz;
    outColor = v.color.xyz;


    outUV.x = v.uv_x;
    outUV.y = v.uv_y;
}