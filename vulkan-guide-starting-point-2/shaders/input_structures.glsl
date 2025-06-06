
layout(set =1, binding =1) uniform sampler2D colorTex;
layout(set =1, binding =2) uniform sampler2D metalRoughTex;
layout(set =1, binding =3) uniform sampler2D cloudTex3;
layout(set =1, binding =4) uniform sampler2D cloudTex4;
layout(set =1, binding =5) uniform sampler2D cloudTex5;
layout(set =1, binding =6) uniform sampler2D cloudTex6;
layout(set =1, binding =7) uniform sampler2D cloudTex7;
layout(set =1, binding =8) uniform sampler2D cloudTex8;
layout(set =1, binding =9) uniform sampler2D cloudTex9;

layout(set =0, binding =0) uniform SceneData
{
	mat4 view;
	mat4 proj;
	mat4 viewproj;
	vec4 ambientColor;
	vec4 sunlightDirection;
	vec4 sunlightColor;
    vec4 cameraPos;
} sceneData;

layout(set=1, binding=0) uniform GLTFMaterialData
{
	vec4 colorFactors;
	vec4 metalRoughFactors;
} materialData;


struct LightStruct
{
    vec3 position;
    float cone;         
    vec3 color;
    float range;        
    vec3 direction;
    float intensity;   
    float constant;
    float linear;
    float quadratic;
    uint lightType; 
};

#define NUM_OF_BILLBOARDS 128

layout(set=2, binding= 0) uniform BillboardData
{
	vec4 position[NUM_OF_BILLBOARDS];
    vec4 scale[NUM_OF_BILLBOARDS/4];   
    vec4 texIndex[NUM_OF_BILLBOARDS/4]; 

} billboardData;

layout(set=2, binding=0) uniform LightData
{
	LightStruct lights[10];
	int numLights;
} lightData;
