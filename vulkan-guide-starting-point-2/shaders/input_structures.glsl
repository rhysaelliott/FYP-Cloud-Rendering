
layout(set =1, binding =1) uniform sampler2D colorTex;
layout(set =1, binding =2) uniform sampler2D metalRoughTex;

layout(set =0, binding =0) uniform SceneData
{
	mat4 view;
	mat4 proj;
	mat4 viewproj;
	vec4 ambientColor;
	vec4 sunlightDirection;
	vec4 sunlightColor;
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

layout(set=2, binding=0) uniform LightData
{
	LightStruct lights[10];
	int numLights;
} lightData;

layout(set =1, binding=3) uniform VolumetricData
{
    float test;
} volumetricData;
