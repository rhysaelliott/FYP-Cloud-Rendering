#version 450
#extension GL_GOOGLE_include_directive : require
#include "input_structures.glsl"
#define PI 3.1415926

layout(set =1, binding =10) uniform sampler3D voxelBuffer;
layout(set =1, binding =11) uniform sampler2D backgroundTex;
layout(set =1, binding =12) uniform sampler2D blueNoiseTex;
layout(set =1, binding =13) uniform sampler2D previousFrameTex;
layout(set=2, binding=0) uniform VoxelInfo
{
	vec4 centrePos;
	vec4 bounds;

	vec2 screenRes;
	float outScatterMultiplier;
	float time;

	float silverIntensity;
	float silverSpread;
	int reprojection;

	mat4 prevViewProj;

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

vec3 sample_cone(vec3 dir, float coneAngle, vec2 rand)
{
    float cosTheta = mix(1.0, cos(coneAngle), rand.x);
    float sinTheta = sqrt(1.0 - cosTheta * cosTheta);
    float phi = 6.2831853 * rand.y;

    vec3 localDir = vec3(cos(phi) * sinTheta, sin(phi) * sinTheta, cosTheta);

    vec3 up = abs(dir.z) < 0.999 ? vec3(0, 0, 1) : vec3(1, 0, 0);
    vec3 tangent = normalize(cross(up, dir));
    vec3 bitangent = cross(dir, tangent);

    return normalize(tangent * localDir.x + bitangent * localDir.y + dir * localDir.z);
}

bool insideBounds(vec3 p, vec3 minV, vec3 maxV)
{
    return all(greaterThanEqual(p, minV)) && all(lessThanEqual(p, maxV));
}

void main()
{

vec2 uv = gl_FragCoord.xy / voxelInfo.screenRes;
vec2 ndc = uv * 2.0 - 1.0;

vec4 currClipPos = vec4(ndc, 0.5, 1.0); 
vec4 worldPos = inverse(sceneData.viewproj) * currClipPos;
worldPos /= worldPos.w;

vec3 rayDir = normalize(worldPos.xyz - rayOrigin); 

float depth = 1000.0 ;
vec3 samplePoint = rayOrigin + rayDir * depth;

vec4 prevClip = voxelInfo.prevViewProj * vec4(samplePoint, 1.0);
vec2 prevUV = (prevClip.xy / prevClip.w) * 0.5 + 0.5;

vec3 noise = texture(blueNoiseTex, fract(uv)).rgb; 

int reprojection = int(floor(noise.r * 4.0));
bool valid = all(greaterThanEqual(prevUV, vec2(0.0))) && all(lessThanEqual(prevUV, vec2(1.0)));

//TODO when camera moves a lot when there is no clouds the background looks jank
if(voxelInfo.reprojection!=reprojection && valid)
{
	outFragColor = texture(previousFrameTex, prevUV);
	return;
}

	float tMin=0;
	float sunTMin =0;
	float sunAccumulatedDensity=0;

	vec3 voxelGridCentre = voxelInfo.centrePos.xyz;
	vec3 voxelDimension = voxelInfo.bounds.xyz;

	vec3 voxelGridMin = voxelGridCentre - voxelDimension*0.5;
	vec3 voxelGridMax = voxelGridCentre + voxelDimension*0.5;

	vec3 sunlightDir = normalize(sceneData.sunlightDirection.xyz);


	vec3 sunlightColor = sceneData.sunlightColor.xyz;

	vec3 toSun = -sunlightDir;

	float cosAngle = cos(dot(rayDir,toSun));
	float eccentricity=0.99;

//TODO fix this for the horizon
	float phase = max(HenyeyGreenstein(eccentricity, cosAngle), voxelInfo.silverIntensity*HenyeyGreenstein(cosAngle,0.99-voxelInfo.silverSpread)) ;
	
	float stepSize = 2.0;
	float stepMax = 256.0;
	float sunStepSize = 1.0;
	float sunStepMax = 6.0;

	vec3 backgroundColor = texture(backgroundTex, uv).xyz;

	float I =0.0; //illumination
	float sunI =0.0;
	float transmit = 1.0;
	float sunTransmit =1.0;

vec3 blueNoise = texture(blueNoiseTex, fract(uv)).rgb;
float jitterOffset = blueNoise.b * stepSize;
tMin = jitterOffset;
sunTMin = jitterOffset;

    while (tMin <= stepMax && I < 0.7 && transmit > 0.0)
    {
        tMin += stepSize;
        vec3 samplePos = rayOrigin + (rayDir * tMin);

        if (!insideBounds(samplePos, voxelGridMin, voxelGridMax)) continue;

        vec3 uvw = (samplePos - voxelGridMin) / (voxelGridMax - voxelGridMin);
        uvw = clamp(uvw, vec3(0.0), vec3(1.0));
        float density = texture(voxelBuffer, uvw).r * stepSize;

        if (density > 0.0)
        {
            const int NUM_SUN_SAMPLES = 6;
            float coneAngle = radians(10.0);
            float sunI = 0.0;

            for (int i = 0; i < NUM_SUN_SAMPLES; ++i)
            {
                vec2 rand = fract(blueNoise.xy + float(i));
                vec3 sunRay = sample_cone(toSun, coneAngle, rand);

                float sunT = 0.0;
                float sunTransmit = 1.0;
                float sampleSunI = 0.0;

                while (sunT < sunStepMax && sunTransmit > 0.01)
                {
                    sunT += sunStepSize;
                    vec3 sunSamplePos = samplePos + sunRay * sunT;

                    if (!insideBounds(sunSamplePos, voxelGridMin, voxelGridMax)) break;

                    vec3 sunUVW = (sunSamplePos - voxelGridMin) / (voxelGridMax - voxelGridMin);
                    sunUVW = clamp(sunUVW, vec3(0.0), vec3(1.0));
                    float sunDensity = texture(voxelBuffer, sunUVW).r * sunStepSize;

                    sunTransmit *= beer(sunDensity);
                    sampleSunI += sunTransmit * powder(sunDensity) * phase;
                }

                sunI += sampleSunI;
            }

            sunI /= float(NUM_SUN_SAMPLES); 
            I += transmit * phase * powder(density);
            I += max((sunI * 0.05), 0.001);

            transmit *= max((beer(density) + powder(density)), beer(density * 0.25) * 0.7) * (1.0 - voxelInfo.outScatterMultiplier);
        }
	}

	vec3 finalColor = (sunlightColor * I) + (backgroundColor * transmit) ;

	outFragColor =vec4(finalColor , 1.0);
}
