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
	return (1.0f - pow(g,2)) / (4.0f * 3.14159 * pow(1 + pow(g, 2) - 2.0f * g * angle, 1.5f));
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

    vec3 toCentre = voxelInfo.centrePos.xyz - rayOrigin;
    float depth = max(dot(toCentre, rayDir), 100);
    vec3 samplePoint = rayOrigin + rayDir * depth;

    vec4 prevClip = voxelInfo.prevViewProj * vec4(samplePoint, 1.0);
    vec2 prevUV = (prevClip.xy / prevClip.w) * 0.5 + 0.5;

    vec2 noiseUV = mod(gl_FragCoord.xy, 64.0) / 64.0;
    float noise = fract(dot(floor(ndc), vec2(0.06711056, 0.00583715)) * 52.9829189); 

    int reprojection = int(floor(noise.r * 2.0));
    bool valid = all(greaterThanEqual(prevUV, vec2(0.0))) && all(lessThanEqual(prevUV, vec2(1.0)));
    vec3 finalColor;

    if(voxelInfo.reprojection!=reprojection && valid)
    {
        outFragColor = texture(previousFrameTex, prevUV);
        return;
    }

	float tMin=0;

	vec3 voxelGridCentre = voxelInfo.centrePos.xyz;
	vec3 voxelDimension = voxelInfo.bounds.xyz;

	vec3 voxelGridMin = voxelGridCentre - voxelDimension*0.5;
	vec3 voxelGridMax = voxelGridCentre + voxelDimension*0.5;

	vec3 sunlightDir = sceneData.sunlightDirection.xyz;

	vec3 sunlightColor = sceneData.sunlightColor.xyz;

	vec3 toSun = normalize (sunlightDir);

	float cosAngle = dot(rayDir, toSun);
	float eccentricity=0.7;

	float phase = max(HenyeyGreenstein(cosAngle, eccentricity), voxelInfo.silverIntensity*HenyeyGreenstein(cos(cosAngle),0.90-voxelInfo.silverSpread));

	vec3 backgroundColor = texture(backgroundTex, uv).xyz;

	const float sunStepSize = 5.0;

	const float minStep = 0.5;
	const float maxStep = 5.0;
	const int maxSteps = 3000;

	float I = 0.0;
    float scatteredLight = 0.0;
	float transmit = 1.0;
	float t = 0.0;
	int steps = 0;

	int emptySamples = 0;
	const int maxEmptySamples = 2;
	bool fineMarch = false;    

    int hits =0;

    while (t<=maxSteps && I < 1.0 && transmit > 0.01 && steps < maxSteps && hits <20)
    {
        vec3 samplePos = rayOrigin + (rayDir * t);
        if (!insideBounds(samplePos, voxelGridMin, voxelGridMax)) 
        {
            t += maxStep; 
            continue;
        }
        float jitter = (noise * (random((gl_FragCoord.xy)- 0.5)));
        t+= jitter + minStep;
        vec3 uvw = (samplePos - voxelGridMin) / (voxelGridMax - voxelGridMin);
        uvw = clamp(uvw, vec3(0.0), vec3(1.0));
        float density = texture(voxelBuffer, uvw).r;

        if (density > 0.001)
        {
            if(fineMarch==false)
            {
                fineMarch = true;
                t-=maxStep;
                emptySamples = 0;
                continue;
            }
            hits++;
            float stepSize = minStep;
            float attenuatedDensity = density * stepSize;

            const int NUM_SUN_SAMPLES = 6;
            float coneAngle = radians(10.0);
            float sunI = 0.0;

            float sunOcclusion = 1.0;
            for (int i = 0; i < NUM_SUN_SAMPLES; ++i) 
            {
                vec2 rand =vec2(fract(noise + float(i)));
                vec3 sunRay = sample_cone(toSun, coneAngle, rand);
                float sunT = jitter + float(i) * sunStepSize;
                vec3 sunSamplePos = samplePos + sunRay * sunT;
                if (!insideBounds(sunSamplePos, voxelGridMin, voxelGridMax)) continue;

                vec3 sunUVW = (sunSamplePos - voxelGridMin) / (voxelGridMax - voxelGridMin);
                float sunDensity = texture(voxelBuffer, sunUVW).r;
                sunOcclusion *= exp(-sunDensity * sunStepSize);
            }
            sunI = sunOcclusion * 10.0;
            sunI /= float(NUM_SUN_SAMPLES);

            float multiScatterApprox = 1.0 / (1.0 + density * density * 0.5);
            I += transmit * phase * sunI * powder(density) * multiScatterApprox;
            scatteredLight += transmit * powder(density)  * phase * sunI;
            transmit *= max((beer(density) + powder(density)), beer(density * 0.25) * 0.7) * (1.0 - voxelInfo.outScatterMultiplier);
            
            t += stepSize;
        }
        else
        {
            emptySamples++;
            if (emptySamples >= maxEmptySamples)
            {
                fineMarch = false;
                t += maxStep;
            }
            else if (fineMarch)
            {
                t += minStep;
            }
            else
            {
                t += maxStep;
            }
        }

        steps++;
    }
    float ambientBoost = smoothstep(0.0, 0.5 , 1.0 - transmit); 
    I +=  0.4 * ambientBoost;

    vec3 godrayColor = sunlightColor * scatteredLight * I;
    finalColor = (godrayColor + sunlightColor * I )+ (backgroundColor * transmit);
    
    outFragColor =vec4(finalColor , 1.0);    
}
