// vulkan_guide.h : Include file for standard system include files,
// or project specific include files.
#pragma once

#include <memory>
#include <optional>
#include <string>
#include <vector>
#include <span>
#include <array>
#include <functional>
#include <deque>

#include <vulkan/vulkan.h>
#include <vulkan/vk_enum_string_helper.h>
#include <vk_mem_alloc.h>

#include <fmt/core.h>

#include <glm/mat4x4.hpp>
#include <glm/vec4.hpp>
#include <vk_descriptors.h>


#define VK_CHECK(x)                                                     \
	do {                                                                \
		VkResult err = x;                                               \
		if (err) {                                                      \
			fmt::println("Detected Vulkan error: {}", string_VkResult(err)); \
			abort();                                                    \
		}                                                               \
	} while (0)

//forward declares
struct MeshAsset;
struct GLTFMaterial;
class VulkanEngine;
struct DescriptorAllocatorGrowable;
struct DrawContext;

enum LightType
{
	PointLight,
	SpotLight
};

struct LightStruct
{
	glm::vec3 position;
	float cone;
	glm::vec3 color;
	float range;
	glm::vec3 direction;
	float intensity;
	float constant;
	float linear;
	float quadratic;
	LightType lightType;

	LightStruct()
		: position(glm::vec3(0)), lightType(LightType::PointLight), color(glm::vec3(0)), cone(0.0f),
		direction(glm::vec3(0)), range(0.0f), intensity(0.0f), constant(0.0f),
		linear(0.0f), quadratic(0.0f) {}
};

struct DelectionQueue
{
	std::deque<std::function<void()>> deletors;

	void push_function(std::function<void()>&& function)
	{
		deletors.push_back(function);
	}

	void flush()
	{
		for (auto it = deletors.rbegin(); it != deletors.rend(); it++)
		{
			(*it)();
		}

		deletors.clear();
	}
};

struct AllocatedImage
{
	VkImage image;
	VkImageView imageView;
	VmaAllocation allocation;
	VkExtent3D imageExtent;
	VkFormat imageFormat;
};

struct ComputePushConstants
{
	glm::vec4 data1;
	glm::vec4 data2;
	glm::vec4 data3;
	glm::vec4 data4;
};

struct ComputeEffect
{
	const char* name;

	VkPipeline pipeline;
	VkPipelineLayout layout;

	ComputePushConstants data;
};

struct AllocatedBuffer
{
	VkBuffer buffer;
	VmaAllocation allocation;
	VmaAllocationInfo info;
};

struct Vertex
{
	glm::vec3 position;
	float uv_x;
	glm::vec3 normal;
	float uv_y;
	glm::vec4 color;
};

struct GPUMeshBuffers
{
	AllocatedBuffer indexBuffer;
	AllocatedBuffer vertexBuffer;
	VkDeviceAddress vertexBufferAddress;
};

struct GPUDrawPushConstants
{
	glm::mat4 worldMatrix;
	VkDeviceAddress vertexBuffer;
};

struct FrameData
{
	VkCommandPool _commandPool;
	VkCommandBuffer _mainCommandBuffer;

	VkSemaphore _swapchainSemaphore, _renderSemaphore;
	VkFence _renderFence;

	DelectionQueue _deletionQueue;
	DescriptorAllocatorGrowable _frameDescriptors;
};

struct GPUSceneData
{
	glm::mat4 view;
	glm::mat4 proj;
	glm::mat4 viewproj;
	glm::vec4 ambientColor;
	glm::vec4 sunlightDirection;
	glm::vec4 sunlightColor;
};

struct LightBuffer
{
	LightStruct lights[10];
	int numLights;
};

enum class MaterialPass : uint8_t
{
	MainColor,
	Transparent,
	Volumetric,
	Billboard,
	Other
};

struct MaterialPipeline
{
	VkPipeline pipeline;
	VkPipelineLayout layout;
};

struct MaterialInstance
{
	MaterialPipeline* pipeline;
	VkDescriptorSet materialSet;
	MaterialPass passType;
};

class IRenderable
{

	virtual void Draw(const glm::mat4& topMatrix, DrawContext& ctx) = 0;
};

struct Node : public IRenderable
{

	std::weak_ptr<Node> parent;
	std::vector<std::shared_ptr<Node>> children;

	glm::mat4 localTransform;
	glm::mat4 worldTransform;

	void refreshTransform(const glm::mat4& parentMatrix)
	{
		worldTransform = parentMatrix * localTransform;
		for (auto c : children) {
			c->refreshTransform(worldTransform);
		}
	}

	virtual void Draw(const glm::mat4& topMatrix, DrawContext& ctx)
	{
		// draw children
		for (auto& c : children) {
			c->Draw(topMatrix, ctx);
		}
	}
};


struct MeshNode : public Node {

	std::shared_ptr<MeshAsset> mesh;

	virtual void Draw(const glm::mat4& topMatrix, DrawContext& ctx) override;
};


struct VoxelGrid
{
	float* density;
	size_t dimenstion = 128;
	glm::vec3 bounds[2] = { glm::vec3(-30,-30,-30), glm::vec3(30,30,30) }; //world space bounds
};