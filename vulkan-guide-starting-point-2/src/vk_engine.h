// vulkan_guide.h : Include file for standard system include files,
// or project specific include files.

#pragma once

#include <vk_types.h>
#include <vk_descriptors.h>
#include <vk_pipelines.h>
#include <vk_loader.h>
#include <camera.h>


struct GLTFMetallic_Roughness
{
	MaterialPipeline opaquePipeline;
	MaterialPipeline transparentPipeline;

	VkDescriptorSetLayout materialLayout;

	struct MaterialConstants
	{
		glm::vec4 colorFactors;
		glm::vec4 metalRoughFactors;
		glm::vec4 extra[14];
	};
	
	struct MaterialResources
	{
		AllocatedImage colorImage;
		VkSampler colorSampler;
		AllocatedImage metalRoughImage;
		VkSampler metalRoughSampler;
		VkBuffer dataBuffer;
		uint32_t dataBufferOffset;
	};

	DescriptorWriter writer;

	void build_pipelines(VulkanEngine* engine);
	void clear_resources(VkDevice device);

	MaterialInstance write_material(VkDevice device, MaterialPass pass, const MaterialResources& resources, DescriptorAllocatorGrowable descriptorAllocator);
};

struct RenderObject
{
	uint32_t indexCount;
	uint32_t firstIndex;
	VkBuffer indexBuffer;

	MaterialInstance* material;
	Bounds bounds;
	glm::mat4 transform;
	VkDeviceAddress vertexBufferAddress;
};

struct DrawContext
{

	std::vector<RenderObject> OpaqueSurfaces;
	std::vector<RenderObject> TransparentSurfaces;
};

struct EngineStats
{
	float frametime;
	int triangleCount;
	int drawcallCount;
	float sceneUpdateTime;
	float meshDrawTime;
};

constexpr unsigned int FRAME_OVERLAP = 2;

class VulkanEngine {
public:
	DelectionQueue _mainDeletionQueue;

	bool _isInitialized{ false };
	int _frameNumber {0};
	bool stop_rendering{ false };
	VkExtent2D _windowExtent{ 1700 , 900 };

	struct SDL_Window* _window{ nullptr };
	bool resize_requested;

	static VulkanEngine& Get();

	DrawContext mainDrawContext;
	std::unordered_map<std::string, std::shared_ptr<Node>> loadedNodes;
	std::unordered_map<std::string, std::shared_ptr<LoadedGLTF>> loadedScenes;

	Camera mainCamera;

	EngineStats stats;

	//initialisation members
	VkInstance _instance;
	VkDebugUtilsMessengerEXT _debug_messenger;
	VkPhysicalDevice _chosenGPU;
	VkDevice _device;
	VkSurfaceKHR _surface;

	//swapchain members
	VkSwapchainKHR _swapchain;
	VkFormat _swapchainImageFormat;
	std::vector<VkImage> _swapchainImages;
	std::vector<VkImageView> _swapchainImageViews;
	VkExtent2D _swapchainExtent;

	//draw resources
	AllocatedImage _drawImage;
	AllocatedImage _depthImage;

	VkExtent2D _drawExtent;
	float renderScale = 1.f;

	//descriptor members
	DescriptorAllocator globalDescriptorAllocator;
	DescriptorAllocatorGrowable globalDescriptorAllocatorGrowable;

	VkDescriptorSet _drawImageDescriptors;
	VkDescriptorSetLayout _drawImageDescriptorLayout;

	GPUSceneData sceneData;
	VkDescriptorSetLayout _gpuSceneDataDescriptorLayout;
	LightBuffer lightData;
	std::vector<LightStruct> sceneLights;
	VkDescriptorSetLayout _gpuLightDataDescriptorLayout;

	//temporary textures
	AllocatedImage _whiteImage;
	AllocatedImage _blackImage;
	AllocatedImage _greyImage;
	AllocatedImage _errorCheckImage;

	VkDescriptorSetLayout _singleImageDescriptorLayout;

	GLTFMetallic_Roughness metalRoughMaterial;

	//image samplers
	VkSampler _defaultSamplerLinear;
	VkSampler _defaultSamplerNearest;

	//frame members
	FrameData _frames[FRAME_OVERLAP];

	//immediate submit members
	VkFence _immFence;
	VkCommandBuffer _immCommandBuffer;
	VkCommandPool _immCommandPool;

	//graphics members
	VkQueue _graphicsQueue;
	uint32_t _graphicsQueueFamily;

	//compute pipeline members
	VkPipeline _gradientPipeline;
	VkPipelineLayout _gradientPipelineLayout;

	//main graphics pipeline members
	VkPipelineLayout _meshPipelineLayout;
	VkPipeline _meshPipeline;

	GPUMeshBuffers rectangle;

	std::vector <std::shared_ptr<MeshAsset>> testMeshes;

	//triangle pipeline members
	VkPipelineLayout _trianglePipelineLayout;
	VkPipeline _trianglePipeline;

	//background effect members
	std::vector<ComputeEffect> backgroundEffects;
	int currentBackgroundEffect{ 0 };

	//memory allocator
	VmaAllocator _allocator;

	//initializes everything in the engine
	void init();

	//shuts down the engine
	void cleanup();

	//draw loop
	void draw();

	void draw_background(VkCommandBuffer cmd);

	void draw_geometry(VkCommandBuffer cmd);

	void draw_imgui(VkCommandBuffer cmd, VkImageView targetImageView);

	void immediate_submit(std::function<void(VkCommandBuffer cmd)>&& function);

	void update_scene();

	//run main loop
	void run();

	//frame member functions
	FrameData& get_current_frame() { return _frames[_frameNumber % FRAME_OVERLAP]; };

	GPUMeshBuffers upload_mesh(std::span<uint32_t> indices, std::span<Vertex> vertices);

	AllocatedImage create_image(VkExtent3D size, VkFormat format, VkImageUsageFlags usage, bool mipmapped = false);
	AllocatedImage create_image(void* data, VkExtent3D size, VkFormat format, VkImageUsageFlags usage, bool mipmapped = false);
	void destroy_image(const AllocatedImage& img);

	AllocatedBuffer create_buffer(size_t allocSize, VkBufferUsageFlags usage, VmaMemoryUsage memoryUsage);
	void destroy_buffer(const AllocatedBuffer& buffer);

private:
	void init_vulkan();
	void init_swapchain();
	void init_commands();
	void init_sync_structures();
	void init_descriptors();
	void init_pipelines();
	void init_background_pipelines();
	void init_mesh_pipeline();
	void init_default_data();
	void init_imgui();

	void create_swapchain(uint32_t width, uint32_t height);
	void destroy_swapchain();
	void resize_swapchain();


};
