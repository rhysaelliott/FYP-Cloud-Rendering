//> includes
#include "vk_engine.h"

#include <SDL.h>
#include <SDL_vulkan.h>
#include <glm/gtx/transform.hpp>
#include "stb_image.h"

#include "imgui.h"
#include "imgui_impl_sdl2.h"
#include "imgui_impl_vulkan.h"

#include <vk_initializers.h>
#include <vk_types.h>
#include <vk_images.h>
#include<map>

#define VMA_IMPLEMENTATION
#include "vk_mem_alloc.h"

//bootstrap library
#include "VkBootstrap.h"

#include <chrono>
#include <thread>

VulkanEngine* loadedEngine = nullptr;

constexpr bool bUseValidationLayers = true;

bool is_visible(const RenderObject& obj, const glm::mat4& viewproj)
{
    std::array<glm::vec3, 8> corners{
        glm::vec3{1,1,1},
        glm::vec3{1,1,-1},
        glm::vec3{1,-1,1},
        glm::vec3{1,-1,-1},
        glm::vec3{-1,1,1},
        glm::vec3{-1,1,-1},
        glm::vec3{-1,-1,1},
        glm::vec3{-1,-1,-1},
    };

    glm::mat4 matrix = viewproj * obj.transform;

    glm::vec3 min = { 1.5, 1.5,1.5 };
    glm::vec3 max = { -1.5, -1.5,-1.5 };

    for (int c = 0; c < 8; c++)
    {
        //project into clip space
        glm::vec4 v = matrix * glm::vec4(obj.bounds.origin + (corners[c] * obj.bounds.extents), 1.f);

        //correct perspective
        float invW = 1.0f / v.w;
        v.x *= invW;
        v.y *= invW;
        v.z *= invW;

        min = glm::min(glm::vec3{ v.x,v.y,v.z }, min);
        max = glm::max(glm::vec3{ v.x,v.y,v.z }, max);
    }
    //check collision
    if (min.z > 1.f || max.z < 0.f || min.x>1.f || max.x < -1.f || min.y>1.f || max.y < -1.f)
    {
        return false;
    }
    else
    {
        return true;
    }
}
bool is_light_affecting_object(const LightStruct& light, const RenderObject& obj)
{
    //todo expand on this
    float distance = glm::length(light.position - obj.bounds.origin);
    float boundsExtent = glm::length(obj.bounds.extents);
    if (distance > light.range + boundsExtent)
    {
        return false;
    }
    
    return true;
}

VulkanEngine& VulkanEngine::Get() { return *loadedEngine; }
void VulkanEngine::init()
{
    // only one engine initialization is allowed with the application.
    assert(loadedEngine == nullptr);
    loadedEngine = this;

    // We initialize SDL and create a window with it.
    SDL_Init(SDL_INIT_VIDEO);

    SDL_WindowFlags window_flags = (SDL_WindowFlags)(SDL_WINDOW_VULKAN | SDL_WINDOW_RESIZABLE);

    _window = SDL_CreateWindow(
        "Vulkan Engine",
        SDL_WINDOWPOS_UNDEFINED,
        SDL_WINDOWPOS_UNDEFINED,
        _windowExtent.width,
        _windowExtent.height,
        window_flags);

    SDL_SetRelativeMouseMode(SDL_TRUE);
    
    init_vulkan();

    init_swapchain();

    init_commands();

    init_sync_structures();

    init_descriptors();
    
    init_pipelines();

    init_default_data();

    init_volumetric_data();

    init_billboard_data();

    init_imgui();

    std::string structurePath = { "..\\..\\assets\\basicmesh.glb" };
    auto structureFile = loadGltf(this, structurePath);

    assert(structureFile.has_value());

    loadedScenes["structure"] = *structureFile;

    // everything went fine
    _isInitialized = true;
}

void VulkanEngine::init_vulkan()
{
    vkb::InstanceBuilder builder;

    //vulkan instance with basic debugging
    auto inst_ret = builder.set_app_name("Vulkan PBR")
        .request_validation_layers(bUseValidationLayers)
        .use_default_debug_messenger()
        .require_api_version(1, 3, 0)
        .build();

    vkb::Instance vkb_inst = inst_ret.value();

    _instance = vkb_inst.instance;
    _debug_messenger = vkb_inst.debug_messenger;

    SDL_Vulkan_CreateSurface(_window, _instance, &_surface);

    //vulkan 1.3 features
    VkPhysicalDeviceVulkan13Features features{};
    features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES;
    features.dynamicRendering = true;
    features.synchronization2 = true;

    //vulkan 1.2 features
    VkPhysicalDeviceVulkan12Features features12{};
    features12.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES;
    features12.bufferDeviceAddress = true;
    features12.descriptorIndexing = true;

    vkb::PhysicalDeviceSelector selector{ vkb_inst };
    vkb::PhysicalDevice physicalDevice = selector
        .set_minimum_version(1, 3)
        .set_required_features_13(features)
        .set_required_features_12(features12)
        .set_surface(_surface)
        .select()
        .value();

    vkb::DeviceBuilder deviceBuilder{ physicalDevice };

    vkb::Device vkbDevice = deviceBuilder.build().value();

    _device = vkbDevice.device;
    _chosenGPU = physicalDevice.physical_device;

    //get graphics queue
    _graphicsQueue = vkbDevice.get_queue(vkb::QueueType::graphics).value();
    _graphicsQueueFamily = vkbDevice.get_queue_index(vkb::QueueType::graphics).value();

    //init memory allocator
    VmaAllocatorCreateInfo allocatorInfo = {};
    allocatorInfo.physicalDevice = _chosenGPU;
    allocatorInfo.device = _device;
    allocatorInfo.instance = _instance;
    allocatorInfo.flags = VMA_ALLOCATOR_CREATE_BUFFER_DEVICE_ADDRESS_BIT;
    vmaCreateAllocator(&allocatorInfo, &_allocator);

    _mainDeletionQueue.push_function([&]()
        {
            if (_allocator != VK_NULL_HANDLE)
            {
                vmaDestroyAllocator(_allocator);
            }
        });
}

void VulkanEngine::init_swapchain()
{
    create_swapchain(_windowExtent.width, _windowExtent.height);

    VkExtent3D drawImageExtent =
    {
        _windowExtent.width,
        _windowExtent.height,
        1
    };

    _drawImage.imageFormat = VK_FORMAT_R16G16B16A16_SFLOAT;
    _drawImage.imageExtent = drawImageExtent;

    VkImageUsageFlags drawImageUsages{};
    drawImageUsages |= VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
    drawImageUsages |= VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    drawImageUsages |= VK_IMAGE_USAGE_STORAGE_BIT;
    drawImageUsages |= VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;

    VkImageCreateInfo rimg_info =
        vkinit::image_create_info(_drawImage.imageFormat, drawImageUsages, drawImageExtent);

    //allocate gpu local memory for the draw image
    VmaAllocationCreateInfo rimg_allocInfo = {};
    rimg_allocInfo.usage = VMA_MEMORY_USAGE_GPU_ONLY;
    rimg_allocInfo.requiredFlags = VkMemoryPropertyFlags(VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

    //allocate and create image
    vmaCreateImage(_allocator, &rimg_info, &rimg_allocInfo, &_drawImage.image, &_drawImage.allocation, nullptr);

    //build image view of image
    VkImageViewCreateInfo rview_info =
        vkinit::imageview_create_info(_drawImage.imageFormat, _drawImage.image, VK_IMAGE_ASPECT_COLOR_BIT);

    VK_CHECK(vkCreateImageView(_device, &rview_info, nullptr, &_drawImage.imageView));

    //create depth image
    _depthImage.imageFormat = VK_FORMAT_D32_SFLOAT;
    _depthImage.imageExtent = drawImageExtent;
    VkImageUsageFlags depthImageUsage{  };
    depthImageUsage |= VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT;

    VkImageCreateInfo dimgInfo =
        vkinit::image_create_info(_depthImage.imageFormat, depthImageUsage, drawImageExtent);

    vmaCreateImage(_allocator, &dimgInfo, &rimg_allocInfo, &_depthImage.image, &_depthImage.allocation, nullptr);

    VkImageViewCreateInfo dviewInfo =
        vkinit::imageview_create_info(_depthImage.imageFormat, _depthImage.image, VK_IMAGE_ASPECT_DEPTH_BIT);

    VK_CHECK(vkCreateImageView(_device, &dviewInfo, nullptr, &_depthImage.imageView));

    _mainDeletionQueue.push_function([=]()
        {
            vkDestroyImageView(_device, _drawImage.imageView, nullptr);
            vmaDestroyImage(_allocator, _drawImage.image, _drawImage.allocation);

            vkDestroyImageView(_device, _depthImage.imageView, nullptr);
            vmaDestroyImage(_allocator, _depthImage.image, _depthImage.allocation);
        });
}

void VulkanEngine::init_commands()
{
    //create command pool for commands for graphics queue
    //create pool for reseting command buffers
    VkCommandPoolCreateInfo commandPoolInfo =
        vkinit::command_pool_create_info(_graphicsQueueFamily, VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT);


    for (int i = 0; i < FRAME_OVERLAP; i++)
    {
        VK_CHECK(vkCreateCommandPool(_device, &commandPoolInfo, nullptr, &_frames[i]._commandPool));

        //allocate default rendering buffer
        VkCommandBufferAllocateInfo cmdAllocInfo =
            vkinit::command_buffer_allocate_info(_frames[i]._commandPool, 1);

        VK_CHECK(vkAllocateCommandBuffers(_device, &cmdAllocInfo, &_frames[i]._mainCommandBuffer));
    }

    VK_CHECK(vkCreateCommandPool(_device, &commandPoolInfo, nullptr, &_immCommandPool));

    VkCommandBufferAllocateInfo cmdAllocInfo =
        vkinit::command_buffer_allocate_info(_immCommandPool, 1);

    VK_CHECK(vkAllocateCommandBuffers(_device, &cmdAllocInfo, &_immCommandBuffer));

    _mainDeletionQueue.push_function([=]()
        {
            vkDestroyCommandPool(_device, _immCommandPool, nullptr);
        });
}

void VulkanEngine::init_descriptors()
{
    //create descriptor pool that holds 10 sets with 1 image each
    std::vector<DescriptorAllocator::PoolSizeRatio> sizes =
    {
        {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,1}
    };

    globalDescriptorAllocator.init_pool(_device, 10, sizes);

    {
        //descriptor set layout for compute draw
        DescriptorLayoutBuilder builder;
        builder.add_binding(0, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE);
        _drawImageDescriptorLayout = builder.build(_device, VK_SHADER_STAGE_COMPUTE_BIT);

        //allocate descriptor set for draw image 
        _drawImageDescriptors = globalDescriptorAllocator.allocate(_device, _drawImageDescriptorLayout);

        DescriptorWriter writer;
        writer.write_image(0, _drawImage.imageView, VK_NULL_HANDLE, VK_IMAGE_LAYOUT_GENERAL, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE);

        writer.update_set(_device, _drawImageDescriptors);
    }





    //todo init the billboard set descriptor here 


    for (int i = 0; i < FRAME_OVERLAP; i++)
    {
        //create descriptor pool
        std::vector<DescriptorAllocatorGrowable::PoolSizeRatio> frameSizes =
        {
            {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,3},
            {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,3},
            {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,3},
            {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,4},
        };

        _frames[i]._frameDescriptors = DescriptorAllocatorGrowable{};
        _frames[i]._frameDescriptors.init(_device, 1000, frameSizes);

        _mainDeletionQueue.push_function([&, i]()
            {
                _frames[i]._frameDescriptors.destroy_pools(_device);
            });
    }

    {
        DescriptorLayoutBuilder builder;
        builder.add_binding(0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER);
        _gpuSceneDataDescriptorLayout = builder.build(_device, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT);
    }

    {
        DescriptorLayoutBuilder builder;
        builder.add_binding(0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER);
        _gpuLightDataDescriptorLayout = builder.build(_device, VK_SHADER_STAGE_FRAGMENT_BIT);
    }

    {
        DescriptorLayoutBuilder builder;
        builder.add_binding(0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER);
        _billboardPositionsDescriptorLayout = builder.build(_device, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT);
    }
 


    {
        DescriptorLayoutBuilder builder;
        builder.add_binding(0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER);
        _singleImageDescriptorLayout = builder.build(_device,VK_SHADER_STAGE_FRAGMENT_BIT);
    }

    _mainDeletionQueue.push_function([&]()
        {
            globalDescriptorAllocator.destroy_pool(_device);

            vkDestroyDescriptorSetLayout(_device, _drawImageDescriptorLayout, nullptr);
            vkDestroyDescriptorSetLayout(_device, _singleImageDescriptorLayout, nullptr);
            vkDestroyDescriptorSetLayout(_device, _gpuLightDataDescriptorLayout, nullptr);
            vkDestroyDescriptorSetLayout(_device, _gpuSceneDataDescriptorLayout, nullptr);
            vkDestroyDescriptorSetLayout(_device, _billboardPositionsDescriptorLayout, nullptr);
        });
}

void VulkanEngine::init_pipelines()
{
    //compute pipelines
    init_background_pipelines();

    //graphics pipelines
    init_mesh_pipeline();

    init_volumetric_pipeline();

    init_billboard_pipeline();

    metalRoughMaterial.build_pipelines(this);

}

void VulkanEngine::init_background_pipelines()
{
    VkPushConstantRange pushConstant{};
    pushConstant.offset = 0;
    pushConstant.size = sizeof(ComputePushConstants);
    pushConstant.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

    VkPipelineLayoutCreateInfo computeLayout = {};
    computeLayout.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    computeLayout.pNext = nullptr;
    computeLayout.pSetLayouts = &_drawImageDescriptorLayout;
    computeLayout.setLayoutCount = 1;

    computeLayout.pPushConstantRanges = &pushConstant;
    computeLayout.pushConstantRangeCount = 1;

    VK_CHECK(vkCreatePipelineLayout(_device, &computeLayout, nullptr, &_gradientPipelineLayout));

    VkShaderModule gradientShader;
    if (!vkutil::load_shader_module("../../shaders/gradient_color.comp.spv", _device, &gradientShader))
    {
        fmt::print("Error when building the compute shader \n");
    }

    VkShaderModule skyShader;
    if (!vkutil::load_shader_module("../../shaders/sky.comp.spv", _device, &skyShader))
    {
        fmt::print("Error when building the compute shader \n");
    }

    VkPipelineShaderStageCreateInfo stageInfo = {};
    stageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stageInfo.pNext = nullptr;
    stageInfo.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    stageInfo.module = gradientShader;
    stageInfo.pName = "main";

    VkComputePipelineCreateInfo computePipelineCreateInfo{};
    computePipelineCreateInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    computePipelineCreateInfo.pNext = nullptr;
    computePipelineCreateInfo.layout = _gradientPipelineLayout;
    computePipelineCreateInfo.stage = stageInfo;

    ComputeEffect gradient;
    gradient.layout = _gradientPipelineLayout;
    gradient.name = "gradient";
    gradient.data = {};

    gradient.data.data1 = glm::vec4(0.53f, 0.81f, 0.98f, 1.0f);
    gradient.data.data2 = glm::vec4(0.98f, 0.87f, 0.67f, 1.0f);

    VK_CHECK(vkCreateComputePipelines(_device, VK_NULL_HANDLE, 1, &computePipelineCreateInfo, nullptr, &gradient.pipeline));

    computePipelineCreateInfo.stage.module = skyShader;

    ComputeEffect sky;
    sky.layout = _gradientPipelineLayout;
    sky.name = "sky";
    sky.data = {};

    sky.data.data1 = glm::vec4(0.1, 0.2, 0.4, 0.97);

    VK_CHECK(vkCreateComputePipelines(_device, VK_NULL_HANDLE, 1, &computePipelineCreateInfo, nullptr, &sky.pipeline));

    backgroundEffects.push_back(gradient);
    backgroundEffects.push_back(sky);

    vkDestroyShaderModule(_device, gradientShader, nullptr);
    vkDestroyShaderModule(_device, skyShader, nullptr);

    _mainDeletionQueue.push_function([=]()
        {
            vkDestroyPipelineLayout(_device, _gradientPipelineLayout, nullptr);
            vkDestroyPipeline(_device, sky.pipeline, nullptr);
            vkDestroyPipeline(_device, gradient.pipeline, nullptr);
        });
}

void VulkanEngine::init_mesh_pipeline()
{
    VkShaderModule meshFragShader;
    if (!vkutil::load_shader_module("../../shaders/tex_image.frag.spv", _device, &meshFragShader))
    {
        fmt::print("Error when building the mesh fragment shader module \n");
    }
    else
    {
        fmt::print("Mesh fragment shader successfully loaded \n");
    }

    VkShaderModule meshVertexShader;
    if (!vkutil::load_shader_module("../../shaders/colored_triangle_mesh.vert.spv", _device, &meshVertexShader))
    {
        fmt::print("Error when building the mesh vertex shader module \n");
    }
    else
    {
        fmt::print("Mesh vertex shader successfully loaded \n");
    }

    VkPushConstantRange bufferRange{};
    bufferRange.offset = 0;
    bufferRange.size = sizeof(GPUDrawPushConstants);
    bufferRange.stageFlags = VK_SHADER_STAGE_VERTEX_BIT;


    VkPipelineLayoutCreateInfo pipeline_layout_info = vkinit::pipeline_layout_create_info();
    pipeline_layout_info.pPushConstantRanges = &bufferRange;
    pipeline_layout_info.pushConstantRangeCount = 1;
    pipeline_layout_info.pSetLayouts = &_singleImageDescriptorLayout;
    pipeline_layout_info.setLayoutCount = 1;

    VK_CHECK(vkCreatePipelineLayout(_device, &pipeline_layout_info, nullptr, &_meshPipelineLayout));

    PipelineBuilder pipelineBuilder;

    pipelineBuilder._pipelineLayout = _meshPipelineLayout;
    pipelineBuilder.set_shaders(meshVertexShader, meshFragShader);
    pipelineBuilder.set_input_topology(VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST);
    pipelineBuilder.set_polygon_mode(VK_POLYGON_MODE_FILL);
    pipelineBuilder.set_cull_mode(VK_CULL_MODE_NONE, VK_FRONT_FACE_CLOCKWISE);
    pipelineBuilder.set_multisampling_none();
    //pipelineBuilder.disable_blending();
    pipelineBuilder.enable_blending_alphablend();
    //pipelineBuilder.disable_depthtest();
    pipelineBuilder.enable_depthtest(true, VK_COMPARE_OP_GREATER_OR_EQUAL);

    pipelineBuilder.set_color_attachment_format(_drawImage.imageFormat);
    pipelineBuilder.set_depth_format(_depthImage.imageFormat);


    _meshPipeline = pipelineBuilder.build_pipeline(_device);

    vkDestroyShaderModule(_device, meshVertexShader, nullptr);
    vkDestroyShaderModule(_device, meshFragShader, nullptr);

    _mainDeletionQueue.push_function([&]()
        {
            vkDestroyPipelineLayout(_device, _meshPipelineLayout, nullptr);
            vkDestroyPipeline(_device, _meshPipeline, nullptr);
        });
}

void VulkanEngine::init_volumetric_pipeline()
{
    
    VkShaderModule volumetricFragShader;
    if (!vkutil::load_shader_module("../../shaders/cloud.frag.spv", _device, &volumetricFragShader))
    {
        fmt::print("Error when building the volumetric fragment shader module \n");
    }
    else
    {
        fmt::print("Volumetric fragment shader successfully loaded \n");
    }

    VkShaderModule volumetricVertexShader;
    if (!vkutil::load_shader_module("../../shaders/cloud.vert.spv", _device, &volumetricVertexShader))
    {
        fmt::print("Error when building the volumetric vertex shader module \n");
    }
    else
    {
        fmt::print("Volumetric vertex shader successfully loaded \n");
    }

    VkPushConstantRange bufferRange{};
    bufferRange.offset = 0;
    bufferRange.size = sizeof(GPUDrawPushConstants);
    bufferRange.stageFlags = VK_SHADER_STAGE_VERTEX_BIT;

    DescriptorLayoutBuilder layoutBuilder;
    layoutBuilder.add_binding(0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER);

    _volumetricDescriptorLayout = layoutBuilder.build(_device,  VK_SHADER_STAGE_FRAGMENT_BIT);

    VkDescriptorSetLayout layouts[] = { _gpuSceneDataDescriptorLayout, _volumetricDescriptorLayout };

    VkPipelineLayoutCreateInfo pipeline_layout_info = vkinit::pipeline_layout_create_info();
    pipeline_layout_info.pPushConstantRanges = &bufferRange;
    pipeline_layout_info.pushConstantRangeCount = 1;
    pipeline_layout_info.pSetLayouts = layouts;
    pipeline_layout_info.setLayoutCount = 2;

    VkPipelineLayout pipelineLayout;

    VK_CHECK(vkCreatePipelineLayout(_device, &pipeline_layout_info, nullptr, &pipelineLayout));
    _volumetricPipeline.layout = pipelineLayout;

    PipelineBuilder pipelineBuilder;

    pipelineBuilder._pipelineLayout = pipelineLayout;
    pipelineBuilder.set_shaders(volumetricVertexShader, volumetricFragShader);
    pipelineBuilder.set_input_topology(VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST);
    pipelineBuilder.set_polygon_mode(VK_POLYGON_MODE_FILL);
    pipelineBuilder.set_cull_mode(VK_CULL_MODE_NONE, VK_FRONT_FACE_CLOCKWISE);
    pipelineBuilder.set_multisampling_none();
    pipelineBuilder.enable_blending_alphablend();
    pipelineBuilder.enable_depthtest(true, VK_COMPARE_OP_GREATER_OR_EQUAL);

    pipelineBuilder.set_color_attachment_format(_drawImage.imageFormat);
    pipelineBuilder.set_depth_format(_depthImage.imageFormat);

    _volumetricPipeline.pipeline = pipelineBuilder.build_pipeline(_device);

    vkDestroyShaderModule(_device, volumetricVertexShader, nullptr);
    vkDestroyShaderModule(_device, volumetricFragShader, nullptr);

    _mainDeletionQueue.push_function([&]()
        {
            vkDestroyDescriptorSetLayout(_device, _volumetricDescriptorLayout, nullptr);
            vkDestroyPipelineLayout(_device, _volumetricPipeline.layout, nullptr);
            vkDestroyPipeline(_device, _volumetricPipeline.pipeline, nullptr);
        });
}

void VulkanEngine::init_billboard_pipeline()
{
   
    VkShaderModule billboardFragShader;
    if (!vkutil::load_shader_module("../../shaders/billboard.frag.spv", _device, &billboardFragShader))
    {
        fmt::print("Error when building the billboard fragment shader module \n");
    }
    else
    {
        fmt::print("Billboard fragment shader successfully loaded \n");
    }

    VkShaderModule billboardDitherFragShader;
    if (!vkutil::load_shader_module("../../shaders/billboardDither.frag.spv", _device, &billboardDitherFragShader))
    {
        fmt::print("Error when building the billboard dither fragment shader module \n");
    }
    else
    {
        fmt::print("Billboard dither fragment shader successfully loaded \n");
    }

    VkShaderModule billboardVertexShader;
    if (!vkutil::load_shader_module("../../shaders/billboard.vert.spv", _device, &billboardVertexShader))
    {
        fmt::print("Error when building the billboard vertex shader module \n");
    }
    else
    {
        fmt::print("Billboard vertex shader successfully loaded \n");
    }

    VkPushConstantRange bufferRange{};
    bufferRange.offset = 0;
    bufferRange.size = sizeof(GPUDrawPushConstants);
    bufferRange.stageFlags = VK_SHADER_STAGE_VERTEX_BIT ;

    DescriptorLayoutBuilder layoutBuilder;
    layoutBuilder.add_binding(1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER); //cloud tex1
    layoutBuilder.add_binding(2, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER); //cloud tex2
    layoutBuilder.add_binding(3, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER); //cloud tex3
    layoutBuilder.add_binding(4, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER); //cloud tex4
    layoutBuilder.add_binding(5, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER); //cloud tex5
    layoutBuilder.add_binding(6, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER); //cloud tex6
    layoutBuilder.add_binding(7, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER); //cloud tex7
    layoutBuilder.add_binding(8, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER); //cloud tex8
    layoutBuilder.add_binding(9, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER); //cloud tex9

    _billboardMaterialDescriptorLayout = layoutBuilder.build(_device,  VK_SHADER_STAGE_FRAGMENT_BIT);

    VkDescriptorSetLayout layouts[] = { _gpuSceneDataDescriptorLayout, _billboardMaterialDescriptorLayout, _billboardPositionsDescriptorLayout};

    //todo change this stuff
    VkPipelineLayoutCreateInfo pipeline_layout_info = vkinit::pipeline_layout_create_info();
    pipeline_layout_info.pPushConstantRanges = &bufferRange;
    pipeline_layout_info.pushConstantRangeCount = 1;
    pipeline_layout_info.pSetLayouts = layouts;
    pipeline_layout_info.setLayoutCount = 3;
    

    VkPipelineLayout pipelineLayout;

    VK_CHECK(vkCreatePipelineLayout(_device, &pipeline_layout_info, nullptr, &pipelineLayout));
    _billboardPipeline[0].layout = pipelineLayout;
    _billboardPipeline[1].layout = pipelineLayout;

    PipelineBuilder pipelineBuilder;

    pipelineBuilder._pipelineLayout = pipelineLayout;
    pipelineBuilder.set_shaders(billboardVertexShader, billboardFragShader);
    pipelineBuilder.set_input_topology(VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST);
    pipelineBuilder.set_polygon_mode(VK_POLYGON_MODE_FILL);
    pipelineBuilder.set_cull_mode(VK_CULL_MODE_NONE, VK_FRONT_FACE_CLOCKWISE);
    pipelineBuilder.set_multisampling_none();
    pipelineBuilder.enable_blending_alphablend();

    pipelineBuilder.enable_depthtest(false, VK_COMPARE_OP_GREATER_OR_EQUAL);

    pipelineBuilder.set_color_attachment_format(_drawImage.imageFormat);
    pipelineBuilder.set_depth_format(_depthImage.imageFormat);

    _billboardPipeline[0].pipeline = pipelineBuilder.build_pipeline(_device);

    pipelineBuilder.set_shaders(billboardVertexShader, billboardDitherFragShader);
    
    _billboardPipeline[1].pipeline = pipelineBuilder.build_pipeline(_device);


    vkDestroyShaderModule(_device, billboardVertexShader, nullptr);
    vkDestroyShaderModule(_device, billboardFragShader, nullptr);
    vkDestroyShaderModule(_device, billboardDitherFragShader, nullptr);

    _mainDeletionQueue.push_function([&]()
        {
            vkDestroyDescriptorSetLayout(_device, _billboardMaterialDescriptorLayout, nullptr);
            vkDestroyPipelineLayout(_device, _billboardPipeline[0].layout, nullptr);
            vkDestroyPipeline(_device, _billboardPipeline[0].pipeline, nullptr);
            vkDestroyPipeline(_device, _billboardPipeline[1].pipeline, nullptr);
        });
}

void VulkanEngine::init_default_data()
{

    //load default textures
    uint32_t white = glm::packUnorm4x8(glm::vec4(1, 1, 1, 1));
    _whiteImage = create_image((void*)&white, VkExtent3D{ 1,1,1 }, VK_FORMAT_R8G8B8A8_UNORM, VK_IMAGE_USAGE_SAMPLED_BIT);

    uint32_t grey = glm::packUnorm4x8(glm::vec4(.66f, .66f, .66f, 1));
    _greyImage = create_image((void*)&grey, VkExtent3D{ 1,1,1 }, VK_FORMAT_R8G8B8A8_UNORM, VK_IMAGE_USAGE_SAMPLED_BIT);

    uint32_t black = glm::packUnorm4x8(glm::vec4(0, 0, 0, 1));
    uint32_t magenta = glm::packUnorm4x8(glm::vec4(1, 0, 1, 1));
    std::array<uint32_t, 16 * 16> pixels;
    for (int x = 0; x < 16; x++)
    {
        for (int y = 0; y < 16; y++)
        {
            pixels[y * 16 + x] = ((x % 2) ^ (y % 2)) ? magenta : black;
        }
    }
    _errorCheckImage = create_image(pixels.data(), VkExtent3D{ 16,16,1 }, VK_FORMAT_R8G8B8A8_UNORM, VK_IMAGE_USAGE_SAMPLED_BIT);


    _cloudSamplers.resize(9, VK_NULL_HANDLE);
    for (int i = 0; i < 9; i++)
    {
        //load textures
        const std::string path("..\\..\\assets\\cloud"+ std::to_string(i+1) +".png");
        int width, height, nrChannels;

        unsigned char* data = stbi_load(path.c_str(), &width, &height, &nrChannels, 4);
        if (data)
        {
            VkExtent3D imageSize;
            imageSize.width = width;
            imageSize.height = height;
            imageSize.depth = 1;

            _cloudImages.push_back(create_image(data, imageSize, VK_FORMAT_R8G8B8A8_UNORM, VK_IMAGE_USAGE_SAMPLED_BIT,true));

            stbi_image_free(data);

            VkSamplerCreateInfo samplerInfo = {};
            samplerInfo.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
            samplerInfo.magFilter = VK_FILTER_LINEAR;
            samplerInfo.minFilter = VK_FILTER_LINEAR;

            vkCreateSampler(_device, &samplerInfo, nullptr, &_cloudSamplers[i]);
        }
    }


    VkSamplerCreateInfo samplerInfo = {};
    samplerInfo.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;

    samplerInfo.magFilter = VK_FILTER_NEAREST;
    samplerInfo.minFilter = VK_FILTER_NEAREST;

    vkCreateSampler(_device, &samplerInfo, nullptr, &_defaultSamplerNearest);

    samplerInfo.magFilter = VK_FILTER_LINEAR;
    samplerInfo.minFilter = VK_FILTER_LINEAR;
    

    vkCreateSampler(_device, &samplerInfo, nullptr, &_defaultSamplerLinear);

    _mainDeletionQueue.push_function([&]() {
        vkDestroySampler(_device, _defaultSamplerNearest, nullptr);
        vkDestroySampler(_device, _defaultSamplerLinear, nullptr);

        destroy_image(_whiteImage);
        destroy_image(_greyImage);
        destroy_image(_errorCheckImage);

        for (int i = _cloudImages.size(); i--;)
        {
            destroy_image(_cloudImages[i]);
            vkDestroySampler(_device, _cloudSamplers[i], nullptr);
        }
        });
    mainCamera.velocity = glm::vec3(0.f);
    mainCamera.position = glm::vec3(30.f, -0.f, -85.f);

    mainCamera.pitch = 0;
    mainCamera.yaw = 0;

    LightStruct light1 = {};
    light1.color = glm::vec3 (1.5f, 0.f, 0.f);
    light1.position = glm::vec3(30.f, -0, -85.f);
    light1.range = 15.f;
    light1.constant = 1.0f;
    light1.linear = 0.1f;
    light1.quadratic = 0.1f;
    light1.intensity = 10;

    LightStruct light2 = {};
    light2.color = glm::vec3(0.0f, 0.0f, 1.0f);
    light2.position = glm::vec3(20.f, -0, -85.f);
    light2.range = 15.f;
    light2.constant = 1.0f;
    light2.linear = 0.1f;
    light2.quadratic = 0.1f;
    light2.intensity = 10;

    sceneLights.push_back(light1);
    sceneLights.push_back(light2);


    _renderTimeTimer = new Timer("Render Time");
}

void VulkanEngine::init_volumetric_data()
{
    std::array<Vertex, 8> vertices = {
        Vertex{{-1.0f, -1.0f, -1.0f}},
        Vertex{{ 1.0f, -1.0f, -1.0f}},
        Vertex{{ 1.0f,  1.0f, -1.0f}},
        Vertex{{-1.0f,  1.0f, -1.0f}}, 
        Vertex{{-1.0f, -1.0f,  1.0f}}, 
        Vertex{{ 1.0f, -1.0f,  1.0f}}, 
        Vertex{{ 1.0f,  1.0f,  1.0f}}, 
        Vertex{{-1.0f,  1.0f,  1.0f}}  
    };

    std::array<uint32_t, 36> indices = {   
        0, 1, 2, 2, 3, 0,
        4, 5, 6, 6, 7, 4,
        0, 4, 7, 7, 3, 0,
        1, 5, 6, 6, 2, 1,
        0, 1, 5, 5, 4, 0,
        3, 2, 6, 6, 7, 3
    };
    GPUMeshBuffers mesh = upload_mesh(indices, vertices);

    RenderObject obj;
    
    _volumetricMaterial.passType = MaterialPass::Volumetric;
    _volumetricMaterial.pipeline = &_volumetricPipeline;

    obj.indexCount = indices.size();
    obj.firstIndex = 0;
    obj.indexBuffer = mesh.indexBuffer.buffer;
    obj.material = &_volumetricMaterial;
    obj.transform = glm::translate(glm::mat4(1.0f), glm::vec3(1.0f, 2.0f, 3.0f));
    obj.vertexBufferAddress = mesh.vertexBufferAddress;
    obj.meshBuffer = mesh;

    mainDrawContext.VolumetricSurfaces.push_back(obj);


    //todo create 3d image for voxel data
    for (int i = 128; i--;)
    {
        _cloudVoxels.density[i] = (float)(i / 128);
    }
    VkExtent3D imageSize;
    imageSize.width = 32;
    imageSize.height = 32;
    imageSize.depth = 32;

    std::vector<uint8_t> voxelData(128); 
    for (size_t i = 0; i < voxelData.size(); i++) 
    {
        voxelData[i] = static_cast<uint8_t>(glm::clamp(_cloudVoxels.density[i], 0.0f, 1.0f) * 255.0f);
    }
    _cloudVoxelImage = create_image(voxelData.data(), imageSize, VK_FORMAT_R8_UNORM, VK_IMAGE_USAGE_SAMPLED_BIT, false);


    _mainDeletionQueue.push_function([&]() {
        destroy_image(_cloudVoxelImage);
        });
}

void VulkanEngine::init_billboard_data()
{

    std::array<Vertex, 4> vertices = {
 Vertex{{-1.0f, -1.0f, 0.0f}, 0.0f, {0.0f, 0.0f, 1.0f}, 0.0f, {1.0f, 0.0f, 0.0f, 1.0f}},

 Vertex{{ 1.0f, -1.0f, 0.0f}, 1.0f, {0.0f, 0.0f, 1.0f}, 0.0f, {0.0f, 1.0f, 0.0f, 1.0f}},

 Vertex{{ 1.0f,  1.0f, 0.0f}, 1.0f, {0.0f, 0.0f, 1.0f}, 1.0f, {0.0f, 0.0f, 1.0f, 1.0f}},

 Vertex{{-1.0f,  1.0f, 0.0f}, 0.0f, {0.0f, 0.0f, 1.0f}, 1.0f, {1.0f, 1.0f, 1.0f, 1.0f}}
    };

    std::array<uint32_t, 6> indices = {
        0, 1, 2, 
        2, 3, 0   
    };
    GPUMeshBuffers mesh = upload_mesh(indices, vertices);

    RenderObject obj;
    _billboardTransparencyType=0;
    _billboardMaterial.passType = MaterialPass::Billboard;
    _billboardMaterial.pipeline = &_billboardPipeline[0];
    


    obj.indexCount = indices.size();
    obj.instanceCount = 128;
    obj.firstIndex = 0;
    obj.indexBuffer = mesh.indexBuffer.buffer;
    obj.material = &_billboardMaterial;
    obj.transform = glm::mat4{ 1.f };

    obj.vertexBufferAddress = mesh.vertexBufferAddress;
    obj.meshBuffer = mesh;

    mainDrawContext.BillboardSurfaces.push_back(obj);


    for (int i = 0; i < obj.instanceCount; i++)
    {
        _billboardData.billboardPos[i] =glm::vec4(rand()%(101), rand() % (101), rand() % (101), 0);

        _billboardData.scale[i/4][i%4] = 10.0f;

        _billboardData.texIndex[i/4][i%4] = rand() % (10);
    }

    _mainDeletionQueue.push_function([&]() {
        
        });
}

void VulkanEngine::init_imgui()
{
    //create descriptor pool for imgui
    VkDescriptorPoolSize poolSizes[] = {
        { VK_DESCRIPTOR_TYPE_SAMPLER, 1000},
        { VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 1000 },
        { VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1000 },
        { VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER, 1000 },
        { VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER, 1000 },
        { VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1000 },
        { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1000 },
        { VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC, 1000 },
        { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC, 1000 },
        { VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT, 1000 }
    };

    VkDescriptorPoolCreateInfo poolInfo = {};
    poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    poolInfo.flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;
    poolInfo.maxSets = 1000;
    poolInfo.poolSizeCount = (uint32_t)std::size(poolSizes);
    poolInfo.pPoolSizes = poolSizes;

    VkDescriptorPool imguiPool;
    VK_CHECK(vkCreateDescriptorPool(_device, &poolInfo, nullptr, &imguiPool));

    //initialise imgui library

    ImGui::CreateContext();

    ImGui_ImplSDL2_InitForVulkan(_window);

    ImGui_ImplVulkan_InitInfo initInfo = {};
    initInfo.Instance = _instance;
    initInfo.PhysicalDevice = _chosenGPU;
    initInfo.Device = _device;
    initInfo.Queue = _graphicsQueue;
    initInfo.DescriptorPool = imguiPool;
    initInfo.MinImageCount = 3;
    initInfo.ImageCount = 3;
    initInfo.UseDynamicRendering = true;

    //dynamic rendering params
    initInfo.PipelineRenderingCreateInfo = { .sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO };
    initInfo.PipelineRenderingCreateInfo.colorAttachmentCount = 1;
    initInfo.PipelineRenderingCreateInfo.pColorAttachmentFormats = &_swapchainImageFormat;

    initInfo.MSAASamples = VK_SAMPLE_COUNT_1_BIT;

    ImGui_ImplVulkan_Init(&initInfo);

    ImGui_ImplVulkan_CreateFontsTexture();

    _mainDeletionQueue.push_function([=]
        {
            ImGui_ImplVulkan_Shutdown();
            vkDestroyDescriptorPool(_device, imguiPool, nullptr);
        });
}

void VulkanEngine::init_sync_structures()
{
    VkFenceCreateInfo fenceCreateInfo =
        vkinit::fence_create_info(VK_FENCE_CREATE_SIGNALED_BIT);
    VkSemaphoreCreateInfo semaphoreCreateInfo =
        vkinit::semaphore_create_info();

    for (int i = 0; i < FRAME_OVERLAP; i++)
    {
        VK_CHECK(vkCreateFence(_device, &fenceCreateInfo, nullptr, &_frames[i]._renderFence));

        VK_CHECK(vkCreateSemaphore(_device, &semaphoreCreateInfo, nullptr, &_frames[i]._swapchainSemaphore));
        VK_CHECK(vkCreateSemaphore(_device, &semaphoreCreateInfo, nullptr, &_frames[i]._renderSemaphore));
    }

    VK_CHECK(vkCreateFence(_device, &fenceCreateInfo, nullptr, &_immFence));
    _mainDeletionQueue.push_function([=]() {vkDestroyFence(_device, _immFence, nullptr); });
}

void VulkanEngine::create_swapchain(uint32_t width, uint32_t height)
{
    vkb::SwapchainBuilder swapchainBuilder{ _chosenGPU,_device, _surface };

    _swapchainImageFormat = VK_FORMAT_R8G8B8A8_UNORM;

    vkb::Swapchain vkbSwapchain = swapchainBuilder
        //.use_default_format_selection()
        .set_desired_format(VkSurfaceFormatKHR{ .format = _swapchainImageFormat,.colorSpace = VK_COLOR_SPACE_SRGB_NONLINEAR_KHR })
        //use vsync present mode
        .set_desired_present_mode(VK_PRESENT_MODE_FIFO_KHR)
        .set_desired_extent(width, height)
        .add_image_usage_flags(VK_IMAGE_USAGE_TRANSFER_DST_BIT)
        .build()
        .value();

    _swapchainExtent = vkbSwapchain.extent;

    _swapchain = vkbSwapchain.swapchain;
    _swapchainImages = vkbSwapchain.get_images().value();
    _swapchainImageViews = vkbSwapchain.get_image_views().value();
}

void VulkanEngine::resize_swapchain()
{
    vkDeviceWaitIdle(_device);

    destroy_swapchain();

    int w, h;
    SDL_GetWindowSize(_window, &w, &h);
    _windowExtent.width = w;
    _windowExtent.height = h;

    create_swapchain(_windowExtent.width, _windowExtent.height);

    resize_requested = false;
}

void VulkanEngine::destroy_swapchain()
{
    vkDestroySwapchainKHR(_device, _swapchain, nullptr);

    for (int i = 0; i < _swapchainImageViews.size(); i++)
    {
        vkDestroyImageView(_device, _swapchainImageViews[i], nullptr);
    }
}

AllocatedBuffer VulkanEngine::create_buffer(size_t allocSize, VkBufferUsageFlags usage, VmaMemoryUsage memoryUsage)
{
    //allocate buffer
    VkBufferCreateInfo bufferInfo = { .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO };
    bufferInfo.pNext = nullptr;
    bufferInfo.size = allocSize;

    bufferInfo.usage = usage;

    VmaAllocationCreateInfo vmaAllocInfo = {};
    vmaAllocInfo.usage = memoryUsage;
    vmaAllocInfo.flags = VMA_ALLOCATION_CREATE_MAPPED_BIT;
    AllocatedBuffer newBuffer;

    VK_CHECK(vmaCreateBuffer(_allocator, &bufferInfo, &vmaAllocInfo, &newBuffer.buffer, &newBuffer.allocation, &newBuffer.info));
    
    return newBuffer;
}

void VulkanEngine::destroy_buffer(const AllocatedBuffer& buffer)
{
    vmaDestroyBuffer(_allocator, buffer.buffer, buffer.allocation);
}

GPUMeshBuffers VulkanEngine::upload_mesh(std::span<uint32_t> indices, std::span<Vertex> vertices)
{
    const size_t vertexBufferSize = vertices.size() * sizeof(Vertex);
    const size_t indexBufferSize = indices.size() * sizeof(uint32_t);

    GPUMeshBuffers newSurface;

    //create vertex buffer
    newSurface.vertexBuffer =
        create_buffer(vertexBufferSize, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT |
            VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT_COPY, VMA_MEMORY_USAGE_GPU_ONLY);

    //find address of vertex buffer
    VkBufferDeviceAddressInfo deviceAddressInfo{ .sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO };
    deviceAddressInfo.buffer = newSurface.vertexBuffer.buffer;
    
    newSurface.vertexBufferAddress = vkGetBufferDeviceAddress(_device, &deviceAddressInfo);

    //create index buffer
    newSurface.indexBuffer =
        create_buffer(indexBufferSize, VK_BUFFER_USAGE_INDEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
            VMA_MEMORY_USAGE_GPU_ONLY);

    //staging buffer and copy
    AllocatedBuffer staging =
        create_buffer(vertexBufferSize + indexBufferSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VMA_MEMORY_USAGE_CPU_ONLY);

    void* data = staging.allocation->GetMappedData();

    memcpy(data, vertices.data(), vertexBufferSize);

    memcpy((char*)data + vertexBufferSize, indices.data(), indexBufferSize);

    immediate_submit([&](VkCommandBuffer cmd)
        {
            VkBufferCopy vertexCopy{ 0 };
            vertexCopy.dstOffset = 0;
            vertexCopy.srcOffset = 0;
            vertexCopy.size = vertexBufferSize;

            vkCmdCopyBuffer(cmd, staging.buffer, newSurface.vertexBuffer.buffer, 1, &vertexCopy);

            VkBufferCopy indexCopy{ 0 };
            indexCopy.dstOffset = 0;
            indexCopy.srcOffset = vertexBufferSize;
            indexCopy.size = indexBufferSize;

            vkCmdCopyBuffer(cmd, staging.buffer, newSurface.indexBuffer.buffer, 1, &indexCopy);
        });

    destroy_buffer(staging);

    return newSurface;
}

AllocatedImage VulkanEngine::create_image(VkExtent3D size, VkFormat format, VkImageUsageFlags usage, bool mipmapped)
{
    AllocatedImage newImage;
    newImage.imageFormat = format;
    newImage.imageExtent = size;

    VkImageCreateInfo imgInfo = vkinit::image_create_info(format, usage, size);
    if (mipmapped)
    {
        imgInfo.mipLevels = static_cast<uint32_t>(std::floor(std::log2(std::max(size.width, size.height)))) + 1;
    }

    VmaAllocationCreateInfo allocInfo = {};
    allocInfo.usage = VMA_MEMORY_USAGE_GPU_ONLY;
    allocInfo.requiredFlags = VkMemoryPropertyFlags(VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

    VK_CHECK(vmaCreateImage(_allocator, &imgInfo, &allocInfo, &newImage.image, &newImage.allocation, nullptr));

    VkImageAspectFlags aspectFlag = VK_IMAGE_ASPECT_COLOR_BIT;
    if (format == VK_FORMAT_D32_SFLOAT)
    {
        aspectFlag = VK_IMAGE_ASPECT_DEPTH_BIT;
    }

    VkImageViewCreateInfo viewInfo = vkinit::imageview_create_info(format, newImage.image, aspectFlag);
    viewInfo.subresourceRange.levelCount = imgInfo.mipLevels;

    if (size.depth != 1)
    {
        viewInfo.viewType = VK_IMAGE_VIEW_TYPE_3D;
    }

    VK_CHECK(vkCreateImageView(_device, &viewInfo, nullptr, &newImage.imageView));

    return newImage;
}

AllocatedImage VulkanEngine::create_image(void* data, VkExtent3D size, VkFormat format, VkImageUsageFlags usage, bool mipmapped)
{
    size_t dataSize;
    if (size.depth != 1)
    {
        dataSize = size.depth * size.width * size.height * 1;
    }
    else
    {
        dataSize = size.depth * size.width * size.height * 4;
    }
    AllocatedBuffer uploadBuffer = create_buffer(dataSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VMA_MEMORY_USAGE_CPU_TO_GPU);

    memcpy(uploadBuffer.info.pMappedData, data, dataSize);

    AllocatedImage newImage = create_image(size, format, usage | VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT, mipmapped);

    immediate_submit([&](VkCommandBuffer cmd)
        {
            vkutil::transition_image(cmd, newImage.image, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);

            VkBufferImageCopy copyRegion = {};
            copyRegion.bufferOffset = 0;
            copyRegion.bufferRowLength = 0;
            copyRegion.bufferImageHeight = 0;

            copyRegion.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
            copyRegion.imageSubresource.mipLevel = 0;
            copyRegion.imageSubresource.baseArrayLayer = 0;
            copyRegion.imageSubresource.layerCount = 1;
            copyRegion.imageExtent = size;

            vkCmdCopyBufferToImage(cmd, uploadBuffer.buffer, newImage.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &copyRegion);

            if (mipmapped)
            {

                vkutil::generate_mipmaps(cmd, newImage.image, VkExtent2D{ newImage.imageExtent.width, newImage.imageExtent.height });
            }
            else
            {

                vkutil::transition_image(cmd, newImage.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                    VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
            }
        });

    destroy_buffer(uploadBuffer);

    return newImage;
}

void VulkanEngine::update_scene()
{
    auto start = std::chrono::system_clock::now();

    mainDrawContext.OpaqueSurfaces.clear();
    mainDrawContext.TransparentSurfaces.clear();

    loadedScenes["structure"]->Draw(glm::mat4{ 1.f }, mainDrawContext);

    mainCamera.update();
    sceneData.view = mainCamera.getViewMatrix();

    sceneData.proj = glm::perspective(glm::radians(70.f), (float)_windowExtent.width / (float)_windowExtent.height, 10000.f, 0.1f);

    sceneData.proj[1][1] *= -1;
    sceneData.viewproj = sceneData.proj * sceneData.view;

    sceneData.ambientColor = glm::vec4(0.3f, 0.3f, 0.3f, 1.0f);

    sceneData.sunlightColor = glm::vec4(1.0f, 0.8f, 0.6f, 1.0f);

    sceneData.sunlightDirection = glm::vec4(0.5f, -0.5f, 0.5f, 1.0f);

    update_volumetrics();
    update_billboards();

    auto end = std::chrono::system_clock::now();

    auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    stats.sceneUpdateTime = elapsed.count() / 1000.f;
}

void VulkanEngine::update_volumetrics()
{

    // todo 

}

void VulkanEngine::update_billboards()
{
    if (_billboardTransparencyType == 0)
    {
        _billboardMaterial.pipeline = &_billboardPipeline[0];
    }
    else
    {
        _billboardMaterial.pipeline = &_billboardPipeline[1];
    }
}

void VulkanEngine::draw()
{
    update_scene();

    //wait for gpu to render last frame
    VK_CHECK(vkWaitForFences(_device, 1, &get_current_frame()._renderFence, true, 1000000000));

    get_current_frame()._deletionQueue.flush();
    get_current_frame()._frameDescriptors.clear_descriptors(_device);

    VK_CHECK(vkResetFences(_device, 1, &get_current_frame()._renderFence));

    uint32_t swapchainImageIndex;
    VkResult e = vkAcquireNextImageKHR(_device, _swapchain, 1000000000, get_current_frame()._swapchainSemaphore, nullptr, &swapchainImageIndex);
    if (e == VK_ERROR_OUT_OF_DATE_KHR)
    {
        resize_requested = true;
        return;
    }

    VkCommandBuffer cmd = get_current_frame()._mainCommandBuffer;

    VK_CHECK(vkResetCommandBuffer(cmd, 0));

    VkCommandBufferBeginInfo cmdBeginInfo =
        vkinit::command_buffer_begin_info(VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT);

    _drawExtent.width = std::min(_swapchainExtent.width, _drawImage.imageExtent.width) * renderScale;
    _drawExtent.height = std::min(_swapchainExtent.height, _drawImage.imageExtent.height) * renderScale;

    //record to command buffer
    VK_CHECK(vkBeginCommandBuffer(cmd, &cmdBeginInfo));

    vkutil::transition_image(cmd, _drawImage.image, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL);

    draw_background(cmd);

    vkutil::transition_image(cmd, _drawImage.image, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
    vkutil::transition_image(cmd, _depthImage.image, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL);

    draw_geometry(cmd);

    vkutil::transition_image(cmd, _drawImage.image, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);
    vkutil::transition_image(cmd, _swapchainImages[swapchainImageIndex], VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);

    vkutil::copy_image_to_image(cmd, _drawImage.image, _swapchainImages[swapchainImageIndex], _drawExtent, _swapchainExtent);

    vkutil::transition_image(cmd, _swapchainImages[swapchainImageIndex], VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);

    draw_imgui(cmd, _swapchainImageViews[swapchainImageIndex]);

    vkutil::transition_image(cmd, _swapchainImages[swapchainImageIndex], VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_PRESENT_SRC_KHR);


    VK_CHECK(vkEndCommandBuffer(cmd));

    VkCommandBufferSubmitInfo cmdInfo =
        vkinit::command_buffer_submit_info(cmd);

    VkSemaphoreSubmitInfo waitInfo =
        vkinit::semaphore_submit_info(VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT_KHR, get_current_frame()._swapchainSemaphore);
    VkSemaphoreSubmitInfo signalInfo =
        vkinit::semaphore_submit_info(VK_PIPELINE_STAGE_2_ALL_GRAPHICS_BIT, get_current_frame()._renderSemaphore);

    VkSubmitInfo2 submit =
        vkinit::submit_info(&cmdInfo, &signalInfo, &waitInfo);

    VK_CHECK(vkQueueSubmit2(_graphicsQueue, 1, &submit, get_current_frame()._renderFence));

    VkPresentInfoKHR presentInfo = {};
    presentInfo.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
    presentInfo.pNext = nullptr;
    presentInfo.pSwapchains = &_swapchain;
    presentInfo.swapchainCount = 1;

    presentInfo.pWaitSemaphores = &get_current_frame()._renderSemaphore;
    presentInfo.waitSemaphoreCount = 1;

    presentInfo.pImageIndices = &swapchainImageIndex;

    VkResult presentResult = vkQueuePresentKHR(_graphicsQueue, &presentInfo);
    if (presentResult == VK_ERROR_OUT_OF_DATE_KHR)
    {
        resize_requested = true;
    }

    _frameNumber++;
}

void VulkanEngine::draw_background(VkCommandBuffer cmd)
{
    VkClearColorValue clearValue;
    float flash = std::abs(std::sin(_frameNumber / 120.f));
    clearValue = { {0.0f,0.f,flash,1.f} };

    VkImageSubresourceRange clearRange = vkinit::image_subresource_range(VK_IMAGE_ASPECT_COLOR_BIT);

    ComputeEffect& effect = backgroundEffects[currentBackgroundEffect];

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, effect.pipeline);

    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, _gradientPipelineLayout, 0, 1, &_drawImageDescriptors, 0, nullptr);

    vkCmdPushConstants(cmd, _gradientPipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(ComputePushConstants), &effect.data);

    vkCmdDispatch(cmd, std::ceil(_drawExtent.width / 16.0), std::ceil(_drawExtent.height / 16.0), 1);
}

void VulkanEngine::draw_geometry(VkCommandBuffer cmd)
{
    _renderTimeTimer->Start();

    std::vector<uint32_t> opaqueDraws;
    opaqueDraws.reserve(mainDrawContext.OpaqueSurfaces.size());

    for (uint32_t i = 0; i < mainDrawContext.OpaqueSurfaces.size(); i++)
    {
        if (is_visible(mainDrawContext.OpaqueSurfaces[i], sceneData.viewproj))
        {
            opaqueDraws.push_back(i);
        }
    }

    std::sort(opaqueDraws.begin(), opaqueDraws.end(), [&](const auto& iA, const auto& iB)
        {
            const RenderObject& A = mainDrawContext.OpaqueSurfaces[iA];
            const RenderObject& B = mainDrawContext.OpaqueSurfaces[iB];
            if (A.material == B.material)
            {
                return A.indexBuffer < B.indexBuffer;
            }
            else
            {
                return A.material < B.material;
            }
        });

    BillboardData sortedBillboardData;
    std::vector<CloudInstance> cloudInstances;

    if (_billboardTransparencyType == 0)
    {
        for (int i = 0; i < 128; i++)
        {
            float distance = glm::length(mainCamera.position - glm::vec3(_billboardData.billboardPos[i].x, _billboardData.billboardPos[i].y, _billboardData.billboardPos[i].z));
            cloudInstances.push_back({ distance,i });
        }

        std::sort(cloudInstances.begin(), cloudInstances.end(), [](CloudInstance& a, CloudInstance& b) {
            return a.distance > b.distance;
            });
        for (int i = 0; i < 128; i++)
        {
            int sortedIndex = cloudInstances[i].index;

            sortedBillboardData.billboardPos[i] = _billboardData.billboardPos[sortedIndex];
            sortedBillboardData.scale[i / 4][i % 4] = _billboardData.scale[sortedIndex / 4][sortedIndex % 4];
            sortedBillboardData.texIndex[i / 4][i % 4] = _billboardData.texIndex[sortedIndex / 4][sortedIndex % 4];
        }
    }
    else
    {
        sortedBillboardData = _billboardData;
    }

    VkRenderingAttachmentInfo colorAttachment =
        vkinit::attachment_info(_drawImage.imageView, nullptr, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
    VkRenderingAttachmentInfo depthAttachment =
        vkinit::depth_attachment_info(_depthImage.imageView, VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL);


    VkRenderingInfo renderInfo =
        vkinit::rendering_info(_drawExtent, &colorAttachment, &depthAttachment);
    vkCmdBeginRendering(cmd, &renderInfo);

    //handle scene data
    AllocatedBuffer gpuSceneDataBuffer = create_buffer(sizeof(GPUSceneData), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, VMA_MEMORY_USAGE_CPU_TO_GPU);

    GPUSceneData* sceneUniformData = (GPUSceneData*)gpuSceneDataBuffer.allocation->GetMappedData();
    *sceneUniformData = sceneData;

    VkDescriptorSet globalDescriptor = get_current_frame()._frameDescriptors.allocate(_device, _gpuSceneDataDescriptorLayout);

    DescriptorWriter writer;
    writer.write_buffer(0, gpuSceneDataBuffer.buffer, sizeof(sceneData), 0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER);
    writer.update_set(_device, globalDescriptor);

    AllocatedBuffer gpuLightBuffer = create_buffer(sizeof(LightBuffer), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, VMA_MEMORY_USAGE_CPU_TO_GPU);
    LightBuffer* lightBufferData = (LightBuffer*)gpuLightBuffer.allocation->GetMappedData();

    VkDescriptorSet lightDescriptor = get_current_frame()._frameDescriptors.allocate(_device, _gpuLightDataDescriptorLayout);
    writer.write_buffer(0, gpuLightBuffer.buffer, sizeof(lightData), 0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER);
    writer.update_set(_device, lightDescriptor);

    AllocatedBuffer gpuBillboardBuffer = create_buffer(sizeof(BillboardData), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, VMA_MEMORY_USAGE_CPU_TO_GPU);
    BillboardData* billboardBufferData = (BillboardData*)gpuBillboardBuffer.allocation->GetMappedData();
    *billboardBufferData = sortedBillboardData;
    VkDescriptorSet billboardPosDescriptor = get_current_frame()._frameDescriptors.allocate(_device, _billboardPositionsDescriptorLayout);

    writer.write_buffer(0, gpuBillboardBuffer.buffer, sizeof(_billboardData), 0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER);
    writer.update_set(_device, billboardPosDescriptor);

    //reset counters
    stats.drawcallCount = 0;
    stats.triangleCount = 0;


    MaterialPipeline* lastPipeline = nullptr;
    MaterialInstance* lastMaterial = nullptr;
    VkBuffer lastIndexBuffer = VK_NULL_HANDLE;
    LightBuffer lastLightData{};

    auto draw = [&](const RenderObject& draw)
        {

            if (draw.material != lastMaterial)
            {
                lastMaterial = draw.material;

                if (draw.material->pipeline != lastPipeline)
                {
                    lastPipeline = draw.material->pipeline;
                    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, draw.material->pipeline->pipeline);

                    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, draw.material->pipeline->layout, 0, 1, &globalDescriptor, 0, nullptr);


                    VkViewport viewport = {};
                    viewport.x = 0;
                    viewport.y = 0;
                    viewport.width = _drawExtent.width;
                    viewport.height = _drawExtent.height;
                    viewport.minDepth = 0.f;
                    viewport.maxDepth = 1.f;

                    vkCmdSetViewport(cmd, 0, 1, &viewport);

                    VkRect2D scissor = {};
                    scissor.offset.x = 0;
                    scissor.offset.y = 0;
                    scissor.extent.width = _drawExtent.width;
                    scissor.extent.height = _drawExtent.height;

                    vkCmdSetScissor(cmd, 0, 1, &scissor);
                }

                if (draw.material->passType == MaterialPass::Billboard)
                {
                    draw.material->materialSet = get_current_frame()._frameDescriptors.allocate(_device, _billboardMaterialDescriptorLayout);

                    DescriptorWriter writer;
                    for (int i = _cloudImages.size(); i--;)
                    {
                        writer.write_image(i + 1, _cloudImages[i].imageView, _cloudSamplers[i], VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER);
                    }

                    writer.update_set(_device, draw.material->materialSet);

                    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, draw.material->pipeline->layout, 2, 1, &billboardPosDescriptor, 0, nullptr);

                    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, draw.material->pipeline->layout, 1, 1, &draw.material->materialSet, 0, nullptr);
                }


                if (draw.material->passType != MaterialPass::Volumetric && draw.material->passType != MaterialPass::Billboard)
                {
                    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, draw.material->pipeline->layout, 1, 1, &draw.material->materialSet, 0, nullptr);
                }
                else if(draw.material->passType==MaterialPass::Volumetric && draw.material->passType != MaterialPass::Billboard)
                {
                    //todo send a 3d texture to gpu
                    //vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, draw.material->pipeline->layout, 1, 1, , 0, nullptr);
                }

            }

            if (draw.indexBuffer != lastIndexBuffer)
            {
                lastIndexBuffer = draw.indexBuffer;
                vkCmdBindIndexBuffer(cmd, draw.indexBuffer, 0, VK_INDEX_TYPE_UINT32);
            }


            //handle light data
            //todo sort light buffer
            //todo clear light data
            if (draw.material->passType != MaterialPass::Volumetric && draw.material->passType != MaterialPass::Billboard)
            {
                std::fill(std::begin(lightData.lights), std::end(lightData.lights), LightStruct{});
                lightData.numLights = 0;

                for (const auto& l : sceneLights)
                {
                    if (lightData.numLights < 10)
                    {
                        lightData.lights[lightData.numLights++] = l;
                    }

                }

                *lightBufferData = lightData;
                vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, draw.material->pipeline->layout, 2, 1, &lightDescriptor, 0, nullptr);
            }




            GPUDrawPushConstants pushConstants;
            pushConstants.vertexBuffer = draw.vertexBufferAddress;
            pushConstants.worldMatrix = draw.transform;
            vkCmdPushConstants(cmd, draw.material->pipeline->layout, VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(GPUDrawPushConstants), &pushConstants);

            if (draw.material->passType == MaterialPass::Billboard)
            {
                vkCmdDrawIndexed(cmd, draw.indexCount, 128, draw.firstIndex, 0, 0);
            }
            else
            {
                vkCmdDrawIndexed(cmd, draw.indexCount, 1, draw.firstIndex, 0, 0);
            }

            stats.drawcallCount++;
            stats.triangleCount += draw.indexCount / 3;

        };



    for (auto r : opaqueDraws)
    {
        draw(mainDrawContext.OpaqueSurfaces[r]);
    }
    for (auto r : mainDrawContext.TransparentSurfaces)
    {
        draw(r);
    }
    for (auto r : mainDrawContext.BillboardSurfaces)
    {
        draw(r);
    }
    for (auto r : mainDrawContext.VolumetricSurfaces)
    {
        draw(r);
    }

    get_current_frame()._deletionQueue.push_function([=, this]() {
        destroy_buffer(gpuSceneDataBuffer);
        destroy_buffer(gpuLightBuffer);
        destroy_buffer(gpuBillboardBuffer);
        });


    auto end = std::chrono::system_clock::now();

    _renderTimeTimer->Stop();
    stats.meshDrawTime = _renderTimeTimer->GetElapsed();


    vkCmdEndRendering(cmd);
}

void VulkanEngine::draw_imgui(VkCommandBuffer cmd, VkImageView targetImageView)
{
    VkRenderingAttachmentInfo colorAttachment =
        vkinit::attachment_info(targetImageView, nullptr, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
    VkRenderingInfo renderInfo =
        vkinit::rendering_info(_swapchainExtent, &colorAttachment, nullptr);

    vkCmdBeginRendering(cmd, &renderInfo);

    ImGui_ImplVulkan_RenderDrawData(ImGui::GetDrawData(), cmd);

    vkCmdEndRendering(cmd);
}

void VulkanEngine::immediate_submit(std::function<void(VkCommandBuffer cmd)>&& function)
{
    VK_CHECK(vkResetFences(_device, 1, &_immFence));
    VK_CHECK(vkResetCommandBuffer(_immCommandBuffer, 0));

    VkCommandBuffer cmd = _immCommandBuffer;

    VkCommandBufferBeginInfo cmdBeginInfo =
        vkinit::command_buffer_begin_info(VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT);

    VK_CHECK(vkBeginCommandBuffer(cmd, &cmdBeginInfo));

    function(cmd);

    VK_CHECK(vkEndCommandBuffer(cmd));

    VkCommandBufferSubmitInfo cmdInfo =
        vkinit::command_buffer_submit_info(cmd);
    VkSubmitInfo2 submit =
        vkinit::submit_info(&cmdInfo, nullptr, nullptr);

    VK_CHECK(vkQueueSubmit2(_graphicsQueue, 1, &submit, _immFence));;

    VK_CHECK(vkWaitForFences(_device, 1, &_immFence, true, 9999999999999));
}

void VulkanEngine::run()
{
    SDL_Event e;
    bool bQuit = false;

    // main loop
    while (!bQuit) {
        auto start = std::chrono::system_clock::now();
        // Handle events on queue
        while (SDL_PollEvent(&e) != 0) {
            // close the window when user alt-f4s or clicks the X button
            if (e.type == SDL_QUIT)
                bQuit = true;

            if (e.type == SDL_KEYDOWN)
            {
                if (e.key.keysym.sym == SDLK_ESCAPE)
                {
                    bQuit = true;
                }
                if (e.key.keysym.sym == SDLK_1)
                {
                    if (SDL_GetRelativeMouseMode() == SDL_TRUE)
                    {
                        SDL_SetRelativeMouseMode(SDL_FALSE);
                        mainCamera.isActive = false;
                    }
                    else
                    {
                        SDL_SetRelativeMouseMode(SDL_TRUE);
                        mainCamera.isActive = true;
                    }
                }
            }

            mainCamera.processSDLEvent(e);
            ImGui_ImplSDL2_ProcessEvent(&e);

            if (e.type == SDL_WINDOWEVENT) {
                if (e.window.event == SDL_WINDOWEVENT_MINIMIZED) {
                    stop_rendering = true;
                }
                if (e.window.event == SDL_WINDOWEVENT_RESTORED) {
                    stop_rendering = false;
                }
            }
        }

        // do not draw if we are minimized
        if (stop_rendering) {
            // throttle the speed to avoid the endless spinning
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
            continue;
        }

        if (resize_requested)
        {
            resize_swapchain();
        }

        ImGui_ImplVulkan_NewFrame();
        ImGui_ImplSDL2_NewFrame();
        ImGui::NewFrame();
        
        if (ImGui::Begin("debugging"))
        {
            if (ImGui::BeginTabBar("debugging"))
            {
                if (ImGui::BeginTabItem("billboard"))
                {
                    ImGui::SliderInt("Transparency Mode", &_billboardTransparencyType, 0, 1);
                    ImGui::EndTabItem();
                }
                if (ImGui::BeginTabItem("stats"))
                {

                    ImGui::Text("Frametime %f ms", stats.frametime);
                    ImGui::Text("Draw time %f ms", stats.meshDrawTime);
                    ImGui::Text("Update time %f ms", stats.sceneUpdateTime);
                    ImGui::Text("Triangles %i", stats.triangleCount);
                    ImGui::Text("Draws %i", stats.drawcallCount);
                    ImGui::EndTabItem();
                }
                ImGui::EndTabBar();
            }
        }
        ImGui::End();

        ImGui::Render();

        draw();

        auto end = std::chrono::system_clock::now();

        auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        stats.frametime = elapsed.count() / 1000.f;
    }
}

void VulkanEngine::destroy_image(const AllocatedImage& img)
{
    vkDestroyImageView(_device, img.imageView, nullptr);
    vmaDestroyImage(_allocator, img.image, img.allocation);
}

void VulkanEngine::cleanup()
{
    if (_isInitialized) {

        vkDeviceWaitIdle(_device);

        loadedScenes.clear();

        for (int i = 0; i < FRAME_OVERLAP; i++)
        {
            vkDestroyCommandPool(_device, _frames[i]._commandPool, nullptr);

            vkDestroyFence(_device, _frames[i]._renderFence, nullptr);
            vkDestroySemaphore(_device, _frames[i]._renderSemaphore, nullptr);
            vkDestroySemaphore(_device, _frames[i]._swapchainSemaphore, nullptr);

            _frames[i]._deletionQueue.flush();
        }

        for (auto& mesh : testMeshes)
        {
            destroy_buffer(mesh->meshBuffers.indexBuffer);
            destroy_buffer(mesh->meshBuffers.vertexBuffer);
        }

        for (int i =0;i<mainDrawContext.VolumetricSurfaces.size();i++)
        {
            destroy_buffer(mainDrawContext.VolumetricSurfaces[i].meshBuffer.indexBuffer);
            destroy_buffer(mainDrawContext.VolumetricSurfaces[i].meshBuffer.vertexBuffer);
        }

        for (int i = 0; i < mainDrawContext.BillboardSurfaces.size(); i++)
        {
            destroy_buffer(mainDrawContext.BillboardSurfaces[i].meshBuffer.indexBuffer);
            destroy_buffer(mainDrawContext.BillboardSurfaces[i].meshBuffer.vertexBuffer);

        }

        metalRoughMaterial.clear_resources(_device);

        _mainDeletionQueue.flush();

        SDL_DestroyWindow(_window);

        destroy_swapchain();

        vkDestroySurfaceKHR(_instance, _surface, nullptr);
        vkDestroyDevice(_device, nullptr);

        vkb::destroy_debug_utils_messenger(_instance, _debug_messenger);
        vkDestroyInstance(_instance, nullptr);
        SDL_DestroyWindow(_window);
    }

    // clear engine pointer
    loadedEngine = nullptr;
}


void GLTFMetallic_Roughness::build_pipelines(VulkanEngine* engine)
{
    VkShaderModule meshFragShader;
    if (!vkutil::load_shader_module("../../shaders/mesh.frag.spv", engine->_device, &meshFragShader))
    {
        fmt::print("Error when building the mesh fragment shader module \n");
    }
    else
    {
        fmt::print("Mesh fragment shader successfully loaded \n");
    }

    VkShaderModule meshVertexShader;
    if (!vkutil::load_shader_module("../../shaders/mesh.vert.spv", engine->_device, &meshVertexShader))
    {
        fmt::print("Error when building the mesh vertex shader module \n");
    }
    else
    {
        fmt::print("Mesh vertex shader successfully loaded \n");
    }

    

    VkPushConstantRange bufferRange{};
    bufferRange.offset = 0;
    bufferRange.size = sizeof(GPUDrawPushConstants);
    bufferRange.stageFlags = VK_SHADER_STAGE_VERTEX_BIT;

    DescriptorLayoutBuilder layoutBuilder;
    layoutBuilder.add_binding(0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER);
    layoutBuilder.add_binding(1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER);
    layoutBuilder.add_binding(2, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER);
    layoutBuilder.add_binding(3, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER);

    materialLayout = layoutBuilder.build(engine->_device, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT);

    VkDescriptorSetLayout layouts[] = { engine->_gpuSceneDataDescriptorLayout, materialLayout, engine->_gpuLightDataDescriptorLayout };

    VkPipelineLayoutCreateInfo meshLayoutInfo = vkinit::pipeline_layout_create_info();
    meshLayoutInfo.setLayoutCount = 3;
    meshLayoutInfo.pSetLayouts = layouts;
    meshLayoutInfo.pPushConstantRanges = &bufferRange;
    meshLayoutInfo.pushConstantRangeCount = 1;

    VkPipelineLayout newLayout;
    VK_CHECK(vkCreatePipelineLayout(engine->_device, &meshLayoutInfo, nullptr, &newLayout));

    opaquePipeline.layout = newLayout;
    transparentPipeline.layout = newLayout;

    PipelineBuilder pipelineBuilder;


    pipelineBuilder.set_shaders(meshVertexShader, meshFragShader);
    pipelineBuilder.set_input_topology(VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST);
    pipelineBuilder.set_polygon_mode(VK_POLYGON_MODE_FILL);
    pipelineBuilder.set_cull_mode(VK_CULL_MODE_NONE, VK_FRONT_FACE_CLOCKWISE);
    pipelineBuilder.set_multisampling_none();
    pipelineBuilder.disable_blending();
    pipelineBuilder.enable_depthtest(true, VK_COMPARE_OP_GREATER_OR_EQUAL);

    pipelineBuilder.set_color_attachment_format(engine->_drawImage.imageFormat);
    pipelineBuilder.set_depth_format(engine->_depthImage.imageFormat);

    pipelineBuilder._pipelineLayout = newLayout;

    opaquePipeline.pipeline = pipelineBuilder.build_pipeline(engine->_device);

    pipelineBuilder.enable_blending_additive();
    pipelineBuilder.enable_depthtest(false, VK_COMPARE_OP_GREATER_OR_EQUAL);

    transparentPipeline.pipeline = pipelineBuilder.build_pipeline(engine->_device);

    vkDestroyShaderModule(engine->_device, meshVertexShader, nullptr);
    vkDestroyShaderModule(engine->_device, meshFragShader, nullptr);
}
void GLTFMetallic_Roughness::clear_resources(VkDevice device)
{
    vkDestroyDescriptorSetLayout(device, materialLayout, nullptr);
    vkDestroyPipelineLayout(device, transparentPipeline.layout, nullptr);

    vkDestroyPipeline(device, transparentPipeline.pipeline, nullptr);
    vkDestroyPipeline(device, opaquePipeline.pipeline, nullptr);

}

MaterialInstance GLTFMetallic_Roughness::write_material(VkDevice device, MaterialPass pass, const MaterialResources& resources, DescriptorAllocatorGrowable descriptorAllocator)
{
    MaterialInstance matData;
    matData.passType = pass;
    if (pass == MaterialPass::Transparent)
    {
        matData.pipeline = &transparentPipeline;
    }
    else
    {
        matData.pipeline = &opaquePipeline;
    }

    matData.materialSet = descriptorAllocator.allocate(device, materialLayout);

    writer.clear();
    writer.write_buffer(0, resources.dataBuffer, sizeof(MaterialConstants), resources.dataBufferOffset, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER);
    writer.write_image(1, resources.colorImage.imageView, resources.colorSampler, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER);
    writer.write_image(2, resources.metalRoughImage.imageView, resources.metalRoughSampler, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER);

    writer.update_set(device, matData.materialSet);

    return matData;
}


void MeshNode::Draw(const glm::mat4& topMatrix, DrawContext& ctx)
{
    glm::mat4 nodeMatrix = topMatrix * worldTransform;

    for (auto& s : mesh->surfaces)
    {
        RenderObject def;
        def.indexCount = s.count;
        def.firstIndex = s.startIndex;
        def.instanceCount = 1;
        def.indexBuffer = mesh->meshBuffers.indexBuffer.buffer;
        def.material = &s.material->data;
        def.bounds = s.bounds;
        def.transform = nodeMatrix;
        def.vertexBufferAddress = mesh->meshBuffers.vertexBufferAddress;

        if (s.material->data.passType == MaterialPass::MainColor)
        {
            ctx.OpaqueSurfaces.push_back(def);
        }
        else if (s.material->data.passType == MaterialPass::Transparent)
        {
            ctx.TransparentSurfaces.push_back(def);
        }
 
    }

    Node::Draw(topMatrix, ctx);
}
