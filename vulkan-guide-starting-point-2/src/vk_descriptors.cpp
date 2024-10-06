#include <vk_descriptors.h>
#include <vk_types.h>

void DescriptorLayoutBuilder::add_binding(uint32_t binding, VkDescriptorType type)
{
	VkDescriptorSetLayoutBinding newBind = {};
	newBind.binding = binding;
	newBind.descriptorCount = 1;
	newBind.descriptorType = type;

	bindings.push_back(newBind);
}

void DescriptorLayoutBuilder::clear()
{
	bindings.clear();
}

VkDescriptorSetLayout DescriptorLayoutBuilder::build(VkDevice device, VkShaderStageFlags shaderStages, void* pNext, VkDescriptorSetLayoutCreateFlags flags)
{
	for (auto& b : bindings)
	{
		b.stageFlags |= shaderStages;
	}

	VkDescriptorSetLayoutCreateInfo info = { .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO };
	info.pNext = pNext;

	info.pBindings = bindings.data();
	info.bindingCount = (uint32_t)bindings.size();
	info.flags = flags;

	VkDescriptorSetLayout set;
	VK_CHECK(vkCreateDescriptorSetLayout(device, &info, nullptr, &set));

	return set;
}

void DescriptorAllocator::init_pool(VkDevice device, uint32_t maxSets, std::span<PoolSizeRatio> poolRatios)
{
	std::vector<VkDescriptorPoolSize> poolSizes;
	for (PoolSizeRatio ratio : poolRatios)
	{
		poolSizes.push_back(VkDescriptorPoolSize
			{
				.type = ratio.type,
				.descriptorCount = std::max( uint32_t(ratio.ratio / maxSets), (uint32_t)1) 
			});
	}

	VkDescriptorPoolCreateInfo poolInfo = { .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO };
	poolInfo.flags = 0;
	poolInfo.maxSets = maxSets;
	poolInfo.poolSizeCount = (uint32_t)poolSizes.size();
	poolInfo.pPoolSizes = poolSizes.data();

	vkCreateDescriptorPool(device, &poolInfo, nullptr, &pool);
}
void DescriptorAllocator::clear_descriptors(VkDevice device)
{
	vkResetDescriptorPool(device, pool, 0);
}

void DescriptorAllocator::destroy_pool(VkDevice device)
{
	vkDestroyDescriptorPool(device, pool, nullptr);
}

VkDescriptorSet DescriptorAllocator::allocate(VkDevice device, VkDescriptorSetLayout layout)
{
	VkDescriptorSetAllocateInfo allocInfo = { .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO };
	allocInfo.pNext = nullptr;
	allocInfo.descriptorPool = pool;
	allocInfo.descriptorSetCount = 1;
	allocInfo.pSetLayouts = &layout;

	VkDescriptorSet ds;
	VK_CHECK(vkAllocateDescriptorSets(device, &allocInfo, &ds));

	return ds;
}


//descriptor pool growable
void DescriptorAllocatorGrowable::init(VkDevice device, uint32_t initialSets, std::span<PoolSizeRatio> poolRatios)
{
	ratios.clear();

	for (auto r : poolRatios)
	{
		ratios.push_back(r);
	}

	VkDescriptorPool newPool = create_pool(device, initialSets, poolRatios);

	setsPerPool = initialSets * 1.5;

	readyPools.push_back(newPool);
}
void DescriptorAllocatorGrowable::clear_descriptors(VkDevice device)
{
	for (auto p : readyPools)
	{
		vkResetDescriptorPool(device, p, 0);
	}
	for (auto p : fullPools)
	{
		vkResetDescriptorPool(device, p, 0);
		readyPools.push_back(p);
	}
	fullPools.clear();
}
void DescriptorAllocatorGrowable::destroy_pools(VkDevice device)
{
	for (auto p : readyPools)
	{
		vkDestroyDescriptorPool(device, p, nullptr);
	}
	readyPools.clear();
	for (auto p : fullPools)
	{
		vkDestroyDescriptorPool(device, p, nullptr);
	}
	fullPools.clear();
}

VkDescriptorSet DescriptorAllocatorGrowable::allocate(VkDevice device, VkDescriptorSetLayout layout, void* pNext)
{
	VkDescriptorPool poolToUse = get_pool(device);

	VkDescriptorSetAllocateInfo allocInfo = {};
	allocInfo.pNext = pNext;
	allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
	allocInfo.descriptorPool = poolToUse;
	allocInfo.descriptorSetCount = 1;
	allocInfo.pSetLayouts = &layout;

	VkDescriptorSet ds;
	VkResult result = vkAllocateDescriptorSets(device, &allocInfo, &ds);

	if (result == VK_ERROR_OUT_OF_POOL_MEMORY || result == VK_ERROR_FRAGMENTED_POOL)
	{
		fullPools.push_back(poolToUse);

		poolToUse = get_pool(device);
		allocInfo.descriptorPool = poolToUse;

		VK_CHECK(vkAllocateDescriptorSets(device, &allocInfo, &ds));
	}

	readyPools.push_back(poolToUse);
	return ds;
}

VkDescriptorPool DescriptorAllocatorGrowable::get_pool(VkDevice device)
{
	VkDescriptorPool newPool;
	if (readyPools.size() != 0)
	{
		newPool = readyPools.back();
		readyPools.pop_back();
	}
	else
	{
		newPool = create_pool(device, setsPerPool, ratios);

		setsPerPool = setsPerPool * 1.5;
		if (setsPerPool > 4092)
		{
			setsPerPool = 4092;
		}
	}
	
	return newPool;
}
VkDescriptorPool DescriptorAllocatorGrowable::create_pool(VkDevice device, uint32_t setCount, std::span<PoolSizeRatio> poolRatios)
{
	std::vector<VkDescriptorPoolSize> poolSizes;
	for (PoolSizeRatio ratio : poolRatios)
	{
		poolSizes.push_back(VkDescriptorPoolSize
			{
				.type = ratio.type,
				.descriptorCount = uint32_t(ratio.ratio*setCount)
			});
	}

	VkDescriptorPoolCreateInfo poolInfo = {};
	poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
	poolInfo.flags = 0;
	poolInfo.maxSets = setCount;
	poolInfo.poolSizeCount = (uint32_t)poolSizes.size();
	poolInfo.pPoolSizes = poolSizes.data();

	VkDescriptorPool newPool;
	vkCreateDescriptorPool(device, &poolInfo, nullptr, &newPool);
	return newPool;
}

//descriptor writer

void DescriptorWriter::write_image(int binding, VkImageView image, VkSampler sampler, VkImageLayout layout, VkDescriptorType type)
{
	VkDescriptorImageInfo& info = imageInfos.emplace_back(VkDescriptorImageInfo
		{
			.sampler = sampler,
			.imageView = image,
			.imageLayout = layout
		});

	VkWriteDescriptorSet write = {};
	write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;

	write.dstBinding = binding;
	write.dstSet = VK_NULL_HANDLE;
	write.descriptorCount = 1;
	write.descriptorType = type;
	write.pImageInfo = &info;

	writes.push_back(write);
}
void DescriptorWriter::write_buffer(int binding, VkBuffer buffer, size_t size, size_t offset, VkDescriptorType type)
{
	VkDescriptorBufferInfo& info = bufferInfos.emplace_back(VkDescriptorBufferInfo
		{
			.buffer = buffer,
			.offset = offset,
			.range = size
		});

	VkWriteDescriptorSet write = {};
	write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;

	write.dstBinding = binding;
	write.dstSet = VK_NULL_HANDLE;
	write.descriptorCount = 1;
	write.descriptorType = type;
	write.pBufferInfo = &info;

	writes.push_back(write);
}

void DescriptorWriter::clear()
{
	imageInfos.clear();
	writes.clear();
	bufferInfos.clear();
}
void DescriptorWriter::update_set(VkDevice device, VkDescriptorSet set)
{
	for (VkWriteDescriptorSet& write : writes)
	{
		write.dstSet = set;
	}

	vkUpdateDescriptorSets(device, (uint32_t)writes.size(), writes.data(), 0, nullptr);
}