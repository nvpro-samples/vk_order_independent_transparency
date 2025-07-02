/*
 * Copyright (c) 2020-2025, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * SPDX-FileCopyrightText: Copyright (c) 2020-2025 NVIDIA CORPORATION
 * SPDX-License-Identifier: Apache-2.0
 */


// This file contains implementations of the resource creation functions of
// Sample that are specific to order-independent transparency (for instance,
// render passes, descriptor sets, and A-buffers, but not swapchain creation.)

#include "oit.h"

#include <nvvk/formats.hpp>
#include <nvvk/helpers.hpp>

void Sample::destroyFrameImages()
{
  const VkDevice device = m_app->getDevice();
  m_colorImage.deinit(device, m_allocator);
  m_depthImage.deinit(device, m_allocator);
  m_downsampleImage.deinit(device, m_allocator);
  m_oitABuffer.deinit(device, m_allocator);
  m_oitAuxImage.deinit(device, m_allocator);
  m_oitAuxSpinImage.deinit(device, m_allocator);
  m_oitAuxDepthImage.deinit(device, m_allocator);
  m_oitCounterImage.deinit(device, m_allocator);
  m_oitWeightedColorImage.deinit(device, m_allocator);
  m_oitWeightedRevealImage.deinit(device, m_allocator);
}

void Sample::createFrameImages(VkCommandBuffer cmdBuffer)
{
  destroyFrameImages();

  const VkDevice   device       = m_app->getDevice();
  const VkExtent2D viewportSize = getViewportSize();
  // We implement supersample anti-aliasing by rendering to a larger texture.
  const uint32_t bufferWidth  = viewportSize.width * uint32_t(m_state.supersample);
  const uint32_t bufferHeight = viewportSize.height * uint32_t(m_state.supersample);

  // Offscreen color and depth buffer
  {
    // Color image, created with an sRGB format.
    m_colorImage.init(device, m_allocator, VK_IMAGE_TYPE_2D, VK_IMAGE_ASPECT_COLOR_BIT, VK_FORMAT_B8G8R8A8_SRGB, bufferWidth,
                      bufferHeight, 1, VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT, m_state.msaa);
    NVVK_DBG_NAME(m_colorImage.image.image);
    // We'll put it into the layout for a color attachment later.

    // Depth image
    VkFormat depthFormat = nvvk::findDepthFormat(m_app->getPhysicalDevice());

    m_depthImage.init(device, m_allocator, VK_IMAGE_TYPE_2D, VK_IMAGE_ASPECT_DEPTH_BIT, depthFormat, bufferWidth,
                      bufferHeight, 1, VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT, m_state.msaa);
    NVVK_DBG_NAME(m_depthImage.image.image);

    // Intermediate storage for resolve - 1spp, swapchain sized, with the same format as the color image.
    m_downsampleImage.init(device, m_allocator, VK_IMAGE_TYPE_2D, VK_IMAGE_ASPECT_COLOR_BIT, m_colorImage.getFormat(),
                           viewportSize.width, viewportSize.height, 1,
                           VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT, 1);
    NVVK_DBG_NAME(m_downsampleImage.image.image);

    // Initial resource transitions
    m_colorImage.transitionTo(cmdBuffer, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
    m_depthImage.transitionTo(cmdBuffer, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL);
  }

  // A-buffers

  // Compute which buffers we need to allocate and their sizes
  VkDeviceSize aBufferElementsPerSample = 1;
  VkDeviceSize aBufferStrideBytes       = 0;
  VkFormat     aBufferFormat            = VK_FORMAT_UNDEFINED;

  bool allocCounter  = false;
  bool allocAux      = false;
  bool allocAuxSpin  = false;
  bool allocAuxDepth = false;

  // Mode  Coverage  Sample
  // 1x    False     False
  // MSAA  True      False
  // SSAA  False     True
  const bool coverageShading = m_state.coverageShading();
  const bool sampleShading   = m_state.sampleShading;

  switch(m_state.algorithm)
  {
    case OIT_SIMPLE:
      allocAux                                 = true;
      aBufferElementsPerSample                 = m_state.oitLayers;
      aBufferStrideBytes                       = coverageShading ? sizeof(glm::uvec4) : sizeof(glm::uvec2);
      aBufferFormat                            = coverageShading ? VK_FORMAT_R32G32B32A32_UINT : VK_FORMAT_R32G32_UINT;
      m_sceneUbo.linkedListAllocatedPerElement = m_state.oitLayers;
      break;
    case OIT_INTERLOCK:
    case OIT_SPINLOCK:
      allocAux                                 = true;
      allocAuxSpin                             = (m_state.algorithm == OIT_SPINLOCK);
      allocAuxDepth                            = true;
      aBufferElementsPerSample                 = m_state.oitLayers;
      aBufferStrideBytes                       = coverageShading ? sizeof(glm::uvec4) : sizeof(glm::uvec2);
      aBufferFormat                            = coverageShading ? VK_FORMAT_R32G32B32A32_UINT : VK_FORMAT_R32G32_UINT;
      m_sceneUbo.linkedListAllocatedPerElement = m_state.oitLayers;
      break;
    case OIT_LINKEDLIST:
      allocAux                                 = true;
      allocCounter                             = true;
      aBufferElementsPerSample                 = m_state.linkedListAllocatedPerElement;
      aBufferStrideBytes                       = sizeof(glm::uvec4);
      aBufferFormat                            = VK_FORMAT_R32G32B32A32_UINT;
      m_sceneUbo.linkedListAllocatedPerElement = m_state.linkedListAllocatedPerElement * bufferWidth * bufferHeight;
      break;
    case OIT_LOOP:
      allocAux                                 = true;
      aBufferElementsPerSample                 = static_cast<VkDeviceSize>(m_state.oitLayers) * 2;
      aBufferStrideBytes                       = sizeof(uint32_t);
      aBufferFormat                            = VK_FORMAT_R32_UINT;
      m_sceneUbo.linkedListAllocatedPerElement = m_state.oitLayers;
      break;
    case OIT_LOOP64:
      allocAux                                 = true;
      aBufferElementsPerSample                 = m_state.oitLayers;
      aBufferStrideBytes                       = sizeof(uint64_t);
      aBufferFormat                            = VK_FORMAT_R32G32_UINT;
      m_sceneUbo.linkedListAllocatedPerElement = m_state.oitLayers;
      break;
    case OIT_WEIGHTED:
      // Don't create anything other than the special textures below
      break;
    default:
      assert(!"createABuffers: Textures for algorithm not implemented!");
  }

  if(sampleShading)
  {
    aBufferElementsPerSample *= m_state.msaa;
    m_sceneUbo.linkedListAllocatedPerElement *= m_state.msaa;
  }

  // Reference: https://www.lei.chat/posts/hlsl-for-vulkan-resources/
  const VkDeviceSize aBufferSize = static_cast<VkDeviceSize>(bufferWidth) * static_cast<VkDeviceSize>(bufferHeight)
                                   * aBufferElementsPerSample * aBufferStrideBytes;
  if(aBufferSize != 0)
  {
    const VkBufferUsageFlagBits aBufferUsage =
        (m_state.algorithm == OIT_LOOP64 ? VK_BUFFER_USAGE_STORAGE_BUFFER_BIT : VK_BUFFER_USAGE_STORAGE_TEXEL_BUFFER_BIT);
    m_oitABuffer.init(device, m_allocator, aBufferSize, aBufferUsage, aBufferFormat);
    NVVK_DBG_NAME(m_oitABuffer.buffer.buffer);
  }

  // Auxiliary images
  // The ways that auxiliary images can be used
  const VkImageUsageFlags auxUsages = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
  // if `sampleShading`, then each auxiliary image is actually a texture array:
  const uint32_t auxLayers = (sampleShading ? m_state.msaa : 1);

  if(allocAux)
  {
    m_oitAuxImage.init(device, m_allocator, VK_IMAGE_TYPE_2D, VK_IMAGE_ASPECT_COLOR_BIT, VK_FORMAT_R32_UINT,
                       bufferWidth, bufferHeight, auxLayers, auxUsages);
    NVVK_DBG_NAME(m_oitAuxImage.image.image);
    m_oitAuxImage.transitionTo(cmdBuffer, VK_IMAGE_LAYOUT_GENERAL);
  }

  if(allocAuxSpin)
  {
    m_oitAuxSpinImage.init(device, m_allocator, VK_IMAGE_TYPE_2D, VK_IMAGE_ASPECT_COLOR_BIT, VK_FORMAT_R32_UINT,
                           bufferWidth, bufferHeight, auxLayers, auxUsages);
    NVVK_DBG_NAME(m_oitAuxSpinImage.image.image);
    m_oitAuxSpinImage.transitionTo(cmdBuffer, VK_IMAGE_LAYOUT_GENERAL);
  }

  if(allocAuxDepth)
  {
    m_oitAuxDepthImage.init(device, m_allocator, VK_IMAGE_TYPE_2D, VK_IMAGE_ASPECT_COLOR_BIT, VK_FORMAT_R32_UINT,
                            bufferWidth, bufferHeight, auxLayers, auxUsages);
    NVVK_DBG_NAME(m_oitAuxDepthImage.image.image);
    m_oitAuxDepthImage.transitionTo(cmdBuffer, VK_IMAGE_LAYOUT_GENERAL);
  }

  if(allocCounter)
  {
    // Here, a counter is really a 1x1x1 image.
    m_oitCounterImage.init(device, m_allocator, VK_IMAGE_TYPE_2D, VK_IMAGE_ASPECT_COLOR_BIT, VK_FORMAT_R32_UINT, 1, 1, 1, auxUsages);
    NVVK_DBG_NAME(m_oitCounterImage.image.image);
    m_oitCounterImage.transitionTo(cmdBuffer, VK_IMAGE_LAYOUT_GENERAL);
  }

  if(m_state.algorithm == OIT_WEIGHTED)
  {
    // Weighted, Blended OIT's color and reveal textures will be used both as
    // color attachments and as storage images (i.e. accessed via imageLoad).
    // We'll handle their transitions inside of drawTransparentWeighted.
    const VkImageUsageFlags weightedUsages = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_INPUT_ATTACHMENT_BIT;
    m_oitWeightedColorImage.init(device, m_allocator, VK_IMAGE_TYPE_2D, VK_IMAGE_ASPECT_COLOR_BIT,
                                 m_oitWeightedColorFormat, bufferWidth, bufferHeight, 1, weightedUsages, m_state.msaa);
    NVVK_DBG_NAME(m_oitWeightedColorImage.image.image);
    m_oitWeightedRevealImage.init(device, m_allocator, VK_IMAGE_TYPE_2D, VK_IMAGE_ASPECT_COLOR_BIT,
                                  m_oitWeightedRevealFormat, bufferWidth, bufferHeight, 1, weightedUsages, m_state.msaa);
    NVVK_DBG_NAME(m_oitWeightedRevealImage.image.image);
    // Transition both of them to color attachments, which is the way they'll first be used:
    // (see m_renderPassWeighted for reference)
    m_oitWeightedColorImage.transitionTo(cmdBuffer, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
    m_oitWeightedRevealImage.transitionTo(cmdBuffer, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
  }
}

void Sample::destroyDescriptorSets()
{
  if(m_pipelineLayout)
  {
    vkDestroyPipelineLayout(m_app->getDevice(), m_pipelineLayout, nullptr);
    m_pipelineLayout = VK_NULL_HANDLE;
  }
  m_descriptorPack.deinit();
}

void Sample::createDescriptorSets()
{
  destroyDescriptorSets();
  // A descriptor is in some sense a pointer to a resource on the GPU.
  // Descriptor sets are sets of descriptors - the application sets many descriptors
  // at once, instead of setting them all individually.
  // Descriptor sets, in turn, are allocated from a descriptor pool.
  // Vulkan pipelines need to know what sorts of resources they will access.
  // Since a pipeline operates on descriptor sets with different contents,
  // we use a descriptor set layout to construct a Vulkan pipeline layout.

  // We'll first specify the layout - in a reflectable way that we can use
  // later on as well. Then we'll create a descriptor pool, allocate
  // descriptor sets from that, and finally create a pipeline layout.
  nvvk::DescriptorBindings& bindings = m_descriptorPack.bindings;

  // Descriptors get assigned to a triplet (descriptor set index,
  // binding index, array index). So we have to let the descriptor
  // set container know that the size of the array of each of these is 1.
  bindings.addBinding(UBO_SCENE, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT);
  // OIT_LOOP64 uses a storage buffer A-buffer, while all other algorithms use a storage texel buffer A-buffer.
  if(m_state.algorithm == OIT_LOOP64)
  {
    bindings.addBinding(IMG_ABUFFER, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_FRAGMENT_BIT);
  }
  else
  {
    bindings.addBinding(IMG_ABUFFER, VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER, 1, VK_SHADER_STAGE_FRAGMENT_BIT);
  }
  bindings.addBinding(IMG_AUX, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_FRAGMENT_BIT);
  bindings.addBinding(IMG_AUXSPIN, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_FRAGMENT_BIT);
  bindings.addBinding(IMG_AUXDEPTH, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_FRAGMENT_BIT);
  bindings.addBinding(IMG_COUNTER, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_FRAGMENT_BIT);
  // For more information about the Weighted, Blended Order-Independent Transparency configuration,
  // see how the render pass is created.
  bindings.addBinding(IMG_WEIGHTED_COLOR, VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT, 1, VK_SHADER_STAGE_FRAGMENT_BIT);
  bindings.addBinding(IMG_WEIGHTED_REVEAL, VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT, 1, VK_SHADER_STAGE_FRAGMENT_BIT);

  m_descriptorPack.initFromBindings(m_app->getDevice(), m_app->getFrameCycleSize());

// Set the descriptor sets' debug names.
#ifdef _DEBUG
  for(size_t i = 0; i < m_descriptorPack.sets.size(); i++)
  {
    nvvk::DebugUtil::getInstance().setObjectName(m_descriptorPack.sets[i], "Descriptor Set " + std::to_string(i));
  }
#endif

  // Create the pipeline layout. This application doesn't use any push constants,
  // so the function is relatively simple.
  VkPipelineLayoutCreateInfo pipelineInfo{.sType          = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
                                          .setLayoutCount = 1,
                                          .pSetLayouts    = &m_descriptorPack.layout};
  NVVK_CHECK(vkCreatePipelineLayout(m_app->getDevice(), &pipelineInfo, nullptr, &m_pipelineLayout));
}

void Sample::updateAllDescriptorSets()
{
  // We create one descriptor set per swapchain image.
  const uint32_t totalDescriptorSets = m_app->getFrameCycleSize();

  // Information about the buffer and image descriptors we'll use.
  // When constructing VkWriteDescriptorSet objects, we'll take references
  // to these.

  // UBO_SCENE
  std::vector<VkDescriptorBufferInfo> uboBufferInfo;
  uboBufferInfo.resize(totalDescriptorSets);
  for(uint32_t ring = 0; ring < totalDescriptorSets; ring++)
  {
    uboBufferInfo[ring] =
        VkDescriptorBufferInfo{.buffer = m_uniformBuffers[ring].buffer, .offset = 0, .range = sizeof(shaderio::SceneData)};
  }

  // Auxiliary images (note that their image views may be nullptr - this is fixed later):
  const VkDescriptorImageInfo oitAuxInfo{
      .sampler     = m_pointSampler,
      .imageView   = m_oitAuxImage.getView(),
      .imageLayout = VK_IMAGE_LAYOUT_GENERAL  // For read and write in shader
  };

  VkDescriptorImageInfo oitAuxSpinInfo = oitAuxInfo;
  oitAuxSpinInfo.imageView             = m_oitAuxSpinImage.getView();

  VkDescriptorImageInfo oitAuxDepthInfo = oitAuxInfo;
  oitAuxDepthInfo.imageView             = m_oitAuxDepthImage.getView();

  VkDescriptorImageInfo oitCounterInfo = oitAuxInfo;
  oitCounterInfo.imageView             = m_oitCounterImage.getView();

  const VkDescriptorImageInfo oitWeightedColorInfo = {.sampler     = VK_NULL_HANDLE,
                                                      .imageView   = m_oitWeightedColorImage.getView(),
                                                      .imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};

  VkDescriptorImageInfo oitWeightedRevealInfo = oitWeightedColorInfo;
  oitWeightedRevealInfo.imageView             = m_oitWeightedRevealImage.getView();

  // IMG_ABUFFER (when used as a storage buffer instead of a storage texel buffer)
  const VkDescriptorBufferInfo oitABufferInfo{.buffer = m_oitABuffer.buffer.buffer, .offset = 0, .range = VK_WHOLE_SIZE};

  // Descriptor sets without the color buffer bound to the shader stage
  nvvk::WriteSetContainer   updates;
  nvvk::DescriptorBindings& bindings = m_descriptorPack.bindings;
  for(uint32_t ring = 0; ring < totalDescriptorSets; ring++)
  {
    updates.append(bindings.getWriteSet(UBO_SCENE, m_descriptorPack.sets[ring]), &uboBufferInfo[ring]);

    if(m_state.algorithm == OIT_LOOP64)
    {
      // IMG_ABUFFER is a storage buffer
      updates.append(bindings.getWriteSet(IMG_ABUFFER, m_descriptorPack.sets[ring]), &oitABufferInfo);
    }
    else
    {
      // IMG_ABUFFER is a storage texel buffer (which is a kind of buffer in
      // Vulkan, but a kind of texture in OpenGL).
      if(m_oitABuffer.view != VK_NULL_HANDLE)
      {
        updates.append(bindings.getWriteSet(IMG_ABUFFER, m_descriptorPack.sets[ring]), &m_oitABuffer.view);
      }
    }

    if(oitAuxInfo.imageView != VK_NULL_HANDLE)
    {
      updates.append(bindings.getWriteSet(IMG_AUX, m_descriptorPack.sets[ring]), &oitAuxInfo);
    }

    if(oitAuxSpinInfo.imageView != VK_NULL_HANDLE)
    {
      updates.append(bindings.getWriteSet(IMG_AUXSPIN, m_descriptorPack.sets[ring]), &oitAuxSpinInfo);
    }

    if(oitAuxDepthInfo.imageView != VK_NULL_HANDLE)
    {
      updates.append(bindings.getWriteSet(IMG_AUXDEPTH, m_descriptorPack.sets[ring]), &oitAuxDepthInfo);
    }

    if(oitCounterInfo.imageView != VK_NULL_HANDLE)
    {
      updates.append(bindings.getWriteSet(IMG_COUNTER, m_descriptorPack.sets[ring]), &oitCounterInfo);
    }

    if(oitWeightedColorInfo.imageView != VK_NULL_HANDLE)
    {
      updates.append(bindings.getWriteSet(IMG_WEIGHTED_COLOR, m_descriptorPack.sets[ring]), &oitWeightedColorInfo);
    }

    if(oitWeightedRevealInfo.imageView != VK_NULL_HANDLE)
    {
      updates.append(bindings.getWriteSet(IMG_WEIGHTED_REVEAL, m_descriptorPack.sets[ring]), &oitWeightedRevealInfo);
    }
  }

  // Now go ahead and update the descriptor sets!
  vkUpdateDescriptorSets(m_app->getDevice(), static_cast<uint32_t>(updates.size()), updates.data(), 0, nullptr);
}

void Sample::destroyRenderPasses()
{
  if(m_renderPassColorDepthClear != VK_NULL_HANDLE)
  {
    vkDestroyRenderPass(m_app->getDevice(), m_renderPassColorDepthClear, NULL);
    m_renderPassColorDepthClear = VK_NULL_HANDLE;
  }

  if(m_renderPassWeighted != VK_NULL_HANDLE)
  {
    vkDestroyRenderPass(m_app->getDevice(), m_renderPassWeighted, NULL);
    m_renderPassWeighted = VK_NULL_HANDLE;
  }
}

void Sample::createRenderPasses()
{
  destroyRenderPasses();

  // m_renderPassColorDepthClear
  // Render pass for rendering to m_colorImage and m_depthImage, clearing them
  // beforehand. Both are in VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL.
  // We create this manually since (as of this writing) nvvk::createRenderPass
  // doesn't support multisampling.
  {
    std::array<VkAttachmentDescription, 2> attachments = {};  // Color attachment, depth attachment
    // Color attachment
    attachments[0] = VkAttachmentDescription{
        .format         = m_colorImage.getFormat(),
        .samples        = static_cast<VkSampleCountFlagBits>(m_state.msaa),
        .loadOp         = VK_ATTACHMENT_LOAD_OP_CLEAR,
        .storeOp        = VK_ATTACHMENT_STORE_OP_STORE,
        .stencilLoadOp  = VK_ATTACHMENT_LOAD_OP_DONT_CARE,
        .stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE,
        .initialLayout  = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
        .finalLayout    = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
    };

    // Color attachment reference
    const VkAttachmentReference colorAttachmentRef{.attachment = 0, .layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL};

    // Depth attachment
    attachments[1]               = attachments[0];
    attachments[1].format        = m_depthImage.getFormat();
    attachments[1].initialLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
    attachments[1].finalLayout   = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

    // Depth attachment reference
    const VkAttachmentReference depthAttachmentRef{.attachment = 1, .layout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL};

    // 1 subpass
    const VkSubpassDescription subpass{.pipelineBindPoint       = VK_PIPELINE_BIND_POINT_GRAPHICS,
                                       .colorAttachmentCount    = 1,
                                       .pColorAttachments       = &colorAttachmentRef,
                                       .pDepthStencilAttachment = &depthAttachmentRef};

    // We only need to specify one dependency: Since the subpass has a barrier, the subpass will
    // need a self-dependency. (There are implicit external dependencies that are automatically added.)
    const VkSubpassDependency selfDependency{
        .srcSubpass      = 0,
        .dstSubpass      = 0,
        .srcStageMask    = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
        .dstStageMask    = selfDependency.srcStageMask,
        .srcAccessMask   = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT,
        .dstAccessMask   = selfDependency.srcAccessMask,
        .dependencyFlags = VK_DEPENDENCY_BY_REGION_BIT  // Required, since we use framebuffer-space stages
    };

    // No dependency on external data
    const VkRenderPassCreateInfo rpInfo{
        .sType           = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO,
        .attachmentCount = static_cast<uint32_t>(attachments.size()),
        .pAttachments    = attachments.data(),
        .subpassCount    = 1,
        .pSubpasses      = &subpass,
        .dependencyCount = 1,
        .pDependencies   = &selfDependency,
    };

    NVVK_CHECK(vkCreateRenderPass(m_app->getDevice(), &rpInfo, NULL, &m_renderPassColorDepthClear));
    NVVK_DBG_NAME(m_renderPassColorDepthClear);
  }

  // m_renderPassWeighted
  // This render pass is used for Weighted, Blended Order-Independent
  // Transparency. It's somewhat tricky, and has two subpasses, with three
  // total attachments (weighted color, weighted reveal, color).
  // The first two attachments are cleared, and the three attachments
  // are all initially laid out for color attachments.
  // Subpass 0 takes attachments 0 and 1, and draws to them.
  // Then subpass 1 takes attachments 0 and 1 as inputs in the
  // VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL layout and attachment 2 as an
  // output attachment, and performs the WBOIT resolve step.
  // See https://www.saschawillems.de/blog/2018/07/19/vulkan-input-attachments-and-sub-passes/
  // for an example of a different type.
  {
    // Describe the attachments at the beginning and end of the render pass.
    const VkAttachmentDescription weightedColorAttachment{.format  = m_oitWeightedColorFormat,
                                                          .samples = static_cast<VkSampleCountFlagBits>(m_state.msaa),
                                                          .loadOp  = VK_ATTACHMENT_LOAD_OP_CLEAR,
                                                          .storeOp = VK_ATTACHMENT_STORE_OP_STORE,
                                                          .stencilLoadOp  = VK_ATTACHMENT_LOAD_OP_DONT_CARE,
                                                          .stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE,
                                                          .initialLayout  = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                                                          .finalLayout    = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL};

    VkAttachmentDescription weightedRevealAttachment = weightedColorAttachment;
    weightedRevealAttachment.format                  = m_oitWeightedRevealFormat;

    VkAttachmentDescription colorAttachment = weightedColorAttachment;
    colorAttachment.format                  = m_colorImage.getFormat();
    colorAttachment.loadOp                  = VK_ATTACHMENT_LOAD_OP_LOAD;

    VkAttachmentDescription depthAttachment = colorAttachment;
    depthAttachment.format                  = m_depthImage.getFormat();
    depthAttachment.initialLayout           = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
    depthAttachment.finalLayout             = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

    const std::array<VkAttachmentDescription, 4> allAttachments = {weightedColorAttachment, weightedRevealAttachment,
                                                                   colorAttachment, depthAttachment};

    std::array<VkSubpassDescription, 2> subpasses{};

    // Subpass 0 - weighted textures & depth texture for testing
    std::array<VkAttachmentReference, 2> subpass0ColorAttachments{};
    subpass0ColorAttachments[0] = VkAttachmentReference{.attachment = 0, .layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL};
    subpass0ColorAttachments[1] = VkAttachmentReference{.attachment = 1, .layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL};

    // 3 is m_depthImage
    const VkAttachmentReference depthAttachmentRef{.attachment = 3, .layout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL};

    subpasses[0] = VkSubpassDescription{
        .pipelineBindPoint       = VK_PIPELINE_BIND_POINT_GRAPHICS,
        .colorAttachmentCount    = static_cast<uint32_t>(subpass0ColorAttachments.size()),
        .pColorAttachments       = subpass0ColorAttachments.data(),
        .pDepthStencilAttachment = &depthAttachmentRef,
    };

    // Subpass 1
    // Attachment 2 is m_colorImage
    const VkAttachmentReference subpass1ColorAttachment{.attachment = 2, .layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL};

    std::array<VkAttachmentReference, 2> subpass1InputAttachments{};
    subpass1InputAttachments[0] = VkAttachmentReference{.attachment = 0, .layout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
    subpass1InputAttachments[1] = VkAttachmentReference{.attachment = 1, .layout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};

    subpasses[1] = VkSubpassDescription{
        .pipelineBindPoint    = VK_PIPELINE_BIND_POINT_GRAPHICS,
        .inputAttachmentCount = static_cast<uint32_t>(subpass1InputAttachments.size()),
        .pInputAttachments    = subpass1InputAttachments.data(),
        .colorAttachmentCount = 1,
        .pColorAttachments    = &subpass1ColorAttachment,
    };

    // Dependencies
    std::array<VkSubpassDependency, 3> subpassDependencies{};
    subpassDependencies[0] = VkSubpassDependency{
        .srcSubpass    = VK_SUBPASS_EXTERNAL,
        .dstSubpass    = 0,
        .srcStageMask  = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
        .dstStageMask  = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
        .srcAccessMask = 0,
        .dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
    };
    subpassDependencies[1] = VkSubpassDependency{
        .srcSubpass    = 0,
        .dstSubpass    = 1,
        .srcStageMask  = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
        .dstStageMask  = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
        .srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
        .dstAccessMask = VK_ACCESS_SHADER_READ_BIT,
    };
    // Finally, we have a dependency at the end to allow the images to transition back to VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL
    subpassDependencies[2] = VkSubpassDependency{
        .srcSubpass    = 1,
        .dstSubpass    = VK_SUBPASS_EXTERNAL,
        .srcStageMask  = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
        .dstStageMask  = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
        .srcAccessMask = VK_ACCESS_SHADER_READ_BIT,
        .dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
    };

    // Finally, create the render pass
    const VkRenderPassCreateInfo renderPassInfo{
        .sType           = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO,
        .attachmentCount = static_cast<uint32_t>(allAttachments.size()),
        .pAttachments    = allAttachments.data(),
        .subpassCount    = static_cast<uint32_t>(subpasses.size()),
        .pSubpasses      = subpasses.data(),
        .dependencyCount = static_cast<uint32_t>(subpassDependencies.size()),
        .pDependencies   = subpassDependencies.data(),
    };
    NVVK_CHECK(vkCreateRenderPass(m_app->getDevice(), &renderPassInfo, nullptr, &m_renderPassWeighted));
    NVVK_DBG_NAME(m_renderPassWeighted);
  }
}

void Sample::destroyShaderModules()
{
  m_shaderCompiler.clear(m_app->getDevice());
}

void Sample::createOrReloadShaderModules()
{
  const CompileDefines defines = {
      {"OIT_LAYERS", std::to_string(m_state.oitLayers)},
      {"OIT_TAILBLEND", m_state.tailBlend ? "1" : "0"},
      {"OIT_INTERLOCK_IS_ORDERED", m_state.interlockIsOrdered ? "1" : "0"},
      {"OIT_MSAA", std::to_string(m_state.msaa)},
      {"OIT_SAMPLE_SHADING", m_state.sampleShading ? "1" : "0"},
  };

  // You can set this to true to make sure that all of the shaders
  // compile correctly.
  const bool     loadEverything = false;
  CompileDefines defineDepth    = defines;
  defineDepth.push_back({"PASS", "PASS_DEPTH"});
  CompileDefines defineColor = defines;
  defineColor.push_back({"PASS", "PASS_COLOR"});
  CompileDefines defineComposite = defines;
  defineComposite.push_back({"PASS", "PASS_COMPOSITE"});

  // Compile shaders

  VkDevice device = m_app->getDevice();

  // Scene (standard mesh rendering) and full-screen triangle vertex shaders
  m_vertexShaders[+VertexShaderIndex::eScene] =
      m_shaderCompiler.compile(device, {shaderc_vertex_shader, "object.vert.glsl", defines});
  m_vertexShaders[+VertexShaderIndex::eFullScreenTriangle] =
      m_shaderCompiler.compile(device, {shaderc_vertex_shader, "fullScreenTriangle.vert.glsl", defines});

  // Opaque pass
  m_fragmentShaders[+PassIndex::eOpaque] =
      m_shaderCompiler.compile(device, {shaderc_fragment_shader, "opaque.frag.glsl", defines});

  if((m_state.algorithm == OIT_SIMPLE) || loadEverything)
  {
    const std::filesystem::path file = "oitSimple.frag.glsl";
    m_fragmentShaders[+PassIndex::eSimpleColor] = m_shaderCompiler.compile(device, {shaderc_fragment_shader, file, defineColor});
    m_fragmentShaders[+PassIndex::eSimpleComposite] =
        m_shaderCompiler.compile(device, {shaderc_fragment_shader, file, defineComposite});
  }
  if((m_state.algorithm == OIT_LINKEDLIST) || loadEverything)
  {
    const std::filesystem::path file = "oitLinkedList.frag.glsl";
    m_fragmentShaders[+PassIndex::eLinkedListColor] =
        m_shaderCompiler.compile(device, {shaderc_fragment_shader, file, defineColor});
    m_fragmentShaders[+PassIndex::eLinkedListComposite] =
        m_shaderCompiler.compile(device, {shaderc_fragment_shader, file, defineComposite});
  }
  if((m_state.algorithm == OIT_LOOP) || loadEverything)
  {
    const std::string file = "oitLoop.frag.glsl";
    m_fragmentShaders[+PassIndex::eLoopDepth] = m_shaderCompiler.compile(device, {shaderc_fragment_shader, file, defineDepth});
    m_fragmentShaders[+PassIndex::eLoopColor] = m_shaderCompiler.compile(device, {shaderc_fragment_shader, file, defineColor});
    m_fragmentShaders[+PassIndex::eLoopComposite] =
        m_shaderCompiler.compile(device, {shaderc_fragment_shader, file, defineComposite});
  }
  if((m_state.algorithm == OIT_LOOP64) || loadEverything)
  {
    const std::string file = "oitLoop64.frag.glsl";
    m_fragmentShaders[+PassIndex::eLoop64Color] = m_shaderCompiler.compile(device, {shaderc_fragment_shader, file, defineColor});
    m_fragmentShaders[+PassIndex::eLoop64Composite] =
        m_shaderCompiler.compile(device, {shaderc_fragment_shader, file, defineComposite});
  }
  if((m_state.algorithm == OIT_INTERLOCK) || loadEverything)
  {
    assert(m_ctx->hasExtensionEnabled(VK_EXT_FRAGMENT_SHADER_INTERLOCK_EXTENSION_NAME));
    const std::string file = "oitInterlock.frag.glsl";
    m_fragmentShaders[+PassIndex::eInterlockColor] =
        m_shaderCompiler.compile(device, {shaderc_fragment_shader, file, defineColor});
    m_fragmentShaders[+PassIndex::eInterlockComposite] =
        m_shaderCompiler.compile(device, {shaderc_fragment_shader, file, defineComposite});
  }
  if((m_state.algorithm == OIT_SPINLOCK) || loadEverything)
  {
    const std::string file = "oitSpinlock.frag.glsl";
    m_fragmentShaders[+PassIndex::eSpinlockColor] =
        m_shaderCompiler.compile(device, {shaderc_fragment_shader, file, defineColor});
    m_fragmentShaders[+PassIndex::eSpinlockComposite] =
        m_shaderCompiler.compile(device, {shaderc_fragment_shader, file, defineComposite});
  }
  if((m_state.algorithm == OIT_WEIGHTED) || loadEverything)
  {
    const std::string file = "oitWeighted.frag.glsl";
    m_fragmentShaders[+PassIndex::eWeightedColor] =
        m_shaderCompiler.compile(device, {shaderc_fragment_shader, file, defineColor});
    m_fragmentShaders[+PassIndex::eWeightedComposite] =
        m_shaderCompiler.compile(device, {shaderc_fragment_shader, file, defineComposite});
  }
}

void Sample::destroyGraphicsPipelines()
{
  for(size_t i = 0; i < m_pipelines.size(); i++)
  {
    VkPipeline& pipeline = m_pipelines[i];
    if(pipeline != VK_NULL_HANDLE)
    {
      vkDestroyPipeline(m_app->getDevice(), pipeline, nullptr);
      pipeline = VK_NULL_HANDLE;
    }
  }
}

void Sample::createGraphicsPipelines()
{
  destroyGraphicsPipelines();

  // We always need the opaque pipeline:
  m_pipelines[+PassIndex::eOpaque] = createGraphicsPipeline("Opaque", m_vertexShaders[+VertexShaderIndex::eScene],
                                                            m_fragmentShaders[+PassIndex::eOpaque], BlendMode::NONE,
                                                            true, false, m_renderPassColorDepthClear);

  const bool transparentDoubleSided = true;  // Iff transparent objects are double-sided

  // Switch off between algorithms:
  switch(m_state.algorithm)
  {
    case OIT_SIMPLE:
      m_pipelines[+PassIndex::eSimpleColor] =
          createGraphicsPipeline("SimpleColor", m_vertexShaders[+VertexShaderIndex::eScene],
                                 m_fragmentShaders[+PassIndex::eSimpleColor], BlendMode::PREMULTIPLIED, true,
                                 transparentDoubleSided, m_renderPassColorDepthClear);
      m_pipelines[+PassIndex::eSimpleComposite] =
          createGraphicsPipeline("SimpleComposite", m_vertexShaders[+VertexShaderIndex::eFullScreenTriangle],
                                 m_fragmentShaders[+PassIndex::eSimpleComposite], BlendMode::PREMULTIPLIED, false,
                                 transparentDoubleSided, m_renderPassColorDepthClear);
      break;
    case OIT_LINKEDLIST:
      m_pipelines[+PassIndex::eLinkedListColor] =
          createGraphicsPipeline("LinkedListColor", m_vertexShaders[+VertexShaderIndex::eScene],
                                 m_fragmentShaders[+PassIndex::eLinkedListColor], BlendMode::PREMULTIPLIED, true,
                                 transparentDoubleSided, m_renderPassColorDepthClear);
      m_pipelines[+PassIndex::eLinkedListComposite] =
          createGraphicsPipeline("LinkedListComposite", m_vertexShaders[+VertexShaderIndex::eFullScreenTriangle],
                                 m_fragmentShaders[+PassIndex::eLinkedListComposite], BlendMode::PREMULTIPLIED, false,
                                 transparentDoubleSided, m_renderPassColorDepthClear);
      break;
    case OIT_LOOP:
      m_pipelines[+PassIndex::eLoopDepth] =
          createGraphicsPipeline("LoopDepth", m_vertexShaders[+VertexShaderIndex::eScene], m_fragmentShaders[+PassIndex::eLoopDepth],
                                 BlendMode::PREMULTIPLIED, true, transparentDoubleSided, m_renderPassColorDepthClear);
      m_pipelines[+PassIndex::eLoopColor] =
          createGraphicsPipeline("LoopColor", m_vertexShaders[+VertexShaderIndex::eScene], m_fragmentShaders[+PassIndex::eLoopColor],
                                 BlendMode::PREMULTIPLIED, true, transparentDoubleSided, m_renderPassColorDepthClear);
      m_pipelines[+PassIndex::eLoopComposite] =
          createGraphicsPipeline("LoopComposite", m_vertexShaders[+VertexShaderIndex::eFullScreenTriangle],
                                 m_fragmentShaders[+PassIndex::eLoopComposite], BlendMode::PREMULTIPLIED, false,
                                 transparentDoubleSided, m_renderPassColorDepthClear);
      break;
    case OIT_LOOP64:
      m_pipelines[+PassIndex::eLoop64Color] =
          createGraphicsPipeline("Loop64Color", m_vertexShaders[+VertexShaderIndex::eScene],
                                 m_fragmentShaders[+PassIndex::eLoop64Color], BlendMode::PREMULTIPLIED, true,
                                 transparentDoubleSided, m_renderPassColorDepthClear);
      m_pipelines[+PassIndex::eLoop64Composite] =
          createGraphicsPipeline("Loop64Composite", m_vertexShaders[+VertexShaderIndex::eFullScreenTriangle],
                                 m_fragmentShaders[+PassIndex::eLoop64Composite], BlendMode::PREMULTIPLIED, false,
                                 transparentDoubleSided, m_renderPassColorDepthClear);
      break;
    case OIT_INTERLOCK:
      m_pipelines[+PassIndex::eInterlockColor] =
          createGraphicsPipeline("InterlockColor", m_vertexShaders[+VertexShaderIndex::eScene],
                                 m_fragmentShaders[+PassIndex::eInterlockColor], BlendMode::PREMULTIPLIED, true,
                                 transparentDoubleSided, m_renderPassColorDepthClear);
      m_pipelines[+PassIndex::eInterlockComposite] =
          createGraphicsPipeline("InterlockComposite", m_vertexShaders[+VertexShaderIndex::eFullScreenTriangle],
                                 m_fragmentShaders[+PassIndex::eInterlockComposite], BlendMode::PREMULTIPLIED, false,
                                 transparentDoubleSided, m_renderPassColorDepthClear);
      break;
    case OIT_SPINLOCK:
      m_pipelines[+PassIndex::eSpinlockColor] =
          createGraphicsPipeline("InterlockColor", m_vertexShaders[+VertexShaderIndex::eScene],
                                 m_fragmentShaders[+PassIndex::eSpinlockColor], BlendMode::PREMULTIPLIED, true,
                                 transparentDoubleSided, m_renderPassColorDepthClear);
      m_pipelines[+PassIndex::eSpinlockComposite] =
          createGraphicsPipeline("InterlockComposite", m_vertexShaders[+VertexShaderIndex::eFullScreenTriangle],
                                 m_fragmentShaders[+PassIndex::eSpinlockComposite], BlendMode::PREMULTIPLIED, false,
                                 transparentDoubleSided, m_renderPassColorDepthClear);
      break;
    case OIT_WEIGHTED:
      m_pipelines[+PassIndex::eWeightedColor] =
          createGraphicsPipeline("WeightedColor", m_vertexShaders[+VertexShaderIndex::eScene],
                                 m_fragmentShaders[+PassIndex::eWeightedColor], BlendMode::WEIGHTED_COLOR, true,
                                 transparentDoubleSided, m_renderPassWeighted, 0);
      m_pipelines[+PassIndex::eWeightedComposite] =
          createGraphicsPipeline("WeightedComposite", m_vertexShaders[+VertexShaderIndex::eFullScreenTriangle],
                                 m_fragmentShaders[+PassIndex::eWeightedComposite], BlendMode::WEIGHTED_COMPOSITE,
                                 false, transparentDoubleSided, m_renderPassWeighted, 1);
      break;
  }
}
