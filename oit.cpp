/*
 * Copyright (c) 2020-2021, NVIDIA CORPORATION.  All rights reserved.
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
 * SPDX-FileCopyrightText: Copyright (c) 2020-2021 NVIDIA CORPORATION
 * SPDX-License-Identifier: Apache-2.0
 */


// This file contains implementations of the resource creation functions of
// Sample that are specific to order-independent transparency (for instance,
// render passes, descriptor sets, and A-buffers, but not swapchain creation.)

#include "oit.h"
#include <nvvk/error_vk.hpp>
#include <nvvk/renderpasses_vk.hpp>

void Sample::destroyFrameImages()
{
  m_colorImage.destroy(m_context, m_allocatorDma);
  m_depthImage.destroy(m_context, m_allocatorDma);
  m_oitABuffer.destroy(m_context, m_allocatorDma);
  m_oitAuxImage.destroy(m_context, m_allocatorDma);
  m_oitAuxSpinImage.destroy(m_context, m_allocatorDma);
  m_oitAuxDepthImage.destroy(m_context, m_allocatorDma);
  m_oitCounterImage.destroy(m_context, m_allocatorDma);
  m_oitWeightedColorImage.destroy(m_context, m_allocatorDma);
  m_oitWeightedRevealImage.destroy(m_context, m_allocatorDma);
  m_downsampleImage.destroy(m_context, m_allocatorDma);
  m_guiCompositeImage.destroy(m_context, m_allocatorDma);
}

void Sample::createFrameImages(VkCommandBuffer cmdBuffer)
{
  destroyFrameImages();

  const int swapchainWidth  = m_windowState.m_swapSize[0];
  const int swapchainHeight = m_windowState.m_swapSize[1];
  // We implement supersample anti-aliasing by rendering to a larger texture.
  const int bufferWidth  = swapchainWidth * m_state.supersample;
  const int bufferHeight = swapchainHeight * m_state.supersample;

  // Offscreen color and depth buffer
  {
    // Color image, created with an sRGB format.
    m_colorImage.create(m_context, m_allocatorDma, VK_IMAGE_TYPE_2D, VK_IMAGE_ASPECT_COLOR_BIT, VK_FORMAT_B8G8R8A8_SRGB, bufferWidth,
                        bufferHeight, 1, VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT, m_state.msaa);
    m_colorImage.setName(m_debug, "m_colorImage");
    // We'll put it into the layout for a color attachment later.

    // Depth image
    VkFormat depthFormat = nvvk::findDepthFormat(m_context.m_physicalDevice);

    m_depthImage.create(m_context, m_allocatorDma, VK_IMAGE_TYPE_2D, VK_IMAGE_ASPECT_DEPTH_BIT, depthFormat,
                        bufferWidth, bufferHeight, 1, VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT, m_state.msaa);
    m_depthImage.setName(m_debug, "m_depthImage");

    // Intermediate storage for resolve - 1spp, swapchain sized, with the same format as the color image.
    m_downsampleImage.create(m_context, m_allocatorDma, VK_IMAGE_TYPE_2D, VK_IMAGE_ASPECT_COLOR_BIT,
                             m_colorImage.c_format, swapchainWidth, swapchainHeight, 1,
                             VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT, 1);
    m_downsampleImage.setName(m_debug, "m_downsampleTargetImage");

    // Intermediate storage for rendering the GUI - 1spp, swapchain sized, with almost the same format as the swapchain
    // (with the exception that the channels have to be in the same order as m_colorImage)
    m_guiCompositeImage.create(m_context, m_allocatorDma, VK_IMAGE_TYPE_2D, VK_IMAGE_ASPECT_COLOR_BIT,
                               m_guiCompositeColorFormat, swapchainWidth, swapchainHeight, 1,
                               VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT,
                               1);
    m_guiCompositeImage.setName(m_debug, "m_guiCompositeImage");

    // Initial resource transitions
    m_colorImage.transitionTo(cmdBuffer, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT);
    m_depthImage.transitionTo(cmdBuffer, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL, VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT);
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
      aBufferStrideBytes                       = coverageShading ? sizeof(uvec4) : sizeof(uvec2);
      aBufferFormat                            = coverageShading ? VK_FORMAT_R32G32B32A32_UINT : VK_FORMAT_R32G32_UINT;
      m_sceneUbo.linkedListAllocatedPerElement = m_state.oitLayers;
      break;
    case OIT_INTERLOCK:
    case OIT_SPINLOCK:
      allocAux                                 = true;
      allocAuxSpin                             = (m_state.algorithm == OIT_SPINLOCK);
      allocAuxDepth                            = true;
      aBufferElementsPerSample                 = m_state.oitLayers;
      aBufferStrideBytes                       = coverageShading ? sizeof(uvec4) : sizeof(uvec2);
      aBufferFormat                            = coverageShading ? VK_FORMAT_R32G32B32A32_UINT : VK_FORMAT_R32G32_UINT;
      m_sceneUbo.linkedListAllocatedPerElement = m_state.oitLayers;
      break;
    case OIT_LINKEDLIST:
      allocAux                                 = true;
      allocCounter                             = true;
      aBufferElementsPerSample                 = m_state.linkedListAllocatedPerElement;
      aBufferStrideBytes                       = sizeof(uvec4);
      aBufferFormat                            = VK_FORMAT_R32G32B32A32_UINT;
      m_sceneUbo.linkedListAllocatedPerElement = m_state.linkedListAllocatedPerElement * bufferWidth * bufferHeight;
      break;
    case OIT_LOOP:
      allocAux                                 = true;
      aBufferElementsPerSample                 = static_cast<VkDeviceSize>(m_state.oitLayers) * 2;
      aBufferStrideBytes                       = sizeof(uint);
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

  // Reference: https://antiagainst.github.io/post/hlsl-for-vulkan-resources/
  const VkDeviceSize aBufferSize = static_cast<VkDeviceSize>(bufferWidth) * static_cast<VkDeviceSize>(bufferHeight)
                                   * aBufferElementsPerSample * aBufferStrideBytes;
  if(aBufferSize != 0)
  {
    const VkBufferUsageFlagBits aBufferUsage =
        (m_state.algorithm == OIT_LOOP64 ? VK_BUFFER_USAGE_STORAGE_BUFFER_BIT : VK_BUFFER_USAGE_STORAGE_TEXEL_BUFFER_BIT);
    m_oitABuffer.create(m_context, m_allocatorDma, aBufferSize, aBufferUsage, aBufferFormat);
    m_oitABuffer.setName(m_debug, "m_oitABuffer");
  }

  // Auxiliary images
  // The ways that auxiliary images can be used
  const VkImageUsageFlags auxUsages = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
  // The ways that auxiliary images can be accessed
  const VkAccessFlags auxAccesses = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;
  // if `sampleShading`, then each auxiliary image is actually a texture array:
  const uint32_t auxLayers = (sampleShading ? m_state.msaa : 1);

  if(allocAux)
  {
    m_oitAuxImage.create(m_context, m_allocatorDma, VK_IMAGE_TYPE_2D, VK_IMAGE_ASPECT_COLOR_BIT, VK_FORMAT_R32_UINT,
                         bufferWidth, bufferHeight, auxLayers, auxUsages);
    m_oitAuxImage.setName(m_debug, "m_oitAuxImage");
    m_oitAuxImage.transitionTo(cmdBuffer, VK_IMAGE_LAYOUT_GENERAL, auxAccesses);
  }

  if(allocAuxSpin)
  {
    m_oitAuxSpinImage.create(m_context, m_allocatorDma, VK_IMAGE_TYPE_2D, VK_IMAGE_ASPECT_COLOR_BIT, VK_FORMAT_R32_UINT,
                             bufferWidth, bufferHeight, auxLayers, auxUsages);
    m_oitAuxSpinImage.setName(m_debug, "m_oitAuxSpinImage");
    m_oitAuxSpinImage.transitionTo(cmdBuffer, VK_IMAGE_LAYOUT_GENERAL, auxAccesses);
  }

  if(allocAuxDepth)
  {
    m_oitAuxDepthImage.create(m_context, m_allocatorDma, VK_IMAGE_TYPE_2D, VK_IMAGE_ASPECT_COLOR_BIT,
                              VK_FORMAT_R32_UINT, bufferWidth, bufferHeight, auxLayers, auxUsages);
    m_oitAuxDepthImage.setName(m_debug, "m_oitAuxDepthImage");
    m_oitAuxDepthImage.transitionTo(cmdBuffer, VK_IMAGE_LAYOUT_GENERAL, auxAccesses);
  }

  if(allocCounter)
  {
    // Here, a counter is really a 1x1x1 image.
    m_oitCounterImage.create(m_context, m_allocatorDma, VK_IMAGE_TYPE_2D, VK_IMAGE_ASPECT_COLOR_BIT, VK_FORMAT_R32_UINT,
                             1, 1, 1, auxUsages);
    m_oitCounterImage.setName(m_debug, "m_oitCounter");
    m_oitCounterImage.transitionTo(cmdBuffer, VK_IMAGE_LAYOUT_GENERAL, auxAccesses);
  }

  if(m_state.algorithm == OIT_WEIGHTED)
  {
    // Weighted, Blended OIT's color and reveal textures will be used both as
    // color attachments and as storage images (i.e. accessed via imageLoad).
    // We'll handle their transitions inside of drawTransparentWeighted.
    const VkImageUsageFlags weightedUsages = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_INPUT_ATTACHMENT_BIT;
    m_oitWeightedColorImage.create(m_context, m_allocatorDma, VK_IMAGE_TYPE_2D, VK_IMAGE_ASPECT_COLOR_BIT,
                                   m_oitWeightedColorFormat, bufferWidth, bufferHeight, 1, weightedUsages, m_state.msaa);
    m_oitWeightedRevealImage.create(m_context, m_allocatorDma, VK_IMAGE_TYPE_2D, VK_IMAGE_ASPECT_COLOR_BIT,
                                    m_oitWeightedRevealFormat, bufferWidth, bufferHeight, 1, weightedUsages, m_state.msaa);
    m_oitWeightedColorImage.setName(m_debug, "m_oitWeightedColorImage");
    m_oitWeightedRevealImage.setName(m_debug, "m_oitWeightedRevealImage");
    // Transition both of them to color attachments, which is the way they'll first be used:
    // (see m_renderPassWeighted for reference)
    m_oitWeightedColorImage.transitionTo(cmdBuffer, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT);
    m_oitWeightedRevealImage.transitionTo(cmdBuffer, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT);
  }
}

void Sample::destroyDescriptorSets()
{
  m_descriptorInfo.deinit();
  m_descriptorInfo.setBindings({});  // i.e. clear all bindings.
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

  // We'll use NVVK's helper functions to create these objects related to
  // descriptors in a relatively simple way.
  // We'll first specify the layout - in a reflectable way that we can use
  // later on as well. Then we'll create a descriptor pool, allocate
  // descriptor sets from that, and finally create a pipeline layout.

  m_descriptorInfo.init(m_context);

  // Descriptors get assigned to a triplet (descriptor set index,
  // binding index, array index). So we have to let the descriptor
  // set container know that the size of the array of each of these is 1.
  m_descriptorInfo.addBinding(UBO_SCENE, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT);
  // OIT_LOOP64 uses a storage buffer A-buffer, while all other algorithms use a storage texel buffer A-buffer.
  if(m_state.algorithm == OIT_LOOP64)
  {
    m_descriptorInfo.addBinding(IMG_ABUFFER, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_FRAGMENT_BIT);
  }
  else
  {
    m_descriptorInfo.addBinding(IMG_ABUFFER, VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER, 1, VK_SHADER_STAGE_FRAGMENT_BIT);
  }
  m_descriptorInfo.addBinding(IMG_AUX, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_FRAGMENT_BIT);
  m_descriptorInfo.addBinding(IMG_AUXSPIN, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_FRAGMENT_BIT);
  m_descriptorInfo.addBinding(IMG_AUXDEPTH, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_FRAGMENT_BIT);
  m_descriptorInfo.addBinding(IMG_COUNTER, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_FRAGMENT_BIT);
  // For more information about the Weighted, Blended Order-Independent Transparency configuration,
  // see how the render pass is created.
  m_descriptorInfo.addBinding(IMG_WEIGHTED_COLOR, VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT, 1, VK_SHADER_STAGE_FRAGMENT_BIT);
  m_descriptorInfo.addBinding(IMG_WEIGHTED_REVEAL, VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT, 1, VK_SHADER_STAGE_FRAGMENT_BIT);

  // We'll create one descriptor set per swapchain image.
  const uint32_t totalDescriptorSets = m_swapChain.getImageCount();

  // Create the layout
  m_descriptorInfo.initLayout();

  // Create a descriptor pool with space fot totalDescriptorSets descriptor sets,
  // and allocate the descriptor sets.
  m_descriptorInfo.initPool(totalDescriptorSets);

// Set the descriptor sets' debug names.
#ifdef _DEBUG
  for(uint32_t i = 0; i < m_descriptorInfo.getSetsCount(); i++)
  {
    const std::string generatedName = "Descriptor Set " + std::to_string(i);
    m_debug.setObjectName(m_descriptorInfo.getSet(i), generatedName.c_str());
  }
#endif

  // Create the pipeline layout. This application doesn't use any push constants,
  // so the function is relatively simple.
  m_descriptorInfo.initPipeLayout(0, nullptr, 0);
}

void Sample::updateAllDescriptorSets()
{
  std::vector<VkWriteDescriptorSet> updates;

  // We create one descriptor set per swapchain image.
  const uint32_t totalDescriptorSets = m_swapChain.getImageCount();

  // Information about the buffer and image descriptors we'll use.
  // When constructing VkWriteDescriptorSet objects, we'll take references
  // to these.

  // UBO_SCENE
  std::vector<VkDescriptorBufferInfo> uboBufferInfo;
  uboBufferInfo.resize(totalDescriptorSets);
  for(uint32_t ring = 0; ring < totalDescriptorSets; ring++)
  {
    uboBufferInfo[ring].buffer = m_uniformBuffers[ring].buffer;
    uboBufferInfo[ring].offset = 0;
    uboBufferInfo[ring].range  = sizeof(SceneData);
  }

  // Auxiliary images (note that their image views may be nullptr - this is fixed later):
  VkDescriptorImageInfo oitAuxInfo = {};
  oitAuxInfo.imageLayout           = VK_IMAGE_LAYOUT_GENERAL;  // For read and write in shader
  oitAuxInfo.imageView             = m_oitAuxImage.view;
  oitAuxInfo.sampler               = m_pointSampler;

  VkDescriptorImageInfo oitAuxSpinInfo = oitAuxInfo;
  oitAuxSpinInfo.imageView             = m_oitAuxSpinImage.view;

  VkDescriptorImageInfo oitAuxDepthInfo = oitAuxInfo;
  oitAuxDepthInfo.imageView             = m_oitAuxDepthImage.view;

  VkDescriptorImageInfo oitCounterInfo = oitAuxInfo;
  oitCounterInfo.imageView             = m_oitCounterImage.view;

  VkDescriptorImageInfo oitWeightedColorInfo = {};
  oitWeightedColorInfo.imageLayout           = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
  oitWeightedColorInfo.imageView             = m_oitWeightedColorImage.view;
  oitWeightedColorInfo.sampler               = VK_NULL_HANDLE;

  VkDescriptorImageInfo oitWeightedRevealInfo = oitWeightedColorInfo;
  oitWeightedRevealInfo.imageView             = m_oitWeightedRevealImage.view;

  // IMG_ABUFFER (when used as a storage buffer instead of a storage texel buffer)
  VkDescriptorBufferInfo oitABufferInfo = {};
  oitABufferInfo.buffer                 = m_oitABuffer.buffer.buffer;
  oitABufferInfo.offset                 = 0;
  oitABufferInfo.range                  = VK_WHOLE_SIZE;

  // Descriptor sets without the color buffer bound to the shader stage
  for(uint32_t ring = 0; ring < totalDescriptorSets; ring++)
  {
    updates.push_back(m_descriptorInfo.makeWrite(ring, UBO_SCENE, &uboBufferInfo[ring]));

    if(m_state.algorithm == OIT_LOOP64)
    {
      // IMG_ABUFFER is a storage buffer
      updates.push_back(m_descriptorInfo.makeWrite(ring, IMG_ABUFFER, &oitABufferInfo));
    }
    else
    {
      // IMG_ABUFFER is a storage texel buffer (which is a kind of buffer in
      // Vulkan, but a kind of texture in OpenGL).
      if(m_oitABuffer.view != nullptr)
      {
        updates.push_back(m_descriptorInfo.makeWrite(ring, IMG_ABUFFER, &m_oitABuffer.view));
      }
    }

    if(oitAuxInfo.imageView != nullptr)
    {
      updates.push_back(m_descriptorInfo.makeWrite(ring, IMG_AUX, &oitAuxInfo));
    }

    if(oitAuxSpinInfo.imageView != nullptr)
    {
      updates.push_back(m_descriptorInfo.makeWrite(ring, IMG_AUXSPIN, &oitAuxSpinInfo));
    }

    if(oitAuxDepthInfo.imageView != nullptr)
    {
      updates.push_back(m_descriptorInfo.makeWrite(ring, IMG_AUXDEPTH, &oitAuxDepthInfo));
    }

    if(oitCounterInfo.imageView != nullptr)
    {
      updates.push_back(m_descriptorInfo.makeWrite(ring, IMG_COUNTER, &oitCounterInfo));
    }

    if(oitWeightedColorInfo.imageView != nullptr)
    {
      updates.push_back(m_descriptorInfo.makeWrite(ring, IMG_WEIGHTED_COLOR, &oitWeightedColorInfo));
    }

    if(oitWeightedRevealInfo.imageView != nullptr)
    {
      updates.push_back(m_descriptorInfo.makeWrite(ring, IMG_WEIGHTED_REVEAL, &oitWeightedRevealInfo));
    }
  }

  // Now go ahead and update the descriptor sets!
  vkUpdateDescriptorSets(m_context, static_cast<uint32_t>(updates.size()), updates.data(), 0, nullptr);
}

void Sample::destroyGUIRenderPass()
{
  if(m_renderPassGUI != nullptr)
  {
    vkDestroyRenderPass(m_context, m_renderPassGUI, NULL);
    m_renderPassGUI = nullptr;
  }
}

void Sample::createGUIRenderPass() {
  // The render pass can't be changed once it's passed to Dear ImGui.
  assert(m_renderPassGUI == nullptr);

  // This is a bit tricky, and ties in to exactly how copyOffscreenToBackBuffer works.
  // It takes m_guiCompositeImage in layout TRANSFER_DST_OPTIMAL. Then it transitions it to
  // TRANSFER_COLOR_ATTACHMENT_OPTIMAL for rendering, and then transitions it to TRANSFER_SRC_OPTIMAL
  // for blitting to the swapchain.

  // Only one attachment
  VkAttachmentDescription attachment = {};
  attachment.format                  = m_guiCompositeColorFormat;
  attachment.samples                 = VK_SAMPLE_COUNT_1_BIT;
  attachment.loadOp                  = VK_ATTACHMENT_LOAD_OP_LOAD;
  attachment.storeOp                 = VK_ATTACHMENT_STORE_OP_STORE;
  attachment.initialLayout           = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
  attachment.finalLayout             = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;  // for blit operation
  attachment.flags                   = 0;

  VkAttachmentReference colorAttachmentRef = {};
  colorAttachmentRef.attachment            = 0;
  colorAttachmentRef.layout                = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

  VkSubpassDescription subpass    = {};
  subpass.pipelineBindPoint       = VK_PIPELINE_BIND_POINT_GRAPHICS;
  subpass.colorAttachmentCount    = 1;
  subpass.pColorAttachments       = &colorAttachmentRef;
  subpass.pDepthStencilAttachment = nullptr;

  // TODO: Should this have a dependency on external data?
  VkRenderPassCreateInfo rpInfo = nvvk::make<VkRenderPassCreateInfo>();
  rpInfo.attachmentCount        = 1;
  rpInfo.pAttachments           = &attachment;
  rpInfo.subpassCount           = 1;
  rpInfo.pSubpasses             = &subpass;
  rpInfo.dependencyCount        = 0;

  NVVK_CHECK(vkCreateRenderPass(m_context, &rpInfo, NULL, &m_renderPassGUI));
  m_debug.setObjectName(m_renderPassGUI, "m_renderPassGUI");
}

void Sample::destroyNonGUIRenderPasses()
{
  if(m_renderPassColorDepthClear != nullptr)
  {
    vkDestroyRenderPass(m_context, m_renderPassColorDepthClear, NULL);
    m_renderPassColorDepthClear = nullptr;
  }

  if(m_renderPassWeighted != nullptr)
  {
    vkDestroyRenderPass(m_context, m_renderPassWeighted, NULL);
    m_renderPassWeighted = nullptr;
  }
}

void Sample::createNonGUIRenderPasses()
{
  destroyNonGUIRenderPasses();

  // m_renderPassColorDepthClear
  // Render pass for rendering to m_colorImage and m_depthImage, clearing them
  // beforehand. Both are in VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL.
  // We create this manually since (as of this writing) nvvk::createRenderPass
  // doesn't support multisampling.
  {
    std::array<VkAttachmentDescription, 2> attachments = {};  // Color attachment, depth attachment
    // Color attachment
    attachments[0].format         = m_colorImage.c_format;
    attachments[0].samples        = getSampleCountFlagBits(m_state.msaa);
    attachments[0].loadOp         = VK_ATTACHMENT_LOAD_OP_CLEAR;
    attachments[0].storeOp        = VK_ATTACHMENT_STORE_OP_STORE;
    attachments[0].stencilLoadOp  = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    attachments[0].stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    attachments[0].initialLayout  = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    attachments[0].finalLayout    = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    attachments[0].flags          = 0;

    // Color attachment reference
    VkAttachmentReference colorAttachmentRef = {};
    colorAttachmentRef.attachment            = 0;
    colorAttachmentRef.layout                = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

    // Depth attachment
    attachments[1]               = attachments[0];
    attachments[1].format        = m_depthImage.c_format;
    attachments[1].initialLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
    attachments[1].finalLayout   = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

    // Depth attachment reference
    VkAttachmentReference depthAttachmentRef = {};
    depthAttachmentRef.attachment            = 1;
    depthAttachmentRef.layout                = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

    // 1 subpass
    VkSubpassDescription subpass    = {};
    subpass.pipelineBindPoint       = VK_PIPELINE_BIND_POINT_GRAPHICS;
    subpass.colorAttachmentCount    = 1;
    subpass.pColorAttachments       = &colorAttachmentRef;
    subpass.pDepthStencilAttachment = &depthAttachmentRef;

    // We only need to specify one dependency: Since the subpass has a barrier, the subpass will
    // need a self-dependency. (There are implicit external dependencies that are automatically added.)
    VkSubpassDependency selfDependency;
    selfDependency.srcSubpass      = 0;
    selfDependency.dstSubpass      = 0;
    selfDependency.srcStageMask    = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
    selfDependency.dstStageMask    = selfDependency.srcStageMask;
    selfDependency.srcAccessMask   = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
    selfDependency.dstAccessMask   = selfDependency.srcAccessMask;
    selfDependency.dependencyFlags = VK_DEPENDENCY_BY_REGION_BIT;  // Required, since we use framebuffer-space stages

    // No dependency on external data
    VkRenderPassCreateInfo rpInfo = nvvk::make<VkRenderPassCreateInfo>();
    rpInfo.attachmentCount        = static_cast<uint32_t>(attachments.size());
    rpInfo.pAttachments           = attachments.data();
    rpInfo.subpassCount           = 1;
    rpInfo.pSubpasses             = &subpass;
    rpInfo.dependencyCount        = 1;
    rpInfo.pDependencies          = &selfDependency;

    NVVK_CHECK(vkCreateRenderPass(m_context, &rpInfo, NULL, &m_renderPassColorDepthClear));
    m_debug.setObjectName(m_renderPassColorDepthClear, "m_renderPassColorDepthClear");
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
    VkAttachmentDescription weightedColorAttachment = {};
    weightedColorAttachment.format                  = m_oitWeightedColorFormat;
    weightedColorAttachment.samples                 = static_cast<VkSampleCountFlagBits>(m_state.msaa);
    weightedColorAttachment.loadOp                  = VK_ATTACHMENT_LOAD_OP_CLEAR;
    weightedColorAttachment.storeOp                 = VK_ATTACHMENT_STORE_OP_STORE;
    weightedColorAttachment.stencilLoadOp           = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    weightedColorAttachment.stencilStoreOp          = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    weightedColorAttachment.initialLayout           = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    weightedColorAttachment.finalLayout             = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

    VkAttachmentDescription weightedRevealAttachment = weightedColorAttachment;
    weightedRevealAttachment.format                  = m_oitWeightedRevealFormat;

    VkAttachmentDescription colorAttachment = weightedColorAttachment;
    colorAttachment.format                  = m_colorImage.c_format;
    colorAttachment.loadOp                  = VK_ATTACHMENT_LOAD_OP_LOAD;

    VkAttachmentDescription depthAttachment = colorAttachment;
    depthAttachment.format                  = m_depthImage.c_format;
    depthAttachment.initialLayout           = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
    depthAttachment.finalLayout             = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

    const std::array<VkAttachmentDescription, 4> allAttachments = {weightedColorAttachment, weightedRevealAttachment,
                                                                   colorAttachment, depthAttachment};

    std::array<VkSubpassDescription, 2> subpasses{};

    // Subpass 0 - weighted textures & depth texture for testing
    std::array<VkAttachmentReference, 2> subpass0ColorAttachments{};
    subpass0ColorAttachments[0].attachment = 0;
    subpass0ColorAttachments[0].layout     = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    subpass0ColorAttachments[1].attachment = 1;
    subpass0ColorAttachments[1].layout     = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

    VkAttachmentReference depthAttachmentRef{};
    depthAttachmentRef.attachment = 3;  // i.e. m_depthImage
    depthAttachmentRef.layout     = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

    subpasses[0].pipelineBindPoint       = VK_PIPELINE_BIND_POINT_GRAPHICS;
    subpasses[0].colorAttachmentCount    = static_cast<uint32_t>(subpass0ColorAttachments.size());
    subpasses[0].pColorAttachments       = subpass0ColorAttachments.data();
    subpasses[0].pDepthStencilAttachment = &depthAttachmentRef;

    // Subpass 1
    VkAttachmentReference subpass1ColorAttachment{};
    subpass1ColorAttachment.attachment = 2;  // i.e. m_colorImage
    subpass1ColorAttachment.layout     = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

    std::array<VkAttachmentReference, 2> subpass1InputAttachments{};
    subpass1InputAttachments[0].attachment = 0;
    subpass1InputAttachments[0].layout     = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    subpass1InputAttachments[1].attachment = 1;
    subpass1InputAttachments[1].layout     = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

    subpasses[1].pipelineBindPoint    = VK_PIPELINE_BIND_POINT_GRAPHICS;
    subpasses[1].colorAttachmentCount = 1;
    subpasses[1].pColorAttachments    = &subpass1ColorAttachment;
    subpasses[1].inputAttachmentCount = static_cast<uint32_t>(subpass1InputAttachments.size());
    subpasses[1].pInputAttachments    = subpass1InputAttachments.data();

    // Dependencies
    std::array<VkSubpassDependency, 3> subpassDependencies{};
    subpassDependencies[0].srcSubpass    = VK_SUBPASS_EXTERNAL;
    subpassDependencies[0].dstSubpass    = 0;
    subpassDependencies[0].srcStageMask  = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    subpassDependencies[0].dstStageMask  = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    subpassDependencies[0].srcAccessMask = 0;
    subpassDependencies[0].dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
    //
    subpassDependencies[1].srcSubpass    = 0;
    subpassDependencies[1].dstSubpass    = 1;
    subpassDependencies[1].srcStageMask  = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    subpassDependencies[1].dstStageMask  = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
    subpassDependencies[1].srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
    subpassDependencies[1].dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    // Finally, we have a dependency at the end to allow the images to transition back to VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL
    subpassDependencies[2].srcSubpass    = 1;
    subpassDependencies[2].dstSubpass    = VK_SUBPASS_EXTERNAL;
    subpassDependencies[2].srcStageMask  = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
    subpassDependencies[2].dstStageMask  = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    subpassDependencies[2].srcAccessMask = VK_ACCESS_SHADER_READ_BIT;
    subpassDependencies[2].dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;

    // Finally create the render pass
    VkRenderPassCreateInfo renderPassInfo = nvvk::make<VkRenderPassCreateInfo>();
    renderPassInfo.attachmentCount        = static_cast<uint32_t>(allAttachments.size());
    renderPassInfo.pAttachments           = allAttachments.data();
    renderPassInfo.dependencyCount        = static_cast<uint32_t>(subpassDependencies.size());
    renderPassInfo.pDependencies          = subpassDependencies.data();
    renderPassInfo.subpassCount           = static_cast<uint32_t>(subpasses.size());
    renderPassInfo.pSubpasses             = subpasses.data();
    NVVK_CHECK(vkCreateRenderPass(m_context, &renderPassInfo, nullptr, &m_renderPassWeighted));
    m_debug.setObjectName(m_renderPassWeighted, "m_renderPassWeighted");
  }
}

void Sample::updateShaderDefinitions()
{
  m_shaderModuleManager.m_prepend = nvh::ShaderFileManager::format(
      "#extension GL_GOOGLE_cpp_style_line_directive : enable\n"
      "#define OIT_LAYERS %d\n"
      "#define OIT_TAILBLEND %d\n"
      "#define OIT_MSAA %d\n"
      "#define OIT_SAMPLE_SHADING %d\n",
      m_state.oitLayers, m_state.tailBlend ? 1 : 0, m_state.msaa, m_state.sampleShading ? 1 : 0);
}

void Sample::createOrReloadShaderModules()
{
  updateShaderDefinitions();

  // You can set this to true to make sure that all of the shaders
  // compile correctly.
  const bool        loadEverything  = false;
  const std::string defineDepth     = "#define PASS PASS_DEPTH\n";
  const std::string defineColor     = "#define PASS PASS_COLOR\n";
  const std::string defineComposite = "#define PASS PASS_COMPOSITE\n";

  // Compile shaders

  // Scene (standard mesh rendering) and full-screen triangle vertex shaders
  createOrReloadShaderModule(m_shaderSceneVert, VK_SHADER_STAGE_VERTEX_BIT, "object.vert.glsl");
  createOrReloadShaderModule(m_shaderFullScreenTriangleVert, VK_SHADER_STAGE_VERTEX_BIT, "fullScreenTriangle.vert.glsl");
  // Opaque pass
  createOrReloadShaderModule(m_shaderOpaqueFrag, VK_SHADER_STAGE_FRAGMENT_BIT, "opaque.frag.glsl");

  if((m_state.algorithm == OIT_SIMPLE) || loadEverything)
  {
    const std::string file = "oitSimple.frag.glsl";
    createOrReloadShaderModule(m_shaderSimpleColorFrag, VK_SHADER_STAGE_FRAGMENT_BIT, file, defineColor);
    createOrReloadShaderModule(m_shaderSimpleCompositeFrag, VK_SHADER_STAGE_FRAGMENT_BIT, file, defineComposite);
  }
  if((m_state.algorithm == OIT_LINKEDLIST) || loadEverything)
  {
    const std::string file = "oitLinkedList.frag.glsl";
    createOrReloadShaderModule(m_shaderLinkedListColorFrag, VK_SHADER_STAGE_FRAGMENT_BIT, file, defineColor);
    createOrReloadShaderModule(m_shaderLinkedListCompositeFrag, VK_SHADER_STAGE_FRAGMENT_BIT, file, defineComposite);
  }
  if((m_state.algorithm == OIT_LOOP) || loadEverything)
  {
    const std::string file = "oitLoop.frag.glsl";
    createOrReloadShaderModule(m_shaderLoopDepthFrag, VK_SHADER_STAGE_FRAGMENT_BIT, file, defineDepth);
    createOrReloadShaderModule(m_shaderLoopColorFrag, VK_SHADER_STAGE_FRAGMENT_BIT, file, defineColor);
    createOrReloadShaderModule(m_shaderLoopCompositeFrag, VK_SHADER_STAGE_FRAGMENT_BIT, file, defineComposite);
  }
  if((m_state.algorithm == OIT_LOOP64) || loadEverything)
  {
    assert(m_context.hasDeviceExtension(VK_KHR_SHADER_ATOMIC_INT64_EXTENSION_NAME));
    const std::string file = "oitLoop64.frag.glsl";
    createOrReloadShaderModule(m_shaderLoop64ColorFrag, VK_SHADER_STAGE_FRAGMENT_BIT, file, defineColor);
    createOrReloadShaderModule(m_shaderLoop64CompositeFrag, VK_SHADER_STAGE_FRAGMENT_BIT, file, defineComposite);
  }
  if((m_state.algorithm == OIT_INTERLOCK) || loadEverything)
  {
    assert(m_context.hasDeviceExtension(VK_EXT_FRAGMENT_SHADER_INTERLOCK_EXTENSION_NAME));
    const std::string file = "oitInterlock.frag.glsl";
    createOrReloadShaderModule(m_shaderInterlockColorFrag, VK_SHADER_STAGE_FRAGMENT_BIT, file, defineColor);
    createOrReloadShaderModule(m_shaderInterlockCompositeFrag, VK_SHADER_STAGE_FRAGMENT_BIT, file, defineComposite);
  }
  if((m_state.algorithm == OIT_SPINLOCK) || loadEverything)
  {
    const std::string file = "oitSpinlock.frag.glsl";
    createOrReloadShaderModule(m_shaderSpinlockColorFrag, VK_SHADER_STAGE_FRAGMENT_BIT, file, defineColor);
    createOrReloadShaderModule(m_shaderSpinlockCompositeFrag, VK_SHADER_STAGE_FRAGMENT_BIT, file, defineComposite);
  }
  if((m_state.algorithm == OIT_WEIGHTED) || loadEverything)
  {
    const std::string file = "oitWeighted.frag.glsl";
    createOrReloadShaderModule(m_shaderWeightedColorFrag, VK_SHADER_STAGE_FRAGMENT_BIT, file, defineColor);
    createOrReloadShaderModule(m_shaderWeightedCompositeFrag, VK_SHADER_STAGE_FRAGMENT_BIT, file, defineComposite);
  }

  // Verify that the shaders compiled correctly:
  assert(m_shaderModuleManager.areShaderModulesValid());
}

void Sample::destroyGraphicsPipelines()
{
  destroyGraphicsPipeline(m_pipelineOpaque);
  destroyGraphicsPipeline(m_pipelineSimpleColor);
  destroyGraphicsPipeline(m_pipelineSimpleComposite);
  destroyGraphicsPipeline(m_pipelineLinkedListColor);
  destroyGraphicsPipeline(m_pipelineLinkedListComposite);
  destroyGraphicsPipeline(m_pipelineLoopDepth);
  destroyGraphicsPipeline(m_pipelineLoopColor);
  destroyGraphicsPipeline(m_pipelineLoopComposite);
  destroyGraphicsPipeline(m_pipelineLoop64Color);
  destroyGraphicsPipeline(m_pipelineLoop64Composite);
  destroyGraphicsPipeline(m_pipelineInterlockColor);
  destroyGraphicsPipeline(m_pipelineInterlockComposite);
  destroyGraphicsPipeline(m_pipelineSpinlockColor);
  destroyGraphicsPipeline(m_pipelineSpinlockComposite);
  destroyGraphicsPipeline(m_pipelineWeightedColor);
  destroyGraphicsPipeline(m_pipelineWeightedComposite);
}

void Sample::createGraphicsPipelines()
{
  destroyGraphicsPipelines();

  // We always need the opaque pipeline:
  m_pipelineOpaque =
      createGraphicsPipeline(m_shaderSceneVert, m_shaderOpaqueFrag, BlendMode::NONE, true, false, m_renderPassColorDepthClear);

  const bool transparentDoubleSided = true;  // Iff transparent objects are double-sided

  // Switch off between algorithms:
  switch(m_state.algorithm)
  {
    case OIT_SIMPLE:
      m_pipelineSimpleColor = createGraphicsPipeline(m_shaderSceneVert, m_shaderSimpleColorFrag, BlendMode::PREMULTIPLIED,
                                                     true, transparentDoubleSided, m_renderPassColorDepthClear);
      m_pipelineSimpleComposite =
          createGraphicsPipeline(m_shaderFullScreenTriangleVert, m_shaderSimpleCompositeFrag, BlendMode::PREMULTIPLIED,
                                 false, transparentDoubleSided, m_renderPassColorDepthClear);
      break;
    case OIT_LINKEDLIST:
      m_pipelineLinkedListColor = createGraphicsPipeline(m_shaderSceneVert, m_shaderLinkedListColorFrag, BlendMode::PREMULTIPLIED,
                                                         true, transparentDoubleSided, m_renderPassColorDepthClear);
      m_pipelineLinkedListComposite =
          createGraphicsPipeline(m_shaderFullScreenTriangleVert, m_shaderLinkedListCompositeFrag,
                                 BlendMode::PREMULTIPLIED, false, transparentDoubleSided, m_renderPassColorDepthClear);
      break;
    case OIT_LOOP:
      m_pipelineLoopDepth = createGraphicsPipeline(m_shaderSceneVert, m_shaderLoopDepthFrag, BlendMode::PREMULTIPLIED,
                                                   true, transparentDoubleSided, m_renderPassColorDepthClear);
      m_pipelineLoopColor = createGraphicsPipeline(m_shaderSceneVert, m_shaderLoopColorFrag, BlendMode::PREMULTIPLIED,
                                                   true, transparentDoubleSided, m_renderPassColorDepthClear);
      m_pipelineLoopComposite =
          createGraphicsPipeline(m_shaderFullScreenTriangleVert, m_shaderLoopCompositeFrag, BlendMode::PREMULTIPLIED,
                                 false, transparentDoubleSided, m_renderPassColorDepthClear);
      break;
    case OIT_LOOP64:
      m_pipelineLoop64Color = createGraphicsPipeline(m_shaderSceneVert, m_shaderLoop64ColorFrag, BlendMode::PREMULTIPLIED,
                                                     true, transparentDoubleSided, m_renderPassColorDepthClear);
      m_pipelineLoop64Composite =
          createGraphicsPipeline(m_shaderFullScreenTriangleVert, m_shaderLoop64CompositeFrag, BlendMode::PREMULTIPLIED,
                                 false, transparentDoubleSided, m_renderPassColorDepthClear);
      break;
    case OIT_INTERLOCK:
      m_pipelineInterlockColor = createGraphicsPipeline(m_shaderSceneVert, m_shaderInterlockColorFrag, BlendMode::PREMULTIPLIED,
                                                        true, transparentDoubleSided, m_renderPassColorDepthClear);
      m_pipelineInterlockComposite =
          createGraphicsPipeline(m_shaderFullScreenTriangleVert, m_shaderInterlockCompositeFrag,
                                 BlendMode::PREMULTIPLIED, false, transparentDoubleSided, m_renderPassColorDepthClear);
      break;
    case OIT_SPINLOCK:
      m_pipelineSpinlockColor = createGraphicsPipeline(m_shaderSceneVert, m_shaderSpinlockColorFrag, BlendMode::PREMULTIPLIED,
                                                       true, transparentDoubleSided, m_renderPassColorDepthClear);
      m_pipelineSpinlockComposite =
          createGraphicsPipeline(m_shaderFullScreenTriangleVert, m_shaderSpinlockCompositeFrag,
                                 BlendMode::PREMULTIPLIED, false, transparentDoubleSided, m_renderPassColorDepthClear);
      break;
    case OIT_WEIGHTED:
      m_pipelineWeightedColor = createGraphicsPipeline(m_shaderSceneVert, m_shaderWeightedColorFrag, BlendMode::WEIGHTED_COLOR,
                                                       true, transparentDoubleSided, m_renderPassWeighted, 0);
      m_pipelineWeightedComposite =
          createGraphicsPipeline(m_shaderFullScreenTriangleVert, m_shaderWeightedCompositeFrag,
                                 BlendMode::WEIGHTED_COMPOSITE, false, transparentDoubleSided, m_renderPassWeighted, 1);
      break;
  }
}