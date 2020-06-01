/* Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

// This file contains implementations of the resource creation functions of
// Sample that are specific to order-independent transparency (for instance,
// render passes, descriptor sets, and A-buffers, but not swapchain creation.)

#include "oit.h"
#include <nvvk/renderpasses_vk.hpp>

void Sample::destroyImages()
{
  m_colorImage.destroy(m_context, m_allocatorDma);
  m_downsampleTargetImage.destroy(m_context, m_allocatorDma);
  m_depthImage.destroy(m_context, m_allocatorDma);
  m_oitABuffer.destroy(m_context, m_allocatorDma);
  m_oitAuxImage.destroy(m_context, m_allocatorDma);
  m_oitAuxSpinImage.destroy(m_context, m_allocatorDma);
  m_oitAuxDepthImage.destroy(m_context, m_allocatorDma);
  m_oitCounterImage.destroy(m_context, m_allocatorDma);
  m_oitWeightedColorImage.destroy(m_context, m_allocatorDma);
  m_oitWeightedRevealImage.destroy(m_context, m_allocatorDma);
}

void Sample::createFrameImages(VkCommandBuffer cmdBuffer)
{
  destroyImages();

  // We implement supersample anti-aliasing by rendering to a larger texture
  const int bufferWidth  = m_swapChain.getWidth() * m_state.supersample;
  const int bufferHeight = m_swapChain.getHeight() * m_state.supersample;

  // Offscreen color and depth buffer
  {
    // Color image, created with an sRGB format.
    m_colorImage.create(m_context, m_allocatorDma, VK_IMAGE_TYPE_2D, VK_IMAGE_ASPECT_COLOR_BIT, VK_FORMAT_B8G8R8A8_SRGB, bufferWidth,
                        bufferHeight, 1, VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT, m_state.msaa);
    m_colorImage.setName(m_debug, "m_colorImage");
    // We'll put it into the layout for a color attachment later.

    // Intermediate storage for resolve - 1spp, and with almost the same format
    // as the backbuffer (but sRGB). In the future, we could replace this with
    // a compute kernel that does downsampling manually.
    m_downsampleTargetImage.create(m_context, m_allocatorDma, VK_IMAGE_TYPE_2D, VK_IMAGE_ASPECT_COLOR_BIT,
                                   VK_FORMAT_B8G8R8A8_SRGB, m_swapChain.getWidth(), m_swapChain.getHeight(), 1,
                                   VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT, 1);
    m_colorImage.setName(m_debug, "m_downsampleTargetImage");

    // Depth image
    VkFormat depthFormat = nvvk::findDepthFormat(m_context.m_physicalDevice);

    m_depthImage.create(m_context, m_allocatorDma, VK_IMAGE_TYPE_2D, VK_IMAGE_ASPECT_DEPTH_BIT, depthFormat,
                        bufferWidth, bufferHeight, 1, VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT, m_state.msaa);
    m_depthImage.setName(m_debug, "m_depthImage");

    // Unfortunately, we do need to transition the depth image here, since the
    // render pass assumes the depth image already has the optimal
    // depth-stencil layout.

    m_depthImage.transitionTo(cmdBuffer,                                         // Command buffer
                              VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,  // New layout
                              VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,                // Stages to block
                              VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT);
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
  // The pipeline stages that auxiliary images can be bound to
  const VkPipelineStageFlags auxPipelineStages = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT;
  // The ways that auxiliary images can be accessed
  const VkAccessFlags auxAccesses = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;
  // if `sampleShading`, then each auxiliary image is actually a texture array:
  const uint32_t auxLayers = (sampleShading ? m_state.msaa : 1);

  if(allocAux)
  {
    m_oitAuxImage.create(m_context, m_allocatorDma, VK_IMAGE_TYPE_2D, VK_IMAGE_ASPECT_COLOR_BIT, VK_FORMAT_R32_UINT,
                         bufferWidth, bufferHeight, auxLayers, auxUsages);
    m_oitAuxImage.setName(m_debug, "m_oitAuxImage");
    m_oitAuxImage.transitionTo(cmdBuffer, VK_IMAGE_LAYOUT_GENERAL, auxPipelineStages, auxAccesses);
  }

  if(allocAuxSpin)
  {
    m_oitAuxSpinImage.create(m_context, m_allocatorDma, VK_IMAGE_TYPE_2D, VK_IMAGE_ASPECT_COLOR_BIT, VK_FORMAT_R32_UINT,
                             bufferWidth, bufferHeight, auxLayers, auxUsages);
    m_oitAuxSpinImage.setName(m_debug, "m_oitAuxSpinImage");
    m_oitAuxSpinImage.transitionTo(cmdBuffer, VK_IMAGE_LAYOUT_GENERAL, auxPipelineStages, auxAccesses);
  }

  if(allocAuxDepth)
  {
    m_oitAuxDepthImage.create(m_context, m_allocatorDma, VK_IMAGE_TYPE_2D, VK_IMAGE_ASPECT_COLOR_BIT,
                              VK_FORMAT_R32_UINT, bufferWidth, bufferHeight, auxLayers, auxUsages);
    m_oitAuxDepthImage.setName(m_debug, "m_oitAuxDepthImage");
    m_oitAuxDepthImage.transitionTo(cmdBuffer, VK_IMAGE_LAYOUT_GENERAL, auxPipelineStages, auxAccesses);
  }

  if(allocCounter)
  {
    // Here, a counter is really a 1x1x1 image.
    m_oitCounterImage.create(m_context, m_allocatorDma, VK_IMAGE_TYPE_2D, VK_IMAGE_ASPECT_COLOR_BIT, VK_FORMAT_R32_UINT,
                             1, 1, 1, auxUsages);
    m_oitCounterImage.setName(m_debug, "m_oitCounter");
    m_oitCounterImage.transitionTo(cmdBuffer, VK_IMAGE_LAYOUT_GENERAL, auxPipelineStages, auxAccesses);
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
    m_oitWeightedColorImage.transitionTo(cmdBuffer, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                                         VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT, 0);
    m_oitWeightedRevealImage.transitionTo(cmdBuffer, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                                          VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT, 0);
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

  // Information about the buffer and image descriptors we'll use.
  // When constructing VkWriteDescriptorSet objects, we'll take references
  // to these.

  // UBO_SCENE
  std::array<VkDescriptorBufferInfo, 3> uboBufferInfo;
  for(size_t ring = 0; ring < m_swapChain.getImageCount(); ring++)
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
  for(uint32_t ring = 0; ring < m_swapChain.getImageCount(); ring++)
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

void Sample::destroyRenderPasses()
{
  if(m_renderPassColorDepthClear != nullptr)
  {
    vkDestroyRenderPass(m_context, m_renderPassColorDepthClear, nullptr);
  }

  if(m_renderPassWeighted != nullptr)
  {
    vkDestroyRenderPass(m_context, m_renderPassWeighted, nullptr);
  }

  if(m_renderPassGUI != nullptr)
  {
    vkDestroyRenderPass(m_context, m_renderPassGUI, nullptr);
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
    std::vector<VkAttachmentDescription> allAttachments;

    VkAttachmentDescription colorAttachment = {};
    colorAttachment.format                  = m_colorImage.c_format;
    colorAttachment.samples                 = static_cast<VkSampleCountFlagBits>(m_state.msaa);
    colorAttachment.loadOp                  = VK_ATTACHMENT_LOAD_OP_CLEAR;
    colorAttachment.storeOp                 = VK_ATTACHMENT_STORE_OP_STORE;
    colorAttachment.stencilLoadOp           = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    colorAttachment.stencilStoreOp          = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    colorAttachment.initialLayout           = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    colorAttachment.finalLayout             = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

    VkAttachmentReference colorAttachmentRef = {};
    colorAttachmentRef.attachment            = static_cast<uint32_t>(allAttachments.size());
    colorAttachmentRef.layout                = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    allAttachments.push_back(colorAttachment);

    VkAttachmentDescription depthAttachment = colorAttachment;
    depthAttachment.format                  = m_depthImage.c_format;
    depthAttachment.initialLayout           = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
    depthAttachment.finalLayout             = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

    VkAttachmentReference depthAttachmentRef = {};
    depthAttachmentRef.attachment            = static_cast<uint32_t>(allAttachments.size());
    depthAttachmentRef.layout                = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
    allAttachments.push_back(depthAttachment);

    // 1 subpass
    VkSubpassDescription subpass    = {};
    subpass.pipelineBindPoint       = VK_PIPELINE_BIND_POINT_GRAPHICS;
    subpass.colorAttachmentCount    = 1;
    subpass.pColorAttachments       = &colorAttachmentRef;
    subpass.pDepthStencilAttachment = &depthAttachmentRef;

    VkSubpassDependency dependency = {};
    dependency.srcSubpass          = VK_SUBPASS_EXTERNAL;
    dependency.dstSubpass          = 0;
    dependency.srcStageMask        = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    dependency.dstStageMask        = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    dependency.srcAccessMask       = 0;
    dependency.dstAccessMask       = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;

    VkRenderPassCreateInfo renderPassInfo = nvvk::make<VkRenderPassCreateInfo>();
    renderPassInfo.attachmentCount        = static_cast<uint32_t>(allAttachments.size());
    renderPassInfo.pAttachments           = allAttachments.data();
    renderPassInfo.dependencyCount        = 1;
    renderPassInfo.pDependencies          = &dependency;
    renderPassInfo.subpassCount           = 1;
    renderPassInfo.pSubpasses             = &subpass;
    VkResult result = vkCreateRenderPass(m_context, &renderPassInfo, nullptr, &m_renderPassColorDepthClear);
    assert(result == VK_SUCCESS);
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
    VkResult result = vkCreateRenderPass(m_context, &renderPassInfo, nullptr, &m_renderPassWeighted);
    assert(result == VK_SUCCESS);
    m_debug.setObjectName(m_renderPassWeighted, "m_renderPassWeighted");
  }

  // Swap chain framebuffer, no depth, no MSAA
  std::vector<VkFormat> guiColorAttachmentFormats{m_swapChain.getFormat()};
  m_renderPassGUI = nvvk::createRenderPass(m_context,                  // Device
                                           guiColorAttachmentFormats,  // List of color attachment formats,
                                           VK_FORMAT_UNDEFINED,        // No depth attachment
                                           1,                          // Number of subpasses
                                           false,                      // Don't clear color
                                           false,                      // Don't clear depth
                                           VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,  // Initial layout
                                           VK_IMAGE_LAYOUT_PRESENT_SRC_KHR);          // Final layout
  m_debug.setObjectName(m_renderPassGUI, "m_renderPassGUI");
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