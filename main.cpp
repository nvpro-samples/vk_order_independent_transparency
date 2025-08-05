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


// This sample demonstrates several ways of rendering transparent objects
// without requiring them to be sorted in advance, including both
// algorithms that produce ground-truth images if given enough memory, and
// an algorithm that produces approximate results.
// For more information on these techniques, run the sample, see
// oitScene.frag.glsl,or read the documentation that comes with this sample.

// Here's how the C++ code is organized:
// oit.h: Main Sample application structure with all functions.
// oitRender.cpp: Main OIT-specific rendering functions.
// oit.cpp: Main OIT-specific resource creation functions.
// oitGui.cpp: GUI for the application.
// utilities_vk.h: Helper functions that can exist without a sample.
// main.cpp: All other functions not specific to OIT.

#pragma warning(disable : 26812)  // Disable the warning about Vulkan's enumerations being untyped in VS2019.

#define VMA_IMPLEMENTATION

#include "oit.h"

#include <nvapp/elem_default_title.hpp>
#include <nvutils/file_operations.hpp>
#include <nvutils/logger.hpp>
#include <nvvk/context.hpp>
#include <nvvk/staging.hpp>

#include <cassert>
#include <random>

// Application constants
constexpr int   GRID_SIZE    = 16;
constexpr float GLOBAL_SCALE = 8.0f;

// TODO: Rename section headers

///////////////////////////////////////////////////////////////////////////////
// Callbacks                                                                 //
///////////////////////////////////////////////////////////////////////////////

void Sample::onResize(VkCommandBuffer cmd, const VkExtent2D& size)
{
  updateRendererFromState(true, false);
  // This is here because it happens before ImGui.
  m_viewportImage.update(cmd, size);
}

///////////////////////////////////////////////////////////////////////////////
// Object Creation, Destruction, and Recreation                              //
///////////////////////////////////////////////////////////////////////////////

void Sample::onAttach(nvapp::Application* app)
{
  m_app = app;

  // Camera
  m_cameraControl = std::make_shared<nvutils::CameraManipulator>();
  m_cameraElement = std::make_shared<nvapp::ElementCamera>();
  m_cameraElement->setCameraManipulator(m_cameraControl);
  m_app->addElement(m_cameraElement);

  // Profiler
  m_profilerTimeline = m_profiler.createTimeline({.name = "Primary"});
  m_profilerGPU.init(m_profilerTimeline, m_ctx->getDevice(), m_ctx->getPhysicalDevice(), m_app->getQueue(0).familyIndex, true);
  m_profilerGUI =
      std::make_shared<nvapp::ElementProfiler>(&m_profiler, std::make_shared<nvapp::ElementProfiler::ViewSettings>());
  m_app->addElement(m_profilerGUI);

  // Debug utility
  nvvk::DebugUtil::getInstance().init(m_ctx->getDevice());

  // Allocator
  m_allocator.init(VmaAllocatorCreateInfo{
      .physicalDevice = m_ctx->getPhysicalDevice(), .device = m_ctx->getDevice(), .instance = m_ctx->getInstance()});

  // Point sampler
  createTextureSampler();

  // Viewport image parameters - 1spp, swapchain sized, with almost the same format as the swapchain
  // (with the exception that the channels have to be in the same order as m_colorImage)
  m_viewportImage.init(nvvk::GBufferInitInfo{.allocator      = &m_allocator,
                                             .colorFormats   = {m_viewportColorFormat},
                                             .imageSampler   = m_pointSampler,
                                             .descriptorPool = m_app->getTextureDescriptorPool()});

  // Configure shader system
  {
    const std::filesystem::path exeDir = nvutils::getExecutablePath().parent_path();
    m_shaderCompiler.addSearchPaths({exeDir / TARGET_NAME "_files/shaders",  // Install path
                                     exeDir / TARGET_EXE_TO_SOURCE_DIRECTORY / "shaders"});
  }

  // Call cmdUpdateRendererFromState with forceRebuildAll = true to set up the rest of the renderer with the initial
  // swapchain size.
  updateRendererFromState(true, true);

  // Initialize camera
  m_cameraControl->setLookat(glm::vec3(0, 0, 0.75f * static_cast<float>(GRID_SIZE)),  // eye
                             glm::vec3(0.0f),                                         //center
                             glm::vec3(0.0f, 1.0f, 0.0f));                            //up

  // Initialize the UBO
  m_sceneUbo.alphaMin   = 0.2f;
  m_sceneUbo.alphaWidth = 0.3f;
}

void Sample::updateRendererFromState(bool swapchainSizeChanged, bool forceRebuildAll)
{
  m_state.recomputeAntialiasingSettings();

  // Determine what needs to be rebuilt
  swapchainSizeChanged |= forceRebuildAll;

  const bool uniformBuffersNeedReinit = (m_uniformBuffers.size() != m_app->getFrameCycleSize()) || forceRebuildAll;

  const bool shadersNeedUpdate = (m_state.algorithm != m_lastState.algorithm)                       //
                                 || (m_state.oitLayers != m_lastState.oitLayers)                    //
                                 || (m_state.tailBlend != m_lastState.tailBlend)                    //
                                 || (m_state.interlockIsOrdered != m_lastState.interlockIsOrdered)  //
                                 || (m_state.msaa != m_lastState.msaa)                              //
                                 || (m_state.sampleShading != m_lastState.sampleShading)            //
                                 || forceRebuildAll;

  const bool sceneNeedsReinit = (m_state.numObjects != m_lastState.numObjects)     //
                                || (m_state.scaleWidth != m_lastState.scaleWidth)  //
                                || (m_state.scaleMin != m_lastState.scaleMin)      //
                                || (m_state.subdiv != m_lastState.subdiv)          //
                                || forceRebuildAll;

  const bool imagesNeedReinit = (m_state.supersample != m_lastState.supersample)         //
                                || (m_state.msaa != m_lastState.msaa)                    //
                                || (m_state.algorithm != m_lastState.algorithm)          //
                                || (m_state.sampleShading != m_lastState.sampleShading)  //
                                || (m_state.oitLayers != m_lastState.oitLayers)          //
                                || ((m_state.algorithm == OIT_LINKEDLIST)
                                    && (m_state.linkedListAllocatedPerElement != m_lastState.linkedListAllocatedPerElement))  //
                                || swapchainSizeChanged  //
                                || forceRebuildAll;

  const bool descriptorSetsNeedReinit = ((m_state.algorithm == OIT_LOOP64) && (m_lastState.algorithm != OIT_LOOP64))  //
                                        || ((m_state.algorithm != OIT_LOOP64) && (m_lastState.algorithm == OIT_LOOP64))  //
                                        || forceRebuildAll;

  const bool framebuffersAndDescriptorsNeedReinit = imagesNeedReinit  //
                                                    || forceRebuildAll;

  const bool renderPassesNeedReinit = (m_state.msaa != m_lastState.msaa)  //
                                      || forceRebuildAll;

  const bool pipelinesNeedReinit = (m_state.algorithm != m_lastState.algorithm)  //
                                   || shadersNeedUpdate || imagesNeedReinit;

  const bool anythingChanged = uniformBuffersNeedReinit || shadersNeedUpdate || sceneNeedsReinit || imagesNeedReinit
                               || descriptorSetsNeedReinit || framebuffersAndDescriptorsNeedReinit || renderPassesNeedReinit;

  if(anythingChanged)
  {
    const VkExtent2D viewportSize = getViewportSize();
    LOGI("Framebuffer: %u x %u, %d MSAA sample(s)\n", viewportSize.width, viewportSize.height, m_state.msaa);
    LOGI("Building:\n");
    if(uniformBuffersNeedReinit)
      LOGI("  Uniform buffers\n");
    if(sceneNeedsReinit)
      LOGI("  Scene\n");
    if(imagesNeedReinit)
      LOGI("  Frame images\n");
    if(descriptorSetsNeedReinit)
      LOGI("  Descriptor sets\n");
    if(renderPassesNeedReinit)
      LOGI("  Render passes\n");
    if(framebuffersAndDescriptorsNeedReinit)
      LOGI("  Framebuffers\n");
    if(shadersNeedUpdate)
      LOGI("  Shaders\n");
    if(pipelinesNeedReinit)
      LOGI("  Pipelines\n");

    vkDeviceWaitIdle(m_ctx->getDevice());
    // TODO: Pass this command buffer to more functions that can use it
    VkCommandBuffer cmd = m_app->createTempCmdBuffer();

    if(uniformBuffersNeedReinit)
    {
      createUniformBuffers();
    }

    if(sceneNeedsReinit)
    {
      initScene();
    }

    if(imagesNeedReinit)
    {
      createFrameImages(cmd);
    }

    if(descriptorSetsNeedReinit)
    {
      createDescriptorSets();
    }

    if(renderPassesNeedReinit)
    {
      createRenderPasses();
    }

    if(framebuffersAndDescriptorsNeedReinit)
    {
      updateAllDescriptorSets();
      createFramebuffers();
    }

    if(shadersNeedUpdate)
    {
      createOrReloadShaderModules();
    }

    if(pipelinesNeedReinit)
    {
      createGraphicsPipelines();
    }

    m_app->submitAndWaitTempCmdBuffer(cmd);

    m_lastState = m_state;
  }
}

void Sample::onDetach()
{
  vkDeviceWaitIdle(m_ctx->getDevice());  // Not sure if still needed
  m_profilerGPU.deinit();
  m_profiler.destroyTimeline(m_profilerTimeline);

  // From updateRendererFromState
  destroyGraphicsPipelines();
  destroyShaderModules();
  destroyFramebuffers();
  destroyRenderPasses();
  destroyDescriptorSets();
  destroyFrameImages();
  destroyScene();
  destroyUniformBuffers();

  // from onAttach()
  m_viewportImage.deinit();
  m_allocator.deinit();
  destroyTextureSampler();
}

void Sample::destroyTextureSampler()
{
  vkDestroySampler(m_ctx->getDevice(), m_pointSampler, nullptr);
}

void Sample::createTextureSampler()
{
  // Create a point sampler using base Vulkan
  const VkSamplerCreateInfo samplerInfo = {
      .sType                   = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
      .magFilter               = VK_FILTER_LINEAR,
      .minFilter               = VK_FILTER_LINEAR,
      .mipmapMode              = VK_SAMPLER_MIPMAP_MODE_NEAREST,
      .addressModeU            = VK_SAMPLER_ADDRESS_MODE_REPEAT,
      .addressModeV            = VK_SAMPLER_ADDRESS_MODE_REPEAT,
      .addressModeW            = VK_SAMPLER_ADDRESS_MODE_REPEAT,
      .anisotropyEnable        = VK_FALSE,
      .compareEnable           = VK_FALSE,
      .borderColor             = VK_BORDER_COLOR_INT_OPAQUE_BLACK,
      .unnormalizedCoordinates = VK_FALSE,
  };

  NVVK_CHECK(vkCreateSampler(m_ctx->getDevice(), &samplerInfo, nullptr, &m_pointSampler));
}

void Sample::destroyUniformBuffers()
{
  for(nvvk::Buffer& uniformBuffer : m_uniformBuffers)
  {
    m_allocator.destroyBuffer(uniformBuffer);
  }
  m_uniformBuffers.clear();
}

void Sample::createUniformBuffers()
{
  destroyUniformBuffers();

  VkDeviceSize bufferSize = sizeof(shaderio::SceneData);

  const uint32_t numSwapChainImages = m_app->getFrameCycleSize();
  m_uniformBuffers.resize(numSwapChainImages);

  for(uint32_t i = 0; i < numSwapChainImages; i++)
  {
    NVVK_CHECK(m_allocator.createBuffer(m_uniformBuffers[i],                  // Buffer
                                        bufferSize,                           // Buffer size
                                        VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,   // Usage
                                        VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE,  // VMA memory usage
                                        VMA_ALLOCATION_CREATE_MAPPED_BIT      // Persistently map the memory
                                            | VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT));  // We'll memcpy to it all at once
  }
}

void Sample::destroyScene()
{
  m_allocator.destroyBuffer(m_indexBuffer);
  m_allocator.destroyBuffer(m_vertexBuffer);
}

void Sample::initScene()
{
  destroyScene();
  // A mesh consists of vectors of vertices and triangle list indices.
  std::vector<Vertex> vertices;
  static_assert(alignof(Vertex) == 4 && sizeof(Vertex) == 40);
  std::vector<glm::uvec3> triangles;
  static_assert(alignof(glm::vec3) == 4 && sizeof(glm::vec3) == 12);

  // It'll contain multiple instances of this sphere. For now, we'll flatten it
  // into a single pair of buffers, but we could certainly use instanced calls
  // here.
  const nvutils::PrimitiveMesh sphere = nvutils::createSphereUv(1.0f, m_state.subdiv * 2, m_state.subdiv);
  m_objectTriangleIndices             = static_cast<uint32_t>(3 * sphere.triangles.size());

  // We'll use C++11-style random number generation here
  std::default_random_engine            rnd(3625);  // Fixed seed
  std::uniform_real_distribution<float> uniformDist;

  for(uint32_t object = 0; object < uint32_t(m_state.numObjects); object++)
  {
    // Generate a random position in [-GLOBAL_SCALE/2, GLOBAL_SCALE/2)^3
    glm::vec3 center(uniformDist(rnd), uniformDist(rnd), uniformDist(rnd));
    center = (center - glm::vec3(0.5)) * GLOBAL_SCALE;

    // Generate a random radius
    float radius = GLOBAL_SCALE * 0.9f / GRID_SIZE;
    radius *= uniformDist(rnd) * m_state.scaleWidth + m_state.scaleMin;

    // Generate a random color and transparency. Since the color we'll store
    // will be in unpremultiplied linear space but we want a perceptual-ish
    // distribution of colors, we square .rgb.
    glm::vec4 color(uniformDist(rnd), uniformDist(rnd), uniformDist(rnd), uniformDist(rnd));
    color.x *= color.x;
    color.y *= color.y;
    color.z *= color.z;

    // What's the index of our first vertex?
    const uint32_t firstVertex = static_cast<uint32_t>(vertices.size());

    // Append a scaled and translated version of the sphere.
    for(size_t v = 0; v < sphere.vertices.size(); v++)
    {
      Vertex vtx = sphere.vertices[v];
      vtx.pos    = vtx.pos * radius + center;
      vtx.color  = color;
      vertices.push_back(vtx);
    }
    for(size_t triIdx = 0; triIdx < sphere.triangles.size(); triIdx++)
    {
      const glm::uvec3 indices = firstVertex + sphere.triangles[triIdx].indices;
      triangles.push_back(indices);
    }
  }

  // Count the total number of triangle indices
  m_sceneTriangleIndices = static_cast<uint32_t>(3 * triangles.size());

  // Create the vertex and index buffers and synchronously upload them to the
  // GPU, waiting for them to finish uploading. Note that applications may wish
  // to implement asynchronous uploads, which you can see how to do in the
  // vk_async_resources sample.
  nvvk::StagingUploader uploader;
  uploader.init(&m_allocator);
  {
    VkCommandBuffer cmd = m_app->createTempCmdBuffer();

    // Create vertex buffer
    const VkDeviceSize vtxBufferSize = static_cast<VkDeviceSize>(sizeof(vertices[0]) * vertices.size());
    NVVK_CHECK(m_allocator.createBuffer(m_vertexBuffer, vtxBufferSize, VK_BUFFER_USAGE_VERTEX_BUFFER_BIT));
    NVVK_DBG_NAME(m_vertexBuffer.buffer);
    uploader.appendBuffer<Vertex>(m_vertexBuffer, 0, vertices);

    const VkDeviceSize idxBufferSize = static_cast<VkDeviceSize>(sizeof(triangles[0]) * triangles.size());
    NVVK_CHECK(m_allocator.createBuffer(m_indexBuffer, idxBufferSize, VK_BUFFER_USAGE_INDEX_BUFFER_BIT));
    NVVK_DBG_NAME(m_indexBuffer.buffer);
    uploader.appendBuffer<glm::uvec3>(m_indexBuffer, 0, triangles);

    uploader.cmdUploadAppended(cmd);
    // Once this returns, all of the copy operations will have been completed.
    m_app->submitAndWaitTempCmdBuffer(cmd);
  }
  uploader.deinit();
}

void Sample::destroyFramebuffers()
{
  vkDestroyFramebuffer(m_app->getDevice(), m_mainColorDepthFramebuffer, nullptr);
  m_mainColorDepthFramebuffer = VK_NULL_HANDLE;

  if(m_weightedFramebuffer != VK_NULL_HANDLE)
  {
    vkDestroyFramebuffer(m_app->getDevice(), m_weightedFramebuffer, nullptr);
    m_weightedFramebuffer = VK_NULL_HANDLE;
  }
}

void Sample::createFramebuffers()
{
  destroyFramebuffers();
  // TODO: Remove and replace with dynamic rendering

  // Color + depth offscreen framebuffer
  {
    const std::array<VkImageView, 2> attachments{m_colorImage.getView(), m_depthImage.getView()};
    const VkFramebufferCreateInfo    fbInfo{.sType           = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO,
                                            .renderPass      = m_renderPassColorDepthClear,
                                            .attachmentCount = static_cast<uint32_t>(attachments.size()),
                                            .pAttachments    = attachments.data(),
                                            .width           = m_colorImage.getWidth(),
                                            .height          = m_colorImage.getHeight(),
                                            .layers          = 1};

    NVVK_CHECK(vkCreateFramebuffer(m_app->getDevice(), &fbInfo, NULL, &m_mainColorDepthFramebuffer));
    NVVK_DBG_NAME(m_mainColorDepthFramebuffer);
  }

  // Weighted color + weighted reveal framebuffer (for Weighted, Blended
  // Order-Independent Transparency). See the render pass description for more info.
  if(m_state.algorithm == OIT_WEIGHTED)
  {
    const std::array<VkImageView, 4> attachments{m_oitWeightedColorImage.getView(),   //
                                                 m_oitWeightedRevealImage.getView(),  //
                                                 m_colorImage.getView(),              //
                                                 m_depthImage.getView()};

    const VkFramebufferCreateInfo fbInfo{.sType           = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO,
                                         .renderPass      = m_renderPassWeighted,
                                         .attachmentCount = static_cast<uint32_t>(attachments.size()),
                                         .pAttachments    = attachments.data(),
                                         .width           = m_oitWeightedColorImage.getWidth(),
                                         .height          = m_oitWeightedColorImage.getHeight(),
                                         .layers          = 1};

    NVVK_CHECK(vkCreateFramebuffer(m_app->getDevice(), &fbInfo, nullptr, &m_weightedFramebuffer));
    NVVK_DBG_NAME(m_weightedFramebuffer);
  }
}

VkPipeline Sample::createGraphicsPipeline(const std::string&   debugName,
                                          const VkShaderModule vertShaderModule,
                                          const VkShaderModule fragShaderModule,
                                          BlendMode            blendMode,
                                          bool                 usesVertexInput,
                                          bool                 isDoubleSided,
                                          VkRenderPass         renderPass,
                                          uint32_t             subpass)
{
  const std::array<VkPipelineShaderStageCreateInfo, 2> stages = {
      VkPipelineShaderStageCreateInfo{.sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                                      .stage  = VK_SHADER_STAGE_VERTEX_BIT,
                                      .module = vertShaderModule,
                                      .pName  = "main"},
      VkPipelineShaderStageCreateInfo{.sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                                      .stage  = VK_SHADER_STAGE_FRAGMENT_BIT,
                                      .module = fragShaderModule,
                                      .pName  = "main"}};

  const VkVertexInputBindingDescription vtxBindingDescription = Vertex::getBindingDescription();
  const auto                            vtxAttributes         = Vertex::getAttributeDescriptions();

  VkPipelineVertexInputStateCreateInfo vertexInput{.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
  if(usesVertexInput)
  {
    vertexInput.vertexBindingDescriptionCount   = 1;
    vertexInput.pVertexBindingDescriptions      = &vtxBindingDescription;
    vertexInput.vertexAttributeDescriptionCount = uint32_t(vtxAttributes.size());
    vertexInput.pVertexAttributeDescriptions    = vtxAttributes.data();
  }

  const VkPipelineInputAssemblyStateCreateInfo inputAssembly{.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
                                                             .topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST};

  const VkViewport viewport{.width    = static_cast<float>(m_colorImage.getWidth()),
                            .height   = static_cast<float>(m_colorImage.getHeight()),
                            .minDepth = 0.0f,
                            .maxDepth = 1.0f};

  const VkRect2D scissor{.extent = {m_colorImage.getWidth(), m_colorImage.getHeight()}};

  const VkPipelineViewportStateCreateInfo viewportInfo{.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
                                                       .viewportCount = 1,
                                                       .pViewports    = &viewport,
                                                       .scissorCount  = 1,
                                                       .pScissors     = &scissor};

  const VkPipelineRasterizationStateCreateInfo rasterization{
      .sType       = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
      .polygonMode = VK_POLYGON_MODE_FILL,
      .cullMode    = VkCullModeFlags(isDoubleSided ? VK_CULL_MODE_NONE : VK_CULL_MODE_BACK_BIT),
      .frontFace   = VK_FRONT_FACE_COUNTER_CLOCKWISE,
  };

  const VkPipelineMultisampleStateCreateInfo msaa{.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
                                                  .rasterizationSamples = static_cast<VkSampleCountFlagBits>(m_state.msaa)};

  VkPipelineDepthStencilStateCreateInfo depthStencilState{.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO,
                                                          .depthTestEnable = true,
                                                          .depthCompareOp  = VK_COMPARE_OP_LESS};
  std::array<VkPipelineColorBlendAttachmentState, 2> blendAttachments{};
  VkPipelineColorBlendStateCreateInfo blendInfo{.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
                                                .attachmentCount = 1,  // This can be modified below
                                                .pAttachments    = blendAttachments.data()};

  constexpr VkColorComponentFlags allBits =
      VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
  switch(blendMode)
  {
    case BlendMode::NONE:
      // Test and write to depth
      depthStencilState.depthWriteEnable = true;
      blendAttachments[0] = VkPipelineColorBlendAttachmentState{.blendEnable = VK_FALSE, .colorWriteMask = allBits};
      // Leave blending disabled
      break;
    case BlendMode::PREMULTIPLIED:
      // Test but don't write to depth
      depthStencilState.depthWriteEnable = false;
      blendAttachments[0]                = VkPipelineColorBlendAttachmentState{.blendEnable         = VK_TRUE,
                                                                               .srcColorBlendFactor = VK_BLEND_FACTOR_ONE,
                                                                               .dstColorBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA,
                                                                               .colorBlendOp        = VK_BLEND_OP_ADD,
                                                                               .srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE,
                                                                               .dstAlphaBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA,
                                                                               .colorWriteMask = allBits};
      break;
    case BlendMode::WEIGHTED_COLOR:
      // Test but don't write to depth
      depthStencilState.depthWriteEnable = false;
      blendInfo.attachmentCount          = 2;
      blendAttachments[0]                = VkPipelineColorBlendAttachmentState{.blendEnable         = VK_TRUE,
                                                                               .srcColorBlendFactor = VK_BLEND_FACTOR_ONE,
                                                                               .dstColorBlendFactor = VK_BLEND_FACTOR_ONE,
                                                                               .colorBlendOp        = VK_BLEND_OP_ADD,
                                                                               .srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE,
                                                                               .dstAlphaBlendFactor = VK_BLEND_FACTOR_ONE,
                                                                               .colorWriteMask      = allBits};
      blendAttachments[1]                = VkPipelineColorBlendAttachmentState{.blendEnable         = VK_TRUE,
                                                                               .srcColorBlendFactor = VK_BLEND_FACTOR_ZERO,
                                                                               .dstColorBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_COLOR,
                                                                               .colorBlendOp        = VK_BLEND_OP_ADD,
                                                                               .srcAlphaBlendFactor = VK_BLEND_FACTOR_ZERO,
                                                                               .dstAlphaBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA,
                                                                               .colorWriteMask = allBits};
      break;
    case BlendMode::WEIGHTED_COMPOSITE:
      // Test but don't write to depth
      depthStencilState.depthWriteEnable = false;
      blendAttachments[0]                = VkPipelineColorBlendAttachmentState{.blendEnable = VK_TRUE,
                                                                               .srcColorBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA,
                                                                               .dstColorBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA,
                                                                               .colorBlendOp        = VK_BLEND_OP_ADD,
                                                                               .srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA,
                                                                               .dstAlphaBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA,
                                                                               .colorWriteMask      = allBits};
      break;
    default:
      assert(!"Blend mode configuration not implemented!");
      break;
  }

  const VkGraphicsPipelineCreateInfo info{.sType               = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
                                          .stageCount          = uint32_t(stages.size()),
                                          .pStages             = stages.data(),
                                          .pVertexInputState   = &vertexInput,
                                          .pInputAssemblyState = &inputAssembly,
                                          .pViewportState      = &viewportInfo,
                                          .pRasterizationState = &rasterization,
                                          .pMultisampleState   = &msaa,
                                          .pDepthStencilState  = &depthStencilState,
                                          .pColorBlendState    = &blendInfo,
                                          .layout              = m_pipelineLayout,
                                          .renderPass          = renderPass,
                                          .subpass             = subpass};

  VkPipeline pipeline = VK_NULL_HANDLE;
  NVVK_CHECK(vkCreateGraphicsPipelines(m_app->getDevice(), VK_NULL_HANDLE, 1, &info, nullptr, &pipeline));
  nvvk::DebugUtil::getInstance().setObjectName(pipeline, debugName);

  return pipeline;
}

///////////////////////////////////////////////////////////////////////////////
// Main rendering logic                                                      //
///////////////////////////////////////////////////////////////////////////////

VkExtent2D Sample::getViewportSize() const
{
  const VkExtent2D rawSize = m_app->getViewportSize();
  return VkExtent2D{.width = std::max(1U, rawSize.width), .height = std::max(1U, rawSize.height)};
}

void Sample::updateUniformBuffer(uint32_t currentImage, double time)
{
  // TODO: This can be changed to all be push constants!
  const uint32_t width       = m_colorImage.getWidth();
  const uint32_t height      = m_colorImage.getHeight();
  const float    aspectRatio = static_cast<float>(width) / static_cast<float>(height);
  glm::mat4      projection  = glm::perspectiveRH_ZO(glm::radians(45.0f), aspectRatio, 0.01f, 50.0f);
  projection[1][1] *= -1;
  glm::mat4 view = m_cameraControl->getViewMatrix();

  m_sceneUbo.projViewMatrix             = projection * view;
  m_sceneUbo.viewMatrix                 = view;
  m_sceneUbo.viewMatrixInverseTranspose = glm::transpose(glm::inverse(view));

  m_sceneUbo.viewport = glm::ivec3(width, height, width * height);

  memcpy(m_uniformBuffers[currentImage].mapping, &m_sceneUbo, sizeof(m_sceneUbo));
  // TODO: Issue CPU -> GPU pipeline barrier
}

void Sample::copyOffscreenToBackBuffer(VkCommandBuffer cmd)
{
  // This function resolves + scales m_colorImage into m_viewportImage.
  // Because m_colorImage is generally a different format (B8G8R8A8_SRGB) than
  // m_viewportImage (R8G8B8A8) (which in turn is required by linear-space
  // rendering) and sometimes a different size xor has different MSAA
  // samples/pixel, the worst case (MSAA resolve + change of format) takes
  // two steps.
  //
  // Note that we could do this in one step, and further customize the filters
  // used, using a custom kernel.
  //
  // Finally, Vulkan allows us to access the swapchain images themselves.
  // However, while a previous version of this sample did that, with the change
  // to nvpro_core2, we now pass the image to Dear ImGui and tell it to draw
  // the image into a viewport pane using `ImGui::Image` in `onUIRender()`.
  //
  // As a result of the differences between MSAA resolve + downscaling, there
  // are a few cases to handle.
  // Here's a high-level node graph overview of this function:
  //
  //       MSAA?          Downsample?    Neither?
  //    m_colorImage     m_colorImage  m_colorImage
  //         |               |              |
  // vkCmdResolveImage  vkCmdBlitImage      |
  //         V               V              |
  //         m_downsampleImage  .-----------*
  //                 |          V
  //                vkCmdCopyImage (reinterpret data)
  //                 V
  //          m_viewportImage
  //                 |
  //    render Dear ImGui GUI (`onUIRender()`)
  //                 V
  //             Swapchain

  NVVK_DBG_SCOPE(cmd);
  auto section = m_profilerGPU.cmdFrameSection(cmd, __FUNCTION__);

  // Prepare to transfer from m_colorImage; check its initial state for soundness
  assert(m_colorImage.getLayout() == VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
  m_colorImage.transitionTo(cmd, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);

  // Tracks the image that will be passed to vkCmdCopyImage
  // These are the defaults if no resolve or downsample is required.
  VkImage       copySrcImage  = m_colorImage.image.image;
  VkImageLayout copySrcLayout = m_colorImage.getLayout();

  // If resolve or downsample required
  if(m_state.msaa != 1 || m_state.supersample != 1)
  {
    // Prepare to transfer data to m_downsampleImage
    m_downsampleImage.transitionTo(cmd, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);

    // MSAA branch
    if(m_state.msaa != 1)
    {
      // Resolve the MSAA image m_colorImage to m_downsampleImage
      const VkImageResolve region{.srcSubresource = {.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT, .layerCount = 1},
                                  .dstSubresource = region.srcSubresource,
                                  .extent = {.width = m_colorImage.getWidth(), .height = m_colorImage.getHeight(), .depth = 1}};

      vkCmdResolveImage(cmd,                            // Command buffer
                        m_colorImage.image.image,       // Source image
                        m_colorImage.getLayout(),       // Source image layout
                        m_downsampleImage.image.image,  // Destination image
                        m_downsampleImage.getLayout(),  // Destination image layout
                        1,                              // Number of regions
                        &region);                       // Regions
    }
    else
    {
      // Downsample m_colorImage to m_downsampleTargeImage
      const VkImageBlit region = {
          .srcSubresource = {.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT, .layerCount = 1},
          .srcOffsets     = {{0, 0, 0}, {int32_t(m_colorImage.getWidth()), int32_t(m_colorImage.getHeight()), 1}},
          .dstSubresource = region.srcSubresource,
          .dstOffsets = {{0, 0, 0}, {int32_t(m_downsampleImage.getWidth()), int32_t(m_downsampleImage.getHeight()), 1}}};

      vkCmdBlitImage(cmd,                            // Command buffer
                     m_colorImage.image.image,       // Source image
                     m_colorImage.getLayout(),       // Source image
                     m_downsampleImage.image.image,  // Destination image
                     m_downsampleImage.getLayout(),  // Destination image layout
                     1,                              // Number of regions
                     &region,                        // Regions
                     VK_FILTER_LINEAR);              // Use tent filtering
    }

    // Prepare to transfer data from m_downsampleImage, and set copySrcImage and copySrcLayout.
    m_downsampleImage.transitionTo(cmd, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);
    copySrcImage  = m_downsampleImage.image.image;
    copySrcLayout = m_downsampleImage.getLayout();
  }

  // Prepare to transfer data to m_viewportImage
  // General -> VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL
  nvvk::cmdImageMemoryBarrier(cmd, {.image            = m_viewportImage.getColorImage(),
                                    .oldLayout        = VK_IMAGE_LAYOUT_GENERAL,
                                    .newLayout        = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                                    .subresourceRange = {.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
                                                         .levelCount = VK_REMAINING_MIP_LEVELS,
                                                         .layerCount = VK_REMAINING_ARRAY_LAYERS}});

  // Now, we want to copy data from copySrcImage to m_viewportImage instead of blitting it, since blitting will try
  // to convert the sRGB data and store it in linear format, which isn't what we want.
  {
    const VkImageCopy region{.srcSubresource = {.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT, .layerCount = 1},
                             .dstSubresource = region.srcSubresource,
                             .extent         = {m_viewportImage.getSize().width, m_viewportImage.getSize().height, 1}};
    vkCmdCopyImage(cmd,                                   // Command buffer
                   copySrcImage,                          // Source image
                   copySrcLayout,                         // Source image layout
                   m_viewportImage.getColorImage(),       // Destination image
                   VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,  // Destination image layout
                   1,                                     // Number of regions
                   &region);                              // Regions
  }

  // Transition m_viewportImage to VK_IMAGE_LAYOUT_GENERAL so that ImGui::Image() can use it.
  nvvk::cmdImageMemoryBarrier(cmd, {.image            = m_viewportImage.getColorImage(),
                                    .oldLayout        = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                                    .newLayout        = VK_IMAGE_LAYOUT_GENERAL,
                                    .subresourceRange = {.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
                                                         .levelCount = VK_REMAINING_MIP_LEVELS,
                                                         .layerCount = VK_REMAINING_ARRAY_LAYERS}});

  // Reset the layout of m_colorImage.
  m_colorImage.transitionTo(cmd, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
}

int main(int argc, const char** argv)
{
  // Vulkan extensions
  // The extension below is optional - there are algorithms we can use if we have it, but
  // if the device doesn't support it, we don't allow the user to select those algorithms.
  VkPhysicalDeviceFragmentShaderInterlockFeaturesEXT fragmentShaderInterlockFeatures{
      .sType                              = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FRAGMENT_SHADER_INTERLOCK_FEATURES_EXT,
      .fragmentShaderSampleInterlock      = VK_TRUE,
      .fragmentShaderPixelInterlock       = VK_TRUE,
      .fragmentShaderShadingRateInterlock = VK_FALSE  // (we don't need this)
  };

  nvvk::ContextInitInfo vkSetup{
      .instanceExtensions = {VK_EXT_DEBUG_UTILS_EXTENSION_NAME},
      .deviceExtensions   = {nvvk::ExtensionInfo{.extensionName = VK_EXT_DESCRIPTOR_INDEXING_EXTENSION_NAME},
                             nvvk::ExtensionInfo{.extensionName = VK_EXT_POST_DEPTH_COVERAGE_EXTENSION_NAME},
                             nvvk::ExtensionInfo{.extensionName = VK_KHR_SAMPLER_MIRROR_CLAMP_TO_EDGE_EXTENSION_NAME},
                             nvvk::ExtensionInfo{.extensionName = VK_EXT_FRAGMENT_SHADER_INTERLOCK_EXTENSION_NAME,
                                                 .feature       = &fragmentShaderInterlockFeatures,
                                                 .required      = false}}};
  // TODO: Headless mode
  if(true)
  {
    nvvk::addSurfaceExtensions(vkSetup.instanceExtensions);
    vkSetup.deviceExtensions.emplace_back(VK_KHR_SWAPCHAIN_EXTENSION_NAME);
  }

  nvvk::Context vkContext;
  NVVK_FAIL_RETURN(vkContext.init(vkSetup));

  // Window + main loop setup
  nvapp::ApplicationCreateInfo appInfo{
      .name           = TARGET_NAME,
      .instance       = vkContext.getInstance(),
      .device         = vkContext.getDevice(),
      .physicalDevice = vkContext.getPhysicalDevice(),
      .queues         = vkContext.getQueueInfos(),
      .windowSize     = {1600, 1024},
#ifdef NDEBUG
      .vSync = false,
#else
      .vSync = true,
#endif
      // This sets up the dock positions for the menus
      .dockSetup =
          [](ImGuiID viewportID) {
            ImGuiID settingsID = ImGui::DockBuilderSplitNode(viewportID, ImGuiDir_Left, 0.2f, nullptr, nullptr);
            ImGui::DockBuilderDockWindow(kUiPaneSettingsName, settingsID);
            ImGuiID profilerID = ImGui::DockBuilderSplitNode(settingsID, ImGuiDir_Down, 0.25f, nullptr, nullptr);
            ImGui::DockBuilderDockWindow(kUiPaneProfilerName, profilerID);
          },
  };
  nvapp::Application app;
  app.init(appInfo);

  // Create the sample element and attach it to the GUI.
  // It's easiest to pass the entire Context here, so that we can look up
  // whether we got optional extensions in its `deviceExtensions` table.
  app.addElement(std::make_shared<Sample>(&vkContext));
  // Add an element that automatically updates the title with the current
  // size and FPS.
  app.addElement(std::make_shared<nvapp::ElementDefaultWindowTitle>());

  // Main loop
  app.run();

  // Teardown
  app.deinit();
  vkContext.deinit();

  return EXIT_SUCCESS;
}
