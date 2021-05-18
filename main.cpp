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

#if defined(_WIN32)
// Include Windows before GLFW3 to fix some errors with std::min and std::max
#define WIN32_LEAN_AND_MEAN
#ifndef NOMINMAX
#define NOMINMAX
#endif  // #ifndef NOMINMAX
#include <Windows.h>
#endif

#ifndef GLFW_INCLUDE_VULKAN
#define GLFW_INCLUDE_VULKAN
#endif
#include <GLFW/glfw3.h>

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <random>
#include <set>
#include <stdexcept>
#include <vector>

#include <backends/imgui_vk_extra.h>
#include <imgui/imgui_helper.h>

#include <nvpwindow.hpp>

#include <nvh/cameracontrol.hpp>
#include <nvh/fileoperations.hpp>
#include <nvh/misc.hpp>
#include <nvh/nvprint.hpp>
#include <nvh/timesampler.hpp>

#include <nvvk/commands_vk.hpp>
#include <nvvk/descriptorsets_vk.hpp>
#include <nvvk/error_vk.hpp>
#include <nvvk/extensions_vk.hpp>
#include <nvvk/images_vk.hpp>
#include <nvvk/memorymanagement_vk.hpp>
#include <nvvk/pipeline_vk.hpp>
#include <nvvk/profiler_vk.hpp>
#include <nvvk/shadermodulemanager_vk.hpp>
#include <nvvk/shaders_vk.hpp>
#include <nvvk/swapchain_vk.hpp>

#include "oit.h"

// Application constants
const int   GRID_SIZE    = 16;
const float GLOBAL_SCALE = 8.0f;

///////////////////////////////////////////////////////////////////////////////
// Callbacks                                                                 //
///////////////////////////////////////////////////////////////////////////////

void Sample::resize(int width, int height)
{
  assert(width == m_windowState.m_swapSize[0]);
  assert(height == m_windowState.m_swapSize[1]);
  updateRendererImmediate(true, false);
}

bool Sample::mouse_pos(int x, int y)
{
  return ImGuiH::mouse_pos(x, y);
}

bool Sample::mouse_button(int button, int action)
{
  return ImGuiH::mouse_button(button, action);
}

bool Sample::mouse_wheel(int wheel)
{
  return ImGuiH::mouse_wheel(wheel);
}

bool Sample::key_char(int key)
{
  return ImGuiH::key_char(key);
}

bool Sample::key_button(int button, int action, int mods)
{
  return ImGuiH::key_button(button, action, mods);
}

///////////////////////////////////////////////////////////////////////////////
// Object Creation, Destruction, and Recreation                              //
///////////////////////////////////////////////////////////////////////////////

bool Sample::begin()
{
  m_profilerPrint = true;
  m_timeInTitle   = true;

  // Initialize Dear ImGui (we'll call InitVK later)
  ImGuiH::Init(m_windowState.m_winSize[0], m_windowState.m_winSize[1], this);
  ImGui::GetIO().IniFilename = nullptr;  // Don't create a .ini file for storing data across application launches
  // Initialize Dear ImGui's Vulkan renderer:
  m_debug.setup(m_context);
  createGUIRenderPass();
  ImGui::InitVK(m_context, m_context.m_physicalDevice, m_context.m_queueGCT, m_context.m_queueGCT.familyIndex, m_renderPassGUI, 0);

  // Initialize all Vulkan components that will be constant throughout the application lifecycle.
  // Components that can change are handled by updateRendererFromState.
  m_ringFences.init(m_context);
  m_ringCmdPool.init(m_context, m_context.m_queueGCT.familyIndex, VK_COMMAND_POOL_CREATE_TRANSIENT_BIT);
  m_submission.init(m_context.m_queueGCT.queue);

  createTextureSampler();
  m_allocatorDma.init(m_context.m_device, m_context.getPhysicalDevices().front());
  // Configure shader system (note that this also creates shader modules as we add them)
  {
    // Initialize shader system (this keeps track of shaders so that you can reload all of them at once):
    m_shaderModuleManager.init(m_context);
    // Add search paths for files and includes
    m_shaderModuleManager.addDirectory("GLSL_" PROJECT_NAME);  // For when running in the install directory
    m_shaderModuleManager.addDirectory(".");
    m_shaderModuleManager.addDirectory(NVPSystem::exePath() + PROJECT_RELDIRECTORY);
    m_shaderModuleManager.addDirectory(NVPSystem::exePath() + PROJECT_RELDIRECTORY + "..");
    m_shaderModuleManager.addDirectory("..");     // for when working directory in Debug is $(ProjectDir)
    m_shaderModuleManager.addDirectory("../..");  // for when using $(TargetDir)
    m_shaderModuleManager.addDirectory("../shipped/" PROJECT_NAME);  // For when running from the bin_x64 directory on Linux
    m_shaderModuleManager.addDirectory("../../shipped/" PROJECT_NAME);     // for when using $(TargetDir)
    m_shaderModuleManager.addDirectory("../../../shipped/" PROJECT_NAME);  // for when using $(TargetDir) and build_all
    // We have to manually set up paths to files we could include.
    m_shaderModuleManager.registerInclude("common.h");
    m_shaderModuleManager.registerInclude("oitColorDepthDefines.glsl");
    m_shaderModuleManager.registerInclude("oitCompositeDefines.glsl");
    m_shaderModuleManager.registerInclude("shaderCommon.glsl");
  }

  // Call updateRendererImmediate to set up the rest of the renderer with the initial swapchain size:
  {
    updateRendererImmediate(true, true);
  }

  // Register enumerations with the Dear ImGui registry
  {
    m_imGuiRegistry.enumAdd(GUI_ALGORITHM, OIT_SIMPLE, "simple");
    m_imGuiRegistry.enumAdd(GUI_ALGORITHM, OIT_LINKEDLIST, "linkedlist");
    m_imGuiRegistry.enumAdd(GUI_ALGORITHM, OIT_LOOP, "loop32 two pass");

    if(m_context.hasDeviceExtension(VK_KHR_SHADER_ATOMIC_INT64_EXTENSION_NAME))
    {
      m_imGuiRegistry.enumAdd(GUI_ALGORITHM, OIT_LOOP64, "loop64");
    }
    m_imGuiRegistry.enumAdd(GUI_ALGORITHM, OIT_SPINLOCK, "spinlock");
    if(m_context.hasDeviceExtension(VK_EXT_FRAGMENT_SHADER_INTERLOCK_EXTENSION_NAME))
    {
      m_imGuiRegistry.enumAdd(GUI_ALGORITHM, OIT_INTERLOCK, "interlock");
    }
    m_imGuiRegistry.enumAdd(GUI_ALGORITHM, OIT_WEIGHTED, "weighted blend");

    m_imGuiRegistry.enumAdd(GUI_OITSAMPLES, 1, "1");
    m_imGuiRegistry.enumAdd(GUI_OITSAMPLES, 2, "2");
    m_imGuiRegistry.enumAdd(GUI_OITSAMPLES, 4, "4");
    m_imGuiRegistry.enumAdd(GUI_OITSAMPLES, 8, "8");
    m_imGuiRegistry.enumAdd(GUI_OITSAMPLES, 16, "16");
    m_imGuiRegistry.enumAdd(GUI_OITSAMPLES, 32, "32");

    m_imGuiRegistry.enumAdd(GUI_AA, AA_NONE, "none");
    m_imGuiRegistry.enumAdd(GUI_AA, AA_MSAA_4X, "msaa 4x pixel-shading");
    m_imGuiRegistry.enumAdd(GUI_AA, AA_SSAA_4X, "msaa 4x sample-shading");
    m_imGuiRegistry.enumAdd(GUI_AA, AA_SUPER_4X, "super 4x");
    m_imGuiRegistry.enumAdd(GUI_AA, AA_MSAA_8X, "msaa 8x pixel-shading");
    m_imGuiRegistry.enumAdd(GUI_AA, AA_SSAA_8X, "msaa 8x sample-shading");
  }

  // Initialize camera
  {
    m_cameraControl.m_sceneOrbit     = nvmath::vec3(0.0f);
    m_cameraControl.m_sceneDimension = static_cast<float>(GRID_SIZE) * 0.25f;
    m_cameraControl.m_viewMatrix =
        nvmath::look_at(m_cameraControl.m_sceneOrbit - (nvmath::vec3(0, 0, -0.6f) * m_cameraControl.m_sceneDimension * 5.0f),
                        m_cameraControl.m_sceneOrbit, vec3(0.0f, 1.0f, 0.0f));
  }

  // Initialize the UBO
  m_sceneUbo.alphaMin   = 0.2f;
  m_sceneUbo.alphaWidth = 0.3f;

  m_frame     = 0;
  m_lastState = m_state;

  return true;  // Initialization succeeded
}

void Sample::updateRendererImmediate(bool swapchainSizeChanged, bool forceRebuildAll)
{
  VkCommandBuffer cmd = createTempCmdBuffer();
  cmdUpdateRendererFromState(cmd, swapchainSizeChanged, forceRebuildAll);
  vkEndCommandBuffer(cmd);

  m_submission.enqueue(cmd);
  submissionExecute();
  vkDeviceWaitIdle(m_context);
  m_ringFences.reset();
  m_ringCmdPool.reset();
}

void Sample::cmdUpdateRendererFromState(VkCommandBuffer cmdBuffer, bool swapchainSizeChanged, bool forceRebuildAll)
{
  m_state.recomputeAntialiasingSettings();

  // Determine what needs to be rebuilt
  swapchainSizeChanged |= forceRebuildAll;

  const bool vsyncChanged = (m_lastVsync != getVsync()) || forceRebuildAll;

  const bool shadersNeedUpdate = (m_state.algorithm != m_lastState.algorithm)             //
                                 || (m_state.oitLayers != m_lastState.oitLayers)          //
                                 || (m_state.tailBlend != m_lastState.tailBlend)          //
                                 || (m_state.msaa != m_lastState.msaa)                    //
                                 || (m_state.sampleShading != m_lastState.sampleShading)  //
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
                                                    || vsyncChanged   //
                                                    || forceRebuildAll;

  const bool renderPassesNeedReinit = (m_state.msaa != m_lastState.msaa)  //
                                      || forceRebuildAll;

  const bool pipelinesNeedReinit = (m_state.algorithm != m_lastState.algorithm)  //
                                   || shadersNeedUpdate || imagesNeedReinit;

  const bool anythingChanged = shadersNeedUpdate || sceneNeedsReinit || imagesNeedReinit || descriptorSetsNeedReinit
                               || framebuffersAndDescriptorsNeedReinit || renderPassesNeedReinit;

  if(anythingChanged)
  {
    vkDeviceWaitIdle(m_context);
    LOGI("framebuffer: %d x %d (%d msaa)\n", m_windowState.m_swapSize[0], m_windowState.m_swapSize[1], m_state.msaa);

    if(vsyncChanged || swapchainSizeChanged)
    {
      m_swapChain.cmdUpdateBarriers(cmdBuffer);
      createUniformBuffers();
    }

    if(sceneNeedsReinit)
    {
      initScene(cmdBuffer);
    }

    if(imagesNeedReinit)
    {
      createFrameImages(cmdBuffer);
    }

    if(descriptorSetsNeedReinit)
    {
      createDescriptorSets();
    }

    if(renderPassesNeedReinit)
    {
      createNonGUIRenderPasses();
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

    setUpViewportsAndScissors();
    m_lastVsync = getVsync();
  }
}

void Sample::end()
{
  vkDeviceWaitIdle(m_context);
  m_profilerVK.deinit();

  ImGui::ShutdownVK();
  ImGui::DestroyContext();

  // From updateRendererFromState
  destroyGraphicsPipelines();
  m_shaderModuleManager.deinit();
  destroyFramebuffers();
  destroyNonGUIRenderPasses();
  destroyGUIRenderPass();
  destroyDescriptorSets();
  destroyFrameImages();
  destroyScene();
  destroyUniformBuffers();
  // From begin
  m_allocatorDma.deinit();

  destroyTextureSampler();
  m_ringCmdPool.deinit();
  m_ringFences.deinit();
}

void Sample::destroyTextureSampler()
{
  vkDestroySampler(m_context, m_pointSampler, nullptr);
}

void Sample::createTextureSampler()
{
  // Create a point sampler using base Vulkan
  VkSamplerCreateInfo samplerInfo     = nvvk::make<VkSamplerCreateInfo>();
  samplerInfo.magFilter               = VK_FILTER_LINEAR;
  samplerInfo.minFilter               = VK_FILTER_LINEAR;
  samplerInfo.addressModeU            = VK_SAMPLER_ADDRESS_MODE_REPEAT;
  samplerInfo.addressModeV            = VK_SAMPLER_ADDRESS_MODE_REPEAT;
  samplerInfo.addressModeW            = VK_SAMPLER_ADDRESS_MODE_REPEAT;
  samplerInfo.anisotropyEnable        = VK_FALSE;
  samplerInfo.borderColor             = VK_BORDER_COLOR_INT_OPAQUE_BLACK;
  samplerInfo.unnormalizedCoordinates = VK_FALSE;
  samplerInfo.compareEnable           = VK_FALSE;
  samplerInfo.mipmapMode              = VK_SAMPLER_MIPMAP_MODE_NEAREST;

  NVVK_CHECK(vkCreateSampler(m_context, &samplerInfo, nullptr, &m_pointSampler));
}

void Sample::destroyUniformBuffers()
{
  for(nvvk::Buffer& uniformBuffer : m_uniformBuffers)
  {
    m_allocatorDma.destroy(uniformBuffer);
  }
}

void Sample::createUniformBuffers()
{
  destroyUniformBuffers();

  VkDeviceSize bufferSize = sizeof(SceneData);

  m_uniformBuffers.resize(m_swapChain.getImageCount());

  for(uint32_t i = 0; i < m_swapChain.getImageCount(); i++)
  {
    m_uniformBuffers[i] = m_allocatorDma.createBuffer(bufferSize,                          // Buffer size
                                                      VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,  // Usage
                                                      VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT  // Memory flags
    );
  }
}

void Sample::destroyScene()
{
  m_allocatorDma.destroy(m_indexBuffer);
  m_allocatorDma.destroy(m_vertexBuffer);
}

void Sample::initScene(VkCommandBuffer commandBuffer)
{
  destroyScene();
  // A Mesh consists of vectors of vertices, triangle list indices, and lines.
  // It assumes that its type contains variables, at least, each vertex's position, normal, and color.
  // (We'll ignore lines when converting this to a vertex and index buffer.)
  nvh::geometry::Mesh<Vertex> completeMesh;

  // We'll use C++11-style random number generation here, but you could also do this
  // with rand() and srand().
  std::default_random_engine            rnd(3625);  // Fixed seed
  std::uniform_real_distribution<float> uniformDist;

  for(uint32_t i = 0; i < m_state.numObjects; i++)
  {
    // Generate a random position in [-GLOBAL_SCALE/2, GLOBAL_SCALE/2)^3
    nvmath::vec3 center(uniformDist(rnd), uniformDist(rnd), uniformDist(rnd));
    center = (center - nvmath::vec3(0.5)) * GLOBAL_SCALE;

    // Generate a random radius
    float radius = GLOBAL_SCALE * 0.9f / GRID_SIZE;
    radius *= uniformDist(rnd) * m_state.scaleWidth + m_state.scaleMin;

    // Our vectors are vertical, so this represents a scale followed by a translation:
    nvmath::mat4 matrix = nvmath::translation_mat4(center) * nvmath::scale_mat4(nvmath::vec3(radius));

    // Add a sphere to the complete mesh, and then color it:
    const uint32_t vtxStart = completeMesh.getVerticesCount();  // First vertex to color

    nvh::geometry::Sphere<Vertex>::add(completeMesh, matrix, m_state.subdiv * 2, m_state.subdiv);

    if(i == 0)
    {
      m_objectTriangleIndices = completeMesh.getTriangleIndicesCount();
    }

    // Color in unpremultiplied linear space
    nvmath::vec4 color(uniformDist(rnd), uniformDist(rnd), uniformDist(rnd), uniformDist(rnd));
    color.x *= color.x;
    color.y *= color.y;
    color.z *= color.z;
    uint32_t vtxEnd = completeMesh.getVerticesCount();
    for(uint32_t v = vtxStart; v < vtxEnd; v++)
    {
      completeMesh.m_vertices[v].color = color;
    }
  }

  // Count the total number of triangle indices
  m_sceneTriangleIndices = completeMesh.getTriangleIndicesCount();

  // Create the vertex and index buffers and synchronously upload them to the
  // GPU, waiting for them to finish uploading. Note that applications may wish
  // to implement asynchronous uploads, which you can see how to do in the
  // vk_async_resources sample.

  nvvk::StagingMemoryManager scopedTransfer(m_allocatorDma.getMemoryAllocator());
  {
    // When this goes out of scope, it'll synchronously perform all of the copy operations.
    // 'scopedTransfer' can then safely go out of scope after it.
    nvvk::ScopeCommandBuffer cmd(m_context, m_context.m_queueT, m_context.m_queueT);

    // Create vertex buffer
    VkDeviceSize vtxBufferSize = static_cast<VkDeviceSize>(completeMesh.getVerticesSize());
    m_vertexBuffer             = m_allocatorDma.createBuffer(vtxBufferSize, VK_BUFFER_USAGE_VERTEX_BUFFER_BIT);
    scopedTransfer.cmdToBuffer(cmd, m_vertexBuffer.buffer, 0, vtxBufferSize, completeMesh.m_vertices.data());
    m_debug.setObjectName(m_vertexBuffer.buffer, "m_vertexBuffer");

    VkDeviceSize idxBufferSize = static_cast<VkDeviceSize>(completeMesh.getTriangleIndicesSize());
    m_indexBuffer              = m_allocatorDma.createBuffer(idxBufferSize, VK_BUFFER_USAGE_INDEX_BUFFER_BIT);
    scopedTransfer.cmdToBuffer(cmd, m_indexBuffer.buffer, 0, idxBufferSize, completeMesh.m_indicesTriangles.data());
    m_debug.setObjectName(m_indexBuffer.buffer, "m_indexBuffer");
  }
}

void Sample::destroyFramebuffers()
{
  vkDestroyFramebuffer(m_context, m_mainColorDepthFramebuffer, nullptr);
  m_mainColorDepthFramebuffer = nullptr;

  vkDestroyFramebuffer(m_context, m_guiFramebuffer, nullptr);
  m_guiFramebuffer = nullptr;

  if(m_weightedFramebuffer != nullptr)
  {
    vkDestroyFramebuffer(m_context, m_weightedFramebuffer, nullptr);
    m_weightedFramebuffer = nullptr;
  }
}

void Sample::createFramebuffers()
{
  destroyFramebuffers();

  // Color + depth offscreen framebuffer
  {
    std::array<VkImageView, 2> attachments = {m_colorImage.view, m_depthImage.view};
    VkFramebufferCreateInfo    fbInfo      = nvvk::make<VkFramebufferCreateInfo>();
    fbInfo.renderPass                      = m_renderPassColorDepthClear;
    fbInfo.attachmentCount                 = static_cast<uint32_t>(attachments.size());
    fbInfo.pAttachments                    = attachments.data();
    fbInfo.width                           = m_colorImage.c_width;
    fbInfo.height                          = m_colorImage.c_height;
    fbInfo.layers                          = 1;

    NVVK_CHECK(vkCreateFramebuffer(m_context, &fbInfo, NULL, &m_mainColorDepthFramebuffer));

    m_debug.setObjectName(m_mainColorDepthFramebuffer, "m_mainColorDepthFramebuffer");
  }

  // Weighted color + weighted reveal framebuffer (for Weighted, Blended
  // Order-Independent Transparency). See the render pass description for more info.
  if(m_state.algorithm == OIT_WEIGHTED)
  {
    std::array<VkImageView, 4> attachments = {m_oitWeightedColorImage.view,   //
                                              m_oitWeightedRevealImage.view,  //
                                              m_colorImage.view,              //
                                              m_depthImage.view};

    VkFramebufferCreateInfo framebufferInfo = nvvk::make<VkFramebufferCreateInfo>();
    framebufferInfo.renderPass              = m_renderPassWeighted;
    framebufferInfo.attachmentCount         = static_cast<uint32_t>(attachments.size());
    framebufferInfo.pAttachments            = attachments.data();
    framebufferInfo.width                   = m_oitWeightedColorImage.c_width;
    framebufferInfo.height                  = m_oitWeightedColorImage.c_height;
    framebufferInfo.layers                  = 1;

    NVVK_CHECK(vkCreateFramebuffer(m_context, &framebufferInfo, nullptr, &m_weightedFramebuffer));

    m_debug.setObjectName(m_weightedFramebuffer, "m_weightedColorRevealFramebuffer");
  }

  // ui related
  {
    VkImageView uiTarget = m_guiCompositeImage.view;

    // Create framebuffers
    VkImageView bindInfos[1];
    bindInfos[0] = uiTarget;

    VkFramebufferCreateInfo fbInfo = {VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO};
    fbInfo.attachmentCount         = NV_ARRAY_SIZE(bindInfos);
    fbInfo.pAttachments            = bindInfos;
    fbInfo.width                   = m_windowState.m_swapSize[0];
    fbInfo.height                  = m_windowState.m_swapSize[1];
    fbInfo.layers                  = 1;

    fbInfo.renderPass = m_renderPassGUI;
    NVVK_CHECK(vkCreateFramebuffer(m_context, &fbInfo, NULL, &m_guiFramebuffer));
  }
}

void Sample::setUpViewportsAndScissors()
{
  m_scissorGUI               = {0};  // Zero-initialize
  m_scissorGUI.extent.width  = m_windowState.m_swapSize[0];
  m_scissorGUI.extent.height = m_windowState.m_swapSize[1];

  m_viewportGUI          = {0};  // Zero-initialize
  m_viewportGUI.width    = static_cast<float>(m_scissorGUI.extent.width);
  m_viewportGUI.height   = static_cast<float>(m_scissorGUI.extent.height);
  m_viewportGUI.minDepth = 0.0f;
  m_viewportGUI.maxDepth = 1.0f;
}

void Sample::createOrReloadShaderModule(nvvk::ShaderModuleID& shaderModule,
                                        VkShaderStageFlags    shaderStage,
                                        const std::string&    filename,
                                        const std::string&    prepend)
{
  if(shaderModule.isValid())
  {
    // Reload and recompile this module from source.
    m_shaderModuleManager.reloadModule(shaderModule);
  }
  else
  {
    // Register and compile the shader module with the shader module manager.
    shaderModule = m_shaderModuleManager.createShaderModule(shaderStage, filename, prepend);
  }
  assert(shaderModule.isValid());
#ifdef _DEBUG
  std::string generatedShaderName = filename + " " + prepend;
  m_debug.setObjectName(m_shaderModuleManager.get(shaderModule), generatedShaderName.c_str());
#endif  // #if _DEBUG
}

void Sample::destroyGraphicsPipeline(VkPipeline& pipeline)
{
  if(pipeline != nullptr)
  {
    vkDestroyPipeline(m_context, pipeline, nullptr);
    pipeline = nullptr;
  }
}

VkPipeline Sample::createGraphicsPipeline(const nvvk::ShaderModuleID& vertShaderModuleID,
                                          const nvvk::ShaderModuleID& fragShaderModuleID,
                                          BlendMode                   blendMode,
                                          bool                        usesVertexInput,
                                          bool                        isDoubleSided,
                                          VkRenderPass                renderPass,
                                          uint32_t                    subpass)
{
  VkShaderModule vertShaderModule = m_shaderModuleManager.get(vertShaderModuleID);
  VkShaderModule fragShaderModule = m_shaderModuleManager.get(fragShaderModuleID);

  nvvk::GraphicsPipelineGeneratorCombined pipelineState(m_context, m_descriptorInfo.getPipeLayout(), renderPass);

  pipelineState.addShader(vertShaderModule,           // Shader module
                          VK_SHADER_STAGE_VERTEX_BIT  // Stage
  );

  pipelineState.addShader(fragShaderModule,             // Shader module
                          VK_SHADER_STAGE_FRAGMENT_BIT  // Stage
  );

  if(usesVertexInput)
  {
    // Vertex input layout
    VkVertexInputBindingDescription bindingDescription = Vertex::getBindingDescription();
    auto                            attributes         = Vertex::getAttributeDescriptions();

    pipelineState.addBindingDescription(bindingDescription);
    for(const auto& attribute : attributes)
    {
      pipelineState.addAttributeDescription(attribute);
    }
  }

  pipelineState.inputAssemblyState.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;

  VkViewport viewport = {};
  viewport.x          = 0.0f;
  viewport.y          = 0.0f;
  viewport.width      = static_cast<float>(m_colorImage.c_width);
  viewport.height     = static_cast<float>(m_colorImage.c_height);
  viewport.minDepth   = 0.0f;
  viewport.maxDepth   = 1.0f;

  VkRect2D scissor      = {};
  scissor.offset        = {0, 0};
  scissor.extent.width  = m_colorImage.c_width;
  scissor.extent.height = m_colorImage.c_height;

  pipelineState.clearDynamicStateEnables();
  pipelineState.setViewportsCount(1);
  pipelineState.setViewport(0, viewport);
  pipelineState.setScissorsCount(1);
  pipelineState.setScissor(0, scissor);

  // Enable backface culling
  pipelineState.rasterizationState.cullMode        = (isDoubleSided ? VK_CULL_MODE_NONE : VK_CULL_MODE_BACK_BIT);
  pipelineState.rasterizationState.frontFace       = VK_FRONT_FACE_COUNTER_CLOCKWISE;
  pipelineState.rasterizationState.polygonMode     = VK_POLYGON_MODE_FILL;
  pipelineState.rasterizationState.lineWidth       = 1.f;
  pipelineState.rasterizationState.depthBiasEnable = false;
  pipelineState.rasterizationState.depthBiasConstantFactor = 0.f;
  pipelineState.rasterizationState.depthBiasSlopeFactor    = 0.f;

  pipelineState.multisampleState.rasterizationSamples = (static_cast<VkSampleCountFlagBits>(m_state.msaa));

  pipelineState.depthStencilState.depthBoundsTestEnable = false;


  const VkCompareOp           compareOp = VK_COMPARE_OP_LESS;
  const VkColorComponentFlags allBits =
      VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
  switch(blendMode)
  {
    case BlendMode::NONE:
      // Test and write to depth
      pipelineState.depthStencilState.depthTestEnable  = true;
      pipelineState.depthStencilState.depthWriteEnable = true;
      pipelineState.depthStencilState.depthCompareOp   = compareOp;
      pipelineState.setBlendAttachmentState(0,  // Attachment
                                            nvvk::GraphicsPipelineState::makePipelineColorBlendAttachmentState());  // Disable blending
      break;
    case BlendMode::PREMULTIPLIED:
      // Test but don't write to depth
      pipelineState.depthStencilState.depthTestEnable  = true;
      pipelineState.depthStencilState.depthWriteEnable = false;
      pipelineState.depthStencilState.depthCompareOp   = compareOp;
      pipelineState.setBlendAttachmentState(0,  // Attachment
                                            nvvk::GraphicsPipelineState::makePipelineColorBlendAttachmentState(
                                                allBits, VK_TRUE,                     //
                                                VK_BLEND_FACTOR_ONE,                  // Source color blend factor
                                                VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA,  // Destination color blend factor
                                                VK_BLEND_OP_ADD,                      // Color blend operation
                                                VK_BLEND_FACTOR_ONE,                  // Source alpha blend factor
                                                VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA,  // Destination alpha blend factor
                                                VK_BLEND_OP_ADD));                    // Alpha blend operation
      break;
    case BlendMode::WEIGHTED_COLOR:
      // Test but don't write to depth
      pipelineState.depthStencilState.depthTestEnable  = true;
      pipelineState.depthStencilState.depthWriteEnable = false;
      pipelineState.depthStencilState.depthCompareOp   = compareOp;
      pipelineState.setBlendAttachmentCount(2);
      pipelineState.setBlendAttachmentState(0,  // Attachment
                                            nvvk::GraphicsPipelineState::makePipelineColorBlendAttachmentState(
                                                allBits, VK_TRUE,     //
                                                VK_BLEND_FACTOR_ONE,  // Source color blend factor
                                                VK_BLEND_FACTOR_ONE,  // Destination color blend factor
                                                VK_BLEND_OP_ADD,      // Color blend operation
                                                VK_BLEND_FACTOR_ONE,  // Source alpha blend factor
                                                VK_BLEND_FACTOR_ONE,  // Destination alpha blend factor
                                                VK_BLEND_OP_ADD));    // Alpha blend operation
      pipelineState.setBlendAttachmentState(1,                        // Attachment
                                            nvvk::GraphicsPipelineState::makePipelineColorBlendAttachmentState(
                                                allBits, VK_TRUE,                     //
                                                VK_BLEND_FACTOR_ZERO,                 // Source color blend factor
                                                VK_BLEND_FACTOR_ONE_MINUS_SRC_COLOR,  // Destination color blend factor
                                                VK_BLEND_OP_ADD,                      // Color blend operation
                                                VK_BLEND_FACTOR_ZERO,                 // Source alpha blend factor
                                                VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA,  // Destination alpha blend factor
                                                VK_BLEND_OP_ADD));                    // Alpha blend operation
      break;
    case BlendMode::WEIGHTED_COMPOSITE:
      // Test but don't write to depth
      pipelineState.depthStencilState.depthTestEnable  = true;
      pipelineState.depthStencilState.depthWriteEnable = false;
      pipelineState.depthStencilState.depthCompareOp   = compareOp;
      pipelineState.setBlendAttachmentState(0,  // Attachment
                                            nvvk::GraphicsPipelineState::makePipelineColorBlendAttachmentState(
                                                allBits, VK_TRUE,                     //
                                                VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA,  // Source color blend factor
                                                VK_BLEND_FACTOR_SRC_ALPHA,            // Destination color blend factor
                                                VK_BLEND_OP_ADD,                      // Color blend operation
                                                VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA,  // Source alpha blend factor
                                                VK_BLEND_FACTOR_SRC_ALPHA,            // Destination alpha blend factor
                                                VK_BLEND_OP_ADD));                    // Alpha blend operation
      break;
    default:
      assert(!"Blend mode configuration not implemented!");
      break;
  }

  pipelineState.setRenderPass(renderPass);
  pipelineState.createInfo.subpass = subpass;

  VkPipeline pipeline = pipelineState.createPipeline();
  if(pipeline == VK_NULL_HANDLE)
  {
    throw std::runtime_error("Failed to create graphics pipeline!");
  }

#ifdef _DEBUG
  // Generate a name for the graphics pipeline
  std::string generatedPipelineName = std::to_string(vertShaderModuleID.m_value) + " "                //
                                      + std::to_string(fragShaderModuleID.m_value) + " "              //
                                      + std::to_string(static_cast<uint32_t>(blendMode)) + " "        //
                                      + std::to_string(usesVertexInput) + " "                         //
                                      + std::to_string(reinterpret_cast<uint64_t>(renderPass)) + " "  //
                                      + std::to_string(subpass);
  m_debug.setObjectName(pipeline, generatedPipelineName.c_str());
#endif

  return pipeline;
}

VkCommandBuffer Sample::createTempCmdBuffer()
{
  VkCommandBuffer          cmd       = m_ringCmdPool.createCommandBuffer(VK_COMMAND_BUFFER_LEVEL_PRIMARY, false);
  VkCommandBufferBeginInfo beginInfo = nvvk::make<VkCommandBufferBeginInfo>();
  beginInfo.flags                    = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
  NVVK_CHECK(vkBeginCommandBuffer(cmd, &beginInfo));
  return cmd;
}

///////////////////////////////////////////////////////////////////////////////
// Main rendering logic                                                      //
///////////////////////////////////////////////////////////////////////////////

void Sample::updateUniformBuffer(uint32_t currentImage, double time)
{
  const uint32_t width       = m_colorImage.c_width;
  const uint32_t height      = m_colorImage.c_height;
  const float    aspectRatio = static_cast<float>(width) / static_cast<float>(height);
  nvmath::mat4   projection  = nvmath::perspectiveVK(45.0f, aspectRatio, 0.01f, 50.0f);
  nvmath::mat4   view        = m_cameraControl.m_viewMatrix;

  m_sceneUbo.projViewMatrix             = projection * view;
  m_sceneUbo.viewMatrix                 = view;
  m_sceneUbo.viewMatrixInverseTranspose = nvmath::transpose(nvmath::invert(view));

  m_sceneUbo.viewport = nvmath::ivec3(width, height, width * height);

  void* data = m_allocatorDma.map(m_uniformBuffers[currentImage]);
  memcpy(data, &m_sceneUbo, sizeof(m_sceneUbo));
  m_allocatorDma.unmap(m_uniformBuffers[currentImage]);
}

void Sample::copyOffscreenToBackBuffer(int winWidth, int winHeight, ImDrawData* imguiDrawData)
{
  // This function resolves + scales m_colorImage into m_guiCompositeImage, draws the Dear ImGui GUI onto
  // m_guiCompositeImage, and then blits m_guiCompositeImage onto the backbuffer. Because m_colorImage is
  // generally a different format (B8G8R8A8_SRGB) than m_guiCompositeImage (R8G8B8A8) (which in turn is required by
  // linear-space rendering) and sometimes a different size xor has different MSAA samples/pixel, the worst case
  // (MSAA resolve + change of format) takes two steps.
  // Note that we could do this in one step, and further customize the filters used, using a custom kernel.
  // Finally, Vulkan allows us to access the swapchain images themselves. However, while a previous version of this
  // sample did that, we now render the GUI to intermediate offscreen image, as this avoids potential problems with
  // swapchain recreation, and may be more familiar to developers used to OpenGL applications.
  //
  // As a result of the differences between MSAA resolve + downscaling, there are a few cases to handle.
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
  //        m_guiCompositeImage
  //                 |
  //       render Dear ImGui GUI
  //                 V
  //             Swapchain

  // Start a separate command buffer for this function.
  VkCommandBuffer          cmdBuffer = createTempCmdBuffer();
  nvh::Profiler::SectionID sec       = m_profilerVK.beginSection("CopyOffscreenToBackBuffer", cmdBuffer);

  // Prepare to transfer from m_colorImage; check its initial state for soundness
  assert(m_colorImage.currentLayout == VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
  assert(m_colorImage.currentAccesses == (VK_ACCESS_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT));
  m_colorImage.transitionTo(cmdBuffer, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_ACCESS_TRANSFER_READ_BIT);

  // Tracks the image that will be passed to vkCmdCopyImage
  // These are the defaults if no resolve or downsample is required.
  VkImage       copySrcImage  = m_colorImage.image.image;
  VkImageLayout copySrcLayout = m_colorImage.currentLayout;

  // If resolve or downsample required
  if(m_state.msaa != 1 || m_state.supersample != 1)
  {
    // Prepare to transfer data to m_downsampleImage
    m_downsampleImage.transitionTo(cmdBuffer, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_ACCESS_TRANSFER_WRITE_BIT);

    // MSAA branch
    if(m_state.msaa != 1)
    {
      // Resolve the MSAA image m_colorImage to m_downsampleImage
      VkImageResolve region            = {0};  // Zero-initialize
      region.srcSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
      region.srcSubresource.layerCount = 1;
      region.dstSubresource            = region.srcSubresource;
      region.extent                    = {m_colorImage.c_width, m_colorImage.c_height, 1};

      vkCmdResolveImage(cmdBuffer,                        // Command buffer
                        m_colorImage.image.image,         // Source image
                        m_colorImage.currentLayout,       // Source image layout
                        m_downsampleImage.image.image,    // Destination image
                        m_downsampleImage.currentLayout,  // Destination image layout
                        1,                                // Number of regions
                        &region);                         // Regions
    }
    else
    {
      // Downsample m_colorImage to m_downsampleTargeImage
      VkImageBlit region               = {0};  // Zero-initialize
      region.srcSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
      region.srcSubresource.layerCount = 1;
      region.dstSubresource            = region.srcSubresource;
      region.srcOffsets[1]             = {static_cast<int32_t>(m_colorImage.c_width),   //
                              static_cast<int32_t>(m_colorImage.c_height),  //
                              1};
      region.dstOffsets[1]             = {static_cast<int32_t>(m_downsampleImage.c_width),   //
                              static_cast<int32_t>(m_downsampleImage.c_height),  //
                              1};

      vkCmdBlitImage(cmdBuffer,                        // Command buffer
                     m_colorImage.image.image,         // Source image
                     m_colorImage.currentLayout,       // Source image
                     m_downsampleImage.image.image,    // Destination image
                     m_downsampleImage.currentLayout,  // Destination image layout
                     1,                                // Number of regions
                     &region,                          // Regions
                     VK_FILTER_LINEAR);                // Use tent filtering (= box filtering in this case)
    }

    // Prepare to transfer data from m_downsampleImage, and set copySrcImage and copySrcLayout.
    m_downsampleImage.transitionTo(cmdBuffer, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_ACCESS_TRANSFER_READ_BIT);
    copySrcImage  = m_downsampleImage.image.image;
    copySrcLayout = m_downsampleImage.currentLayout;
  }

  // Prepare to transfer data to m_guiCompositeImage
  m_guiCompositeImage.transitionTo(cmdBuffer, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_ACCESS_TRANSFER_WRITE_BIT);

  // Now, we want to copy data from copySrcImage to m_guiCompositeImage instead of blitting it, since blitting will try
  // to convert the sRGB data and store it in linear format, which isn't what we want.
  {
    VkImageCopy region               = {0};
    region.srcSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    region.srcSubresource.layerCount = 1;
    region.dstSubresource            = region.srcSubresource;
    region.extent                    = {m_guiCompositeImage.c_width, m_guiCompositeImage.c_height, 1};
    vkCmdCopyImage(cmdBuffer,                          // Command buffer
                   copySrcImage,                       // Source image
                   copySrcLayout,                      // Source image layout
                   m_guiCompositeImage.image.image,    // Destination image
                   m_guiCompositeImage.currentLayout,  // Destination image layout
                   1,                                  // Number of regions
                   &region);                           // Regions
  }

  // Now, render the GUI.
  // If draw data exists, we begin a new render pass and call ImGui::RenderDrawDataVK.
  // This render pass takes m_guiCompositeImage and transitions it to layout TRANSFER_SRC_OPTIMAL, so if we don't call
  // that render pass, we have to do the transition manually.

  if(imguiDrawData)
  {
    VkRenderPassBeginInfo renderPassBeginInfo    = nvvk::make<VkRenderPassBeginInfo>();
    renderPassBeginInfo.renderPass               = m_renderPassGUI;
    renderPassBeginInfo.framebuffer              = m_guiFramebuffer;
    renderPassBeginInfo.renderArea.offset.x      = 0;
    renderPassBeginInfo.renderArea.offset.y      = 0;
    renderPassBeginInfo.renderArea.extent.width  = winWidth;
    renderPassBeginInfo.renderArea.extent.height = winHeight;
    renderPassBeginInfo.clearValueCount          = 0;
    renderPassBeginInfo.pClearValues             = nullptr;

    vkCmdBeginRenderPass(cmdBuffer, &renderPassBeginInfo, VK_SUBPASS_CONTENTS_INLINE);

    vkCmdSetViewport(cmdBuffer, 0, 1, &m_viewportGUI);
    vkCmdSetScissor(cmdBuffer, 0, 1, &m_scissorGUI);

    ImGui_ImplVulkan_RenderDrawData(imguiDrawData, cmdBuffer);

    vkCmdEndRenderPass(cmdBuffer);

    // Since the render pass changed the layout and accesses, we have to tell the ImageAndView abstraction that
    // these changed:
    m_guiCompositeImage.currentLayout   = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
    m_guiCompositeImage.currentAccesses = VK_ACCESS_TRANSFER_READ_BIT;
  }
  else
  {
    m_guiCompositeImage.transitionTo(cmdBuffer, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_ACCESS_TRANSFER_READ_BIT);
  }

  // Finally, blit to the swapchain.
  {
    // Soundness check
    assert(m_guiCompositeImage.c_width == winWidth);
    assert(m_guiCompositeImage.c_height == winHeight);
    VkImageBlit region               = {0};
    region.dstOffsets[1].x           = winWidth;
    region.dstOffsets[1].y           = winHeight;
    region.dstOffsets[1].z           = 1;
    region.srcOffsets[1].x           = winWidth;
    region.srcOffsets[1].y           = winHeight;
    region.srcOffsets[1].z           = 1;
    region.dstSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    region.dstSubresource.layerCount = 1;
    region.srcSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    region.srcSubresource.layerCount = 1;

    cmdImageTransition(cmdBuffer, m_swapChain.getActiveImage(), VK_IMAGE_ASPECT_COLOR_BIT, 0, VK_ACCESS_TRANSFER_WRITE_BIT,
                       VK_IMAGE_LAYOUT_PRESENT_SRC_KHR, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);

    vkCmdBlitImage(cmdBuffer,                             // Command buffer
                   m_guiCompositeImage.image.image,       // Source image
                   m_guiCompositeImage.currentLayout,     // Source image layout
                   m_swapChain.getActiveImage(),          // Destination image
                   VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,  // Destination image layout
                   1,                                     // Number of regions
                   &region,                               // Region
                   VK_FILTER_NEAREST);                    // Filter

    cmdImageTransition(cmdBuffer, m_swapChain.getActiveImage(), VK_IMAGE_ASPECT_COLOR_BIT, VK_ACCESS_TRANSFER_WRITE_BIT,
                       0, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_PRESENT_SRC_KHR);
  }

  // Reset the layout of m_colorImage.
  m_colorImage.transitionTo(cmdBuffer, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                            VK_ACCESS_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT);

  m_profilerVK.endSection(sec, cmdBuffer);

  vkEndCommandBuffer(cmdBuffer);
  m_submission.enqueue(cmdBuffer);
}

void Sample::submissionExecute(VkFence fence, bool useImageReadWait, bool useImageWriteSignals)
{
  if(useImageReadWait && m_submissionWaitForRead)
  {
    VkSemaphore semRead = m_swapChain.getActiveReadSemaphore();
    if(semRead)
    {
      m_submission.enqueueWait(semRead, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT);
    }
    m_submissionWaitForRead = false;
  }

  if(useImageWriteSignals)
  {
    VkSemaphore semWritten = m_swapChain.getActiveWrittenSemaphore();
    if(semWritten)
    {
      m_submission.enqueueSignal(semWritten);
    }
  }

  m_submission.execute(fence);
}

void Sample::think(double time)
{
  int    width          = m_windowState.m_swapSize[0];
  int    height         = m_windowState.m_swapSize[1];
  double frameStartTime = getTime();

  // Create Dear ImGui interface
  DoGUI(width, height, time);

  // If elements of m_state change, this reinitializes parts of the renderer.
  updateRendererImmediate(false, false);

  // Begin frame
  {
    m_submissionWaitForRead = true;
    m_ringFences.setCycleAndWait(m_frame);
    m_ringCmdPool.setCycle(m_ringFences.getCycleIndex());
  }

  // Update camera
  m_cameraControl.processActions(nvmath::vec2i(getWidth(), getHeight()),
                                 nvmath::vec2f(m_windowState.m_mouseCurrent[0], m_windowState.m_mouseCurrent[1]),
                                 m_windowState.m_mouseButtonFlags, m_windowState.m_mouseWheel);

  // Update the GPU's uniform buffer
  updateUniformBuffer(m_swapChain.getActiveImageIndex(), frameStartTime);

  // Record this frame's command buffer
  VkCommandBuffer cmdBuffer = m_ringCmdPool.createCommandBuffer();
  {
    render(cmdBuffer);
    NVVK_CHECK(vkEndCommandBuffer(cmdBuffer));
    m_submission.enqueue(cmdBuffer);
  }

  // Render Dear ImGui and translate the internal image to the swapchain
  {
    ImGui::Render();
    copyOffscreenToBackBuffer(width, height, ImGui::GetDrawData());
  }

  // End frame
  {
    submissionExecute(m_ringFences.getFence(), true, true);
    m_frame++;
    ImGui::EndFrame();
    m_lastState = m_state;
  }
}

int main(int argc, const char** argv)
{
  NVPSystem system(PROJECT_NAME);

  Sample sample;
  sample.m_contextInfo.addDeviceExtension(VK_KHR_SAMPLER_MIRROR_CLAMP_TO_EDGE_EXTENSION_NAME);
  sample.m_contextInfo.addDeviceExtension(VK_EXT_DESCRIPTOR_INDEXING_EXTENSION_NAME);
  sample.m_contextInfo.addDeviceExtension(VK_KHR_MAINTENANCE3_EXTENSION_NAME);
  VkPhysicalDevicePipelineExecutablePropertiesFeaturesKHR physicalDevicePipelineExecutableProprtiesFeaturesKHR = {
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PIPELINE_EXECUTABLE_PROPERTIES_FEATURES_KHR, nullptr, VK_TRUE};
  sample.m_contextInfo.addDeviceExtension(VK_KHR_PIPELINE_EXECUTABLE_PROPERTIES_EXTENSION_NAME, true,
                                          &physicalDevicePipelineExecutableProprtiesFeaturesKHR);

  // These extensions are both optional - there are algorithms we can use if we have them, but
  // if the device doesn't support these extensions, we don't allow the user to select those algorithms.
  sample.m_contextInfo.addDeviceExtension(VK_KHR_SHADER_ATOMIC_INT64_EXTENSION_NAME, true);
  // VK_EXT_FRAGMENT_SHADER_INTERLOCK uses an extension which will be passed to device creation via
  // VkDeviceCreateInfo's pNext chain:
  VkPhysicalDeviceFragmentShaderInterlockFeaturesEXT m_fragmentShaderInterlockFeatures{
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FRAGMENT_SHADER_INTERLOCK_FEATURES_EXT,  // sType
      nullptr,                                                                   // pNext
      VK_TRUE,                                                                   // fragmentShaderSampleInterlock
      VK_TRUE,                                                                   // fragmentShaderPixelInterlock
      VK_FALSE};
  sample.m_contextInfo.addDeviceExtension(VK_EXT_FRAGMENT_SHADER_INTERLOCK_EXTENSION_NAME, true, &m_fragmentShaderInterlockFeatures);

  const int SAMPLE_WIDTH  = 1200;
  const int SAMPLE_HEIGHT = 1024;
  return sample.run(PROJECT_NAME, argc, argv, SAMPLE_WIDTH, SAMPLE_HEIGHT);
}
