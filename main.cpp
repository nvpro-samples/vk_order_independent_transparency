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

#if defined(_WIN32) && (!defined(VK_USE_PLATFORM_WIN32_KHR))
#define VK_USE_PLATFORM_WIN32_KHR
#endif  // #if defined(_WIN32) && (!defined(VK_USE_PLATFORM_WIN32_KHR))

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

#include <imgui/imgui_helper.h>
#include <imgui/imgui_impl_vk.h>

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

void Sample::onWindowResize(int w, int h)
{
  NVPWindow::onWindowResize(w, h);
  // Don't do anything when the window is minimized:
  if(w != 0 && h != 0)
  {
    m_swapChain.update(w, h, m_vsync);
    {
      nvvk::ScopeCommandBuffer scopedCmdBuffer(m_context, m_context.m_queueGCT, m_context.m_queueGCT);
      resizeCommands(scopedCmdBuffer, false);
      // scopedCmdBuffer goes out of scope here and runs its commands.
    }
    // Complete copies and release staging memory.
    m_allocatorDma.finalizeAndReleaseStaging();
  }
  else
  {
    // Update the previous width and height even if the size was 0 so that
    // resizeCommands knows it needs to rebuild things
    m_lastSwapChainWidth  = w;
    m_lastSwapChainHeight = h;
  }
}

void Sample::onMouseMotion(int x, int y)
{
  if(m_state.drawUI)
    if(ImGuiH::mouse_pos(x, y))
      return;  // Captured by Dear ImGui
  m_controllerState.mouseCurrent[0] = static_cast<nvmath::nv_scalar>(x);
  m_controllerState.mouseCurrent[1] = static_cast<nvmath::nv_scalar>(y);
}

void Sample::onMouseButton(MouseButton button, ButtonAction action, int mods, int x, int y)
{
  if(m_state.drawUI)
    if(ImGuiH::mouse_button(button, action))
      return;  // Captured by Dear ImGui

  int flags = 0;
  switch(button)
  {
    case MouseButton::MOUSE_BUTTON_LEFT:
      flags = 1;
      break;
    case MouseButton::MOUSE_BUTTON_MIDDLE:
      flags = 2;
      break;
    case MouseButton::MOUSE_BUTTON_RIGHT:
      flags = 4;
      break;
    default:
      break;
  }
  if(action == ButtonAction::BUTTON_PRESS)
  {
    // Add the flags
    m_controllerState.mouseButtonFlags |= flags;
  }
  else if(action == ButtonAction::BUTTON_RELEASE)
  {
    // Remove the flags
    m_controllerState.mouseButtonFlags &= ~flags;
  }
}

void Sample::onMouseWheel(int delta)
{
  static int imGuiMouseWheel = 0;
  imGuiMouseWheel += delta;
  if(m_state.drawUI)
    if(ImGuiH::mouse_wheel(delta))
      return;  // Captured by Dear ImGui
  m_controllerState.mouseWheel += delta;
}

void Sample::onKeyboardChar(unsigned char key, int mods, int x, int y)
{
  if(m_state.drawUI)
    if(ImGuiH::key_char(key))
      return;  // Captured by Dear ImGui

  NVPWindow::onKeyboardChar(key, mods, x, y);
}

void Sample::onKeyboard(NVPWindow::KeyCode key, ButtonAction action, int mods, int x, int y)
{
  if(m_state.drawUI)
    if(ImGuiH::key_button(static_cast<int>(key), action, mods))
      return;  // Captured by Dear ImGui

  if(key == NVPWindow::KEY_V && action == NVPWindow::BUTTON_PRESS)
  {
    m_vsync = !m_vsync;
    onWindowResize(m_swapChain.getWidth(), m_swapChain.getHeight());
  }
}


///////////////////////////////////////////////////////////////////////////////
// Object Creation, Destruction, and Recreation                              //
///////////////////////////////////////////////////////////////////////////////

bool Sample::init(int posX, int posY, int width, int height, const char* title)
{
  // Open the window, not requiring a GL context
  if(!open(posX, posY, width, height, title, false))
  {
    return false;  // Initialization failed
  }

  // Initialize the profiler. m_context can be implicitly converted to a VkDevice.
  m_profilerVK.init(m_context, m_context.m_physicalDevice);
  // Create the window surface (using GLFW, our windowing library)
  // m_internal is NVPWindow's GLFWWindow pointer.
  VkResult vkResult = glfwCreateWindowSurface(m_context.m_instance, m_internal, nullptr, &m_surface);
  assert(vkResult == VK_SUCCESS);
  // Initialize the swapchain (which will use a B8R8G8A8 format by default)
  // Make sure the GCT queue can present to the given surface
  bool gctResult = m_context.setGCTQueueWithPresent(m_surface);
  assert(gctResult);
  m_swapChain.init(m_context, m_context.m_physicalDevice, m_context.m_queueGCT, m_context.m_queueGCT.familyIndex, m_surface);
  m_swapChain.update(width, height, m_vsync);

  // Initialize Dear ImGui
  ImGuiH::Init(m_swapChain.getWidth(), m_swapChain.getHeight(), this);
  ImGui::InitVK(m_context, m_context.m_physicalDevice, m_context.m_queueGCT.queue, m_context.m_queueGCT.familyIndex,
                m_renderPassGUI, 0);     // ImGui will use its own render pass
  ImGui::GetIO().IniFilename = nullptr;  // Don't create a .ini file for storing data across application launches

  // Initialize all Vulkan components that will be constant throughout the application lifecycle.
  // Components that can change are handled by updateRendererFromState.
  m_ringFences.init(m_context);
  m_ringCmdPool.init(m_context, m_context.m_queueGCT, VK_COMMAND_POOL_CREATE_TRANSIENT_BIT);
  m_submission.init(m_context.m_queueGCT.queue);
  m_debug.setup(m_context);

  createTextureSampler();
  m_deviceMemoryAllocator.init(m_context, m_context.m_physicalDevice);
  m_allocatorDma.init(m_context, m_context.m_physicalDevice, &m_deviceMemoryAllocator);

  // Configure shader system (note that this also creates shader modules as we add them)
  {
    // Initialize shader system (this keeps track of shaders so that you can reload all of them at once):
    m_shaderModuleManager.init(m_context);
    // Add search paths for files and includes
    m_shaderModuleManager.addDirectory("GLSL_" PROJECT_NAME);  // For when running in the install directory
    m_shaderModuleManager.addDirectory(".");
    m_shaderModuleManager.addDirectory(NVPSystem::exePath() + PROJECT_RELDIRECTORY + "..");
    m_shaderModuleManager.addDirectory("..");     // for when working directory in Debug is $(ProjectDir)
    m_shaderModuleManager.addDirectory("../..");  // for when using $(TargetDir)
    m_shaderModuleManager.addDirectory("../../sandbox/" PROJECT_NAME);     // for when using $(TargetDir)
    m_shaderModuleManager.addDirectory("../../../sandbox/" PROJECT_NAME);  // for when using $(TargetDir) and build_all
    m_shaderModuleManager.addDirectory("../sandbox/" PROJECT_NAME);  // For when running from the bin_x64 directory on Linux
    // We have to manually set up paths to files we could include.
    m_shaderModuleManager.registerInclude("common.h");
    m_shaderModuleManager.registerInclude("oitColorDepthDefines.glsl");
    m_shaderModuleManager.registerInclude("oitCompositeDefines.glsl");
    m_shaderModuleManager.registerInclude("shaderCommon.glsl");
  }

  // Call resizeCommands to set up the rest of the renderer with the initial
  // swapchain size:
  {
    // We use the same indices here as createCommandPool:
    nvvk::ScopeCommandBuffer scopedCmdBuffer(m_context, m_context.m_queueGCT, m_context.m_queueGCT);
    resizeCommands(scopedCmdBuffer, true);
    // scopedCmdBuffer goes out of scope, which calls its destroyer,
    // which ends the command buffer, submits it, and waits idle; if any
    // errors have occurred, this will also call exit(-1).
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

  return true;  // Initialization succeeded
}

void Sample::resizeCommands(nvvk::ScopeCommandBuffer& scopedCmdBuffer, bool forceRebuildAll)
{
  vkDeviceWaitIdle(m_context);
  m_swapChain.cmdUpdateBarriers(scopedCmdBuffer);
  createUniformBuffers();
  updateRendererFromState(scopedCmdBuffer, forceRebuildAll);
  ImGui::ReInitPipelinesVK(m_renderPassGUI, 0);
}

void Sample::updateRendererFromState(VkCommandBuffer cmdBuffer, bool forceRebuildAll)
{
  m_state.recomputeAntialiasingSettings();

  // Determine what needs to be rebuilt

  const bool swapchainSizeChanged = (m_swapChain.getWidth() != m_lastSwapChainWidth)       //
                                    || (m_swapChain.getHeight() != m_lastSwapChainHeight)  //
                                    || forceRebuildAll;

  const bool vsyncChanged = (m_lastVsync != m_vsync) || forceRebuildAll;

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
  }

  // Store this call's GUI state to compare to next call
  m_lastState           = m_state;
  m_lastSwapChainWidth  = m_swapChain.getWidth();
  m_lastSwapChainHeight = m_swapChain.getHeight();
  m_lastVsync           = m_vsync;
}

void Sample::close()
{
  vkDeviceWaitIdle(m_context);

  NVPWindow::close();
  m_profilerVK.deinit();

  ImGui::ShutdownVK();
  ImGui::DestroyContext();

  // From updateRendererFromState
  destroyGraphicsPipelines();
  m_shaderModuleManager.deinit();
  destroyFramebuffers();
  destroyRenderPasses();
  destroyDescriptorSets();
  destroyImages();
  destroyScene();
  // From resizeCommands
  destroyUniformBuffers();
  // From begin
  m_allocatorDma.deinit();
  m_deviceMemoryAllocator.deinit();
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

  VkResult result = vkCreateSampler(m_context, &samplerInfo, nullptr, &m_pointSampler);
  assert(result == VK_SUCCESS);
}

void Sample::destroyUniformBuffers()
{
  for(nvvk::BufferDma& uniformBuffer : m_uniformBuffers)
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

  // When this goes out of scope, it'll synchronously perform all of the copy operations.
  nvvk::StagingMemoryManager scopedTransfer(m_context.m_device, m_context.m_physicalDevice);

  {
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
  if(m_mainColorDepthFramebuffer != nullptr)
  {
    vkDestroyFramebuffer(m_context, m_mainColorDepthFramebuffer, nullptr);
    m_mainColorDepthFramebuffer = nullptr;
  }

  if(m_weightedFramebuffer != nullptr)
  {
    vkDestroyFramebuffer(m_context, m_weightedFramebuffer, nullptr);
    m_weightedFramebuffer = nullptr;
  }

  for(VkFramebuffer framebuffer : m_guiSwapChainFramebuffers)
  {
    vkDestroyFramebuffer(m_context, framebuffer, nullptr);
  }
}

void Sample::createFramebuffers()
{
  destroyFramebuffers();

  const uint32_t numSwapChainImages = m_swapChain.getImageCount();

  // Color + depth offscreen framebuffer
  {
    std::array<VkImageView, 2> attachments = {m_colorImage.view, m_depthImage.view};

    VkFramebufferCreateInfo framebufferInfo = nvvk::make<VkFramebufferCreateInfo>();
    framebufferInfo.renderPass              = m_renderPassColorDepthClear;
    framebufferInfo.attachmentCount         = static_cast<uint32_t>(attachments.size());
    framebufferInfo.pAttachments            = attachments.data();
    framebufferInfo.width                   = m_colorImage.c_width;
    framebufferInfo.height                  = m_colorImage.c_height;
    framebufferInfo.layers                  = 1;

    if(vkCreateFramebuffer(m_context, &framebufferInfo, nullptr, &m_mainColorDepthFramebuffer) != VK_SUCCESS)
    {
      throw std::runtime_error("Failed to create main color+depth framebuffer!");
    }

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

    if(vkCreateFramebuffer(m_context, &framebufferInfo, nullptr, &m_weightedFramebuffer) != VK_SUCCESS)
    {
      throw std::runtime_error("Failed to create the Weighted, Blended OIT framebuffer!");
    }

    m_debug.setObjectName(m_weightedFramebuffer, "m_weightedColorRevealFramebuffer");
  }

  // GUI swapchain framebuffers
  m_guiSwapChainFramebuffers.resize(numSwapChainImages);

  for(uint32_t i = 0; i < numSwapChainImages; i++)
  {
    VkImageView swapChainColorView = m_swapChain.getImageView(i);

    VkFramebufferCreateInfo framebufferInfo = nvvk::make<VkFramebufferCreateInfo>();
    framebufferInfo.renderPass              = m_renderPassGUI;
    framebufferInfo.attachmentCount         = 1;
    framebufferInfo.pAttachments            = &swapChainColorView;
    framebufferInfo.width                   = m_swapChain.getWidth();
    framebufferInfo.height                  = m_swapChain.getHeight();
    framebufferInfo.layers                  = 1;

    if(vkCreateFramebuffer(m_context, &framebufferInfo, nullptr, &m_guiSwapChainFramebuffers[i]) != VK_SUCCESS)
    {
      throw std::runtime_error("Failed to create GUI framebuffer!");
    }

    m_debug.setObjectName(m_guiSwapChainFramebuffers[i], "m_guiSwapChainFramebuffers[...]");
  }
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

void Sample::copyOffscreenToBackBuffer(VkCommandBuffer cmdBuffer)
{
  const nvvk::ProfilerVK::Section scopedTimer(m_profilerVK, "CopyOffscreenToBackBuffer", cmdBuffer);

  // Outline:
  // Transition color
  // set copySrcTexture to color
  // MSAA or SSAA:
  //   transition intermediate
  //   set copySrcTexture to intermediate
  //   MSAA:
  //     resolve color to intermediate
  //   SSAA:
  //     blit color to intermediate
  //   transition intermediate
  // copy copySrcTexture to backbuffer

  // Transition the color attachment to a state that we can read from.
  const VkImageLayout        originalLayout   = m_colorImage.currentLayout;
  const VkPipelineStageFlags originalStages   = m_colorImage.currentStages;
  const VkAccessFlags        originalAccesses = m_colorImage.currentAccesses;
  m_colorImage.transitionTo(cmdBuffer,                             //
                            VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,  //
                            VK_PIPELINE_STAGE_TRANSFER_BIT,        //
                            VK_ACCESS_TRANSFER_READ_BIT);

  VkImage       copySrcImage  = m_colorImage.image.image;
  VkImageLayout copySrcLayout = m_colorImage.currentLayout;

  if(m_state.msaa != 1 || m_state.supersample != 1)
  {
    m_downsampleTargetImage.transitionTo(cmdBuffer,                             //
                                         VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,  //
                                         VK_PIPELINE_STAGE_TRANSFER_BIT,        //
                                         VK_ACCESS_TRANSFER_WRITE_BIT);

    if(m_state.msaa != 1)
    {
      // Resolve the MSAA image m_colorImage to m_downsampleTargetImage
      // Note: These have to be the same size - so it looks like we can't
      // resolve MSAA and downsample in the same step. Instead, we would
      // have to resolve MSAA, then downsample, or use a different approach
      // (e.g. a compute shader).
      VkImageResolve region;
      region.srcSubresource.aspectMask     = VK_IMAGE_ASPECT_COLOR_BIT;
      region.srcSubresource.baseArrayLayer = 0;
      region.srcSubresource.layerCount     = 1;  // Image is not an array
      region.srcSubresource.mipLevel       = 0;
      region.srcOffset                     = {0, 0, 0};
      region.dstSubresource                = region.srcSubresource;
      region.dstOffset                     = {0, 0, 0};
      region.extent                        = {m_colorImage.c_width, m_colorImage.c_height, 1};

      vkCmdResolveImage(cmdBuffer,                              // Command buffer
                        m_colorImage.image.image,               // Source image
                        m_colorImage.currentLayout,             // Source image layout
                        m_downsampleTargetImage.image.image,    // Destination image
                        m_downsampleTargetImage.currentLayout,  // Destination image layout
                        1,                                      // Number of regions
                        &region);                               // Regions
    }
    else
    {
      // Downsample m_colorImage to m_downsampleTargeImage

      VkImageBlit region;
      region.srcSubresource.aspectMask     = VK_IMAGE_ASPECT_COLOR_BIT;
      region.srcSubresource.baseArrayLayer = 0;
      region.srcSubresource.layerCount     = 1;  // Image is not an array
      region.srcSubresource.mipLevel       = 0;
      region.srcOffsets[0]                 = {0, 0, 0};
      region.srcOffsets[1]                 = {static_cast<int32_t>(m_colorImage.c_width),   //
                              static_cast<int32_t>(m_colorImage.c_height),  //
                              1};
      region.dstSubresource                = region.srcSubresource;
      region.dstOffsets[0]                 = {0, 0, 0};
      region.dstOffsets[1]                 = {static_cast<int32_t>(m_downsampleTargetImage.c_width),   //
                              static_cast<int32_t>(m_downsampleTargetImage.c_height),  //
                              1};

      vkCmdBlitImage(cmdBuffer,                              // Command buffer
                     m_colorImage.image.image,               // Source image
                     m_colorImage.currentLayout,             // Source image
                     m_downsampleTargetImage.image.image,    // Destination image
                     m_downsampleTargetImage.currentLayout,  // Destination image layout
                     1,                                      // Number of regions
                     &region,                                // Regions
                     VK_FILTER_LINEAR);                      // Use tent filtering (= box filtering in this case)
    }

    m_downsampleTargetImage.transitionTo(cmdBuffer,                             //
                                         VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,  //
                                         VK_PIPELINE_STAGE_TRANSFER_BIT,        //
                                         VK_ACCESS_TRANSFER_READ_BIT);

    copySrcImage  = m_downsampleTargetImage.image.image;
    copySrcLayout = m_downsampleTargetImage.currentLayout;
  }

  // Transition the backbuffer to a state that we can write to
  // The (VK_IMAGE_LAYOUT_PRESENT_SRC_KHR, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, 0)
  // triplet comes from how the images were created in swapchain_vk.cpp.
  doTransition(cmdBuffer,                             //
               m_swapChain.getActiveImage(),          //
               VK_IMAGE_ASPECT_COLOR_BIT,             //
               VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,       //
               VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,     //
               0,                                     //
               VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,  //
               VK_PIPELINE_STAGE_TRANSFER_BIT,        //
               VK_ACCESS_TRANSFER_WRITE_BIT,          //
               1);                                    // Image is not an array

  // Copy the internal data of copySrcTexture to the backbuffer
  {
    VkImageCopy region;
    region.srcSubresource.aspectMask     = VK_IMAGE_ASPECT_COLOR_BIT;
    region.srcSubresource.baseArrayLayer = 0;
    region.srcSubresource.layerCount     = 1;
    region.srcSubresource.mipLevel       = 0;
    region.srcOffset                     = {0, 0, 0};
    region.dstSubresource                = region.srcSubresource;
    region.dstOffset                     = {0, 0, 0};
    region.extent                        = {m_swapChain.getWidth(), m_swapChain.getHeight(), 1};

    // We want to copy the data instead of blitting it, since blitting
    // will try to convert the sRGB data in m_colorImage and store it in
    // linear format.
    vkCmdCopyImage(cmdBuffer,                             // Command buffer
                   copySrcImage,                          // Source image
                   copySrcLayout,                         // Source image layout
                   m_swapChain.getActiveImage(),          // Destination image
                   VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,  // Destination image layout
                   1,                                     // Number of regions
                   &region);                              // Regions
  }

  // Transition the backbuffer to prepare to write to it
  doTransition(cmdBuffer,                                                                   //
               m_swapChain.getActiveImage(),                                                //
               VK_IMAGE_ASPECT_COLOR_BIT,                                                   //
               VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,                                        //
               VK_PIPELINE_STAGE_TRANSFER_BIT,                                              //
               VK_ACCESS_TRANSFER_WRITE_BIT,                                                //
               VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,                                    //
               VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,                               //
               VK_ACCESS_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,  //
               1);                                                                          // Image is not an array

  // Transition the color layout back
  m_colorImage.transitionTo(cmdBuffer,       //
                            originalLayout,  //
                            originalStages,  //
                            originalAccesses);
}

void Sample::display()
{
  if(getWidth() == 0 || getHeight() == 0)
  {
    // Don't do anything while the window is minimized
    return;
  }

  // Start the profiler print loop
  double frameStartTime = getTime();
  if(m_lastProfilerPrintTime == 0.0)
  {
    m_lastProfilerPrintTime = frameStartTime;
  }
  m_timeCounter.update(true);

  // Get the next swapchain image and begin the frame
  m_swapChain.acquire();
  m_profilerVK.beginFrame();

  // m_ringFences is a ring buffer of fences (which prevent the CPU from
  // advancing until the GPU completes an operation, and m_ringCmdPool is a
  // ring buffer of command pools.
  // The idea is that we'd like to be able to record the instructions to draw
  // a frame while the GPU is drawing another frame. But we don't want to
  // overwrite resources that the GPU is currently using. So, m_ringCmdPool
  // gives us a different pool of command buffers per frame (of which we only
  // use one), and m_ringFences contains an array of fences that prevent us
  // from overwriting the command buffer for a frame until the GPU has
  // finished processing it.
  // There's also synchronization using semaphores between the graphics,
  // compute, and transfer queue, and the present queue, to prevent the GCT
  // queue from overwriting a swapchain image that the present queue is using.
  m_ringFences.setCycleAndWait(m_swapChain.getActiveImageIndex());
  m_ringCmdPool.setCycle(m_ringFences.getCycleIndex());

  // Update the GPU's uniform buffer
  updateUniformBuffer(m_swapChain.getActiveImageIndex(), frameStartTime);

  // Update Dear ImGui configuration
  {
    auto& imguiIO       = ImGui::GetIO();
    imguiIO.DeltaTime   = static_cast<float>(m_timeCounter.getFrameDT());
    imguiIO.DisplaySize = ImVec2(static_cast<float>(m_swapChain.getWidth()), static_cast<float>(m_swapChain.getHeight()));
  }

  // Create Dear ImGui interface
  ImGui::NewFrame();
  DoGUI();

  // Update camera
  m_cameraControl.processActions(nvmath::vec2i(getWidth(), getHeight()), m_controllerState.mouseCurrent,
                                 m_controllerState.mouseButtonFlags, m_controllerState.mouseWheel);

  // Record this frame's command buffer
  // This line creates and begins a primary command buffer:
  VkCommandBuffer cmdBuffer = m_ringCmdPool.createCommandBuffer();
  {
    render(cmdBuffer);

    copyOffscreenToBackBuffer(cmdBuffer);

    // Render Dear ImGui
    {
      const nvvk::ProfilerVK::Section scopedTimer(m_profilerVK, "GUI", cmdBuffer);

      // Set up the render pass
      VkRenderPassBeginInfo renderPassInfo    = nvvk::make<VkRenderPassBeginInfo>();
      renderPassInfo.renderPass               = m_renderPassGUI;
      renderPassInfo.framebuffer              = m_guiSwapChainFramebuffers[m_swapChain.getActiveImageIndex()];
      renderPassInfo.renderArea.offset        = {0, 0};
      renderPassInfo.renderArea.extent.width  = m_swapChain.getWidth();
      renderPassInfo.renderArea.extent.height = m_swapChain.getHeight();

      vkCmdBeginRenderPass(cmdBuffer, &renderPassInfo, VK_SUBPASS_CONTENTS_INLINE);

      if(m_state.drawUI)
      {
        ImGui::Render();
        ImGui::RenderDrawDataVK(cmdBuffer, ImGui::GetDrawData());
      }
      ImGui::EndFrame();

      vkCmdEndRenderPass(cmdBuffer);
    }

    if(vkEndCommandBuffer(cmdBuffer) != VK_SUCCESS)
    {
      assert(!"Failed to record command buffer!");
    }
  }

  m_submission.enqueueWait(m_swapChain.getActiveReadSemaphore(), VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT);

  m_submission.enqueueSignal(m_swapChain.getActiveWrittenSemaphore());

  m_submission.enqueue(cmdBuffer);

  if(m_submission.execute(m_ringFences.getFence()) != VK_SUCCESS)
  {
    throw std::runtime_error("Failed to submit draw command buffer!");
  }

  m_profilerVK.endFrame();
  // Update profiling information
  m_framesSinceLastProfilerPrint++;
  if(frameStartTime > m_lastProfilerPrintTime + 1.0f)
  {
    std::stringstream combinedTitle;

    combinedTitle << PROJECT_NAME << ": " <<  //
        m_timeCounter.getFrameDT() * 1000.0 << " [ms]";
    if(m_vsync)
    {
      combinedTitle << " (vsync on - V for toggle)";
    }
    setTitle(combinedTitle.str().c_str());

    std::string stats;
    m_profilerVK.print(stats);
    if(!stats.empty())
    {
      LOGI("%s\n", stats.c_str());
    }

    m_lastProfilerPrintTime        = frameStartTime;
    m_framesSinceLastProfilerPrint = 0;
  }

  // Present the frame!
  m_swapChain.present();
}

int main(int argc, const char** argv)
{
  // NVPSystem initializes GLFW and sets its error callback:
  NVPSystem sys(argv[0], PROJECT_NAME);
  // Create a single window
  static Sample sample;

  // Create the Vulkan context
#ifdef _DEBUG
  const bool bUseValidation = true;
#else   // #ifdef _DEBUG
  const bool bUseValidation = false;
#endif  // #ifdef _DEBUG
  nvvk::ContextCreateInfo deviceInfo(bUseValidation);
  deviceInfo.apiMajor = 1;
  deviceInfo.apiMinor = 1;
  deviceInfo.addInstanceExtension(VK_KHR_SURFACE_EXTENSION_NAME, false);
#ifdef _WIN32
  deviceInfo.addInstanceExtension(VK_KHR_WIN32_SURFACE_EXTENSION_NAME, false);
#else
  deviceInfo.addInstanceExtension(VK_KHR_XLIB_SURFACE_EXTENSION_NAME);
  deviceInfo.addInstanceExtension(VK_KHR_XCB_SURFACE_EXTENSION_NAME);
#endif
  deviceInfo.addDeviceExtension(VK_KHR_SAMPLER_MIRROR_CLAMP_TO_EDGE_EXTENSION_NAME);
  deviceInfo.addDeviceExtension(VK_EXT_DESCRIPTOR_INDEXING_EXTENSION_NAME, false);
  deviceInfo.addDeviceExtension(VK_KHR_MAINTENANCE3_EXTENSION_NAME, false);
  deviceInfo.addDeviceExtension(VK_KHR_SWAPCHAIN_EXTENSION_NAME, false);

  VkPhysicalDevicePipelineExecutablePropertiesFeaturesKHR physicalDevicePipelineExecutablePropertiesFeaturesKHR = {
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PIPELINE_EXECUTABLE_PROPERTIES_FEATURES_KHR, nullptr, VK_TRUE};
  deviceInfo.addDeviceExtension(VK_KHR_PIPELINE_EXECUTABLE_PROPERTIES_EXTENSION_NAME, true,
                                &physicalDevicePipelineExecutablePropertiesFeaturesKHR);

  // These extensions are both optional - there are algorithms we can use if we have them, but
  // if the device doesn't support these extensions, we don't allow the user to select those algorithms.
  deviceInfo.addDeviceExtension(VK_KHR_SHADER_ATOMIC_INT64_EXTENSION_NAME, true);
  // VK_EXT_FRAGMENT_SHADER_INTERLOCK uses an extension which will be passed to device creation via
  // VkDeviceCreateInfo's pNext chain:
  VkPhysicalDeviceFragmentShaderInterlockFeaturesEXT m_fragmentShaderInterlockFeatures{
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FRAGMENT_SHADER_INTERLOCK_FEATURES_EXT,  // sType
      nullptr,                                                                   // pNext
      VK_TRUE,                                                                   // fragmentShaderSampleInterlock
      VK_TRUE,                                                                   // fragmentShaderPixelInterlock
      VK_FALSE};
  deviceInfo.addDeviceExtension(VK_EXT_FRAGMENT_SHADER_INTERLOCK_EXTENSION_NAME, true, &m_fragmentShaderInterlockFeatures);

  if(!sample.m_context.init(deviceInfo))
  {
    LOGE("Failed to initialize the Vulkan context!\n");
    return EXIT_FAILURE;
  }

  // Create the window
  const int SAMPLE_WIDTH  = 1200;
  const int SAMPLE_HEIGHT = 1024;
  if(!sample.init(0, 0, SAMPLE_WIDTH, SAMPLE_HEIGHT, PROJECT_NAME))
  {
    LOGE("Failed to initialize the sample!\n");
    return EXIT_FAILURE;
  }

  // Message pump loop
  while(sample.pollEvents())
  {
    sample.display();
  }

  // Terminate
  sample.close();

  return EXIT_SUCCESS;
}
