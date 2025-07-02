/*
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

#pragma once

// Contains the declaration of the main sample class.
// Its functions are defined in oit.cpp (resource creation for OIT
// specifically), oitRender.cpp (main command buffer rendering, without GUI),
// oitGui.cpp (GUI), and main.cpp (other resource creation and main()).
#include <imgui/imgui.h>

#include "shaders/common.h"
#include "utilities_vk.h"

#include <nvapp/application.hpp>
#include <nvapp/elem_camera.hpp>
#include <nvapp/elem_profiler.hpp>
#include <nvutils/camera_manipulator.hpp>
#include <nvvk/context.hpp>
#include <nvvk/descriptors.hpp>
#include <nvvk/gbuffers.hpp>
#include <nvvk/profiler_vk.hpp>
#include <nvvk/resource_allocator.hpp>

// An enumeration of each of the enumerations used in the GUI. We use this in
// the combo box registry.
enum GuiEnums : uint32_t
{
  GUI_ALGORITHM,
  GUI_OITSAMPLES,
  GUI_AA,
};

// A simple enumeration for a few blending modes.
enum class BlendMode
{
  NONE,           // With depth writing; (c, a) ov (d, b) = (c, a)
  PREMULTIPLIED,  // No depth writing; (c, a) ov (d, b) = (c + (1-a)d, a + (1-a)b)
  // For these next two, see oitScene.frag.glsl for a description of
  // Weighted, Blended Order-Independent Transparency:
  WEIGHTED_COLOR,      // No depth writing, 2 attachments; ((c, a), r) ov ((d, b), s) = ((c+d, a+b), (1-r)s)
  WEIGHTED_COMPOSITE,  // No depth writing; (c, r) ov (d, s) = (c(1-r) + rd, (1-r) + rs)
};

// Contains the current settings of the rendering algorithm.
// These are initially set to one of the best-looking settings.
struct State
{
  uint32_t algorithm                     = OIT_SPINLOCK;
  uint32_t oitLayers                     = 8;
  int32_t  linkedListAllocatedPerElement = 10;
  int32_t  percentTransparent            = 100;
  bool     tailBlend                     = true;
  bool     interlockIsOrdered            = true;
  int32_t  numObjects                    = 1024;
  int32_t  subdiv                        = 16;
  float    scaleMin                      = 0.1f;
  float    scaleWidth                    = 0.9f;
  uint32_t aaType                        = AA_NONE;
  bool     drawUI                        = true;

  // These are implicitly set by aaType:
  int  msaa          = 1;      // Number of MSAA samples used for color + depth buffers.
  bool sampleShading = false;  // If true, uses an array in the A-buffer per sample instead of per-pixel.
  int  supersample   = 1;
  bool coverageShading() { return ((msaa > 1) && (!sampleShading)); }

  void recomputeAntialiasingSettings()
  {
    sampleShading = false;
    supersample   = 1;
    switch(aaType)
    {
      case AA_NONE:
        msaa = 1;
        break;
      case AA_MSAA_4X:
        msaa = 4;
        break;
      case AA_SSAA_4X:
        msaa          = 4;
        sampleShading = true;
        break;
      case AA_SUPER_4X:
        msaa        = 1;
        supersample = 2;
        break;
      case AA_MSAA_8X:
        msaa = 8;
        break;
      case AA_SSAA_8X:
        msaa          = 8;
        sampleShading = true;
        break;
      default:
        assert(!"Antialiasing mode not implemented!");
        break;
    }
  }
};

using ShaderModuleID = uint32_t;

// Names for the UI panes.
constexpr const char* kUiPaneViewportName = "Viewport";
constexpr const char* kUiPaneSettingsName = "Settings";
constexpr const char* kUiPaneProfilerName = "Profiler";

enum class VertexShaderIndex
{
  eScene,
  eFullScreenTriangle,
  eCount,
};

enum class PassIndex
{
  eOpaque,
  eSimpleColor,
  eSimpleComposite,
  eLinkedListColor,
  eLinkedListComposite,
  eLoopDepth,
  eLoopColor,
  eLoopComposite,
  eLoop64Color,
  eLoop64Composite,
  eInterlockColor,
  eInterlockComposite,
  eSpinlockColor,
  eSpinlockComposite,
  eWeightedColor,
  eWeightedComposite,
  eCount
};

// Allows using unary + to convert to the base type
inline uint32_t operator+(VertexShaderIndex e)
{
  return static_cast<uint32_t>(e);
}

inline uint32_t operator+(PassIndex e)
{
  return static_cast<uint32_t>(e);
}


class Sample : public nvapp::IAppElement
{
public:
  // App and GPU handles
  nvapp::Application* m_app                        = nullptr;
  nvvk::Context*      m_ctx                        = nullptr;
  bool                m_deviceSupportsInt64Atomics = false;
  // Renderer state
  nvvk::ResourceAllocator m_allocator;
  // Per-frame objects.
  // We have one of these per frame since the CPU can be uploading to one while
  // the other is being used for rendering.
  std::vector<nvvk::Buffer> m_uniformBuffers;
  // We only need one of each of these resources, since only one draw operation will run at once.
  VkFramebuffer m_mainColorDepthFramebuffer = VK_NULL_HANDLE;
  VkFramebuffer m_weightedFramebuffer       = VK_NULL_HANDLE;
  ImageAndView  m_depthImage;
  ImageAndView  m_colorImage;
  BufferAndView m_oitABuffer;
  ImageAndView  m_oitAuxImage;
  ImageAndView  m_oitAuxSpinImage;
  ImageAndView  m_oitAuxDepthImage;
  ImageAndView  m_oitCounterImage;
  ImageAndView  m_oitWeightedColorImage;
  ImageAndView  m_oitWeightedRevealImage;
  // Depending on the MSAA settings and resolution, we may want to downsample
  // to a 1 sample per screen pixel texture:
  ImageAndView m_downsampleImage;
  // and then we'll need to copy data from _UNORM_SRGB to _UNORM so that
  // ImGui::Image displays it correctly. We use nvvk::GBuffer here because it
  // takes care of creating a descriptor set for ImGui; m_colorImage and
  // m_depthImage are our real G-buffer.
  nvvk::GBuffer m_viewportImage;
  // TODO: See if we can get rid of this
  // realtime_analysis.cpp uses a sampler pool
  VkSampler    m_pointSampler = VK_NULL_HANDLE;
  nvvk::Buffer m_vertexBuffer;
  nvvk::Buffer m_indexBuffer;
  // Shaders
  std::array<VkShaderModule, size_t(VertexShaderIndex::eCount)> m_vertexShaders{};
  std::array<VkShaderModule, size_t(PassIndex::eCount)>         m_fragmentShaders{};
  CachingShaderCompiler                                         m_shaderCompiler;
  // Descriptors
  nvvk::DescriptorPack m_descriptorPack;
  VkPipelineLayout     m_pipelineLayout = VK_NULL_HANDLE;
  // Render passes
  VkRenderPass m_renderPassColorDepthClear = VK_NULL_HANDLE;
  VkRenderPass m_renderPassWeighted        = VK_NULL_HANDLE;
  // Graphics pipelines (organized by the algorithms that use them)
  std::array<VkPipeline, size_t(PassIndex::eCount)> m_pipelines{};

  // Application state
  State                                       m_state;              // This frame's state
  State                                       m_lastState;          // Last frame's state
  bool                                        m_lastVsync = false;  // Last frame's vsync state
  std::shared_ptr<nvutils::CameraManipulator> m_cameraControl;      // A controllable camera
  std::shared_ptr<nvapp::ElementCamera>       m_cameraElement;      // The camera's GUI connection
  shaderio::SceneData m_sceneUbo{};  // Uniform Buffer Object for the scene, depends on m_cameraControl.
  uint32_t m_objectTriangleIndices = 0;  // The number of indices used in each sphere. (All objects have the same number of indices.)
  uint32_t m_sceneTriangleIndices = 0;  // The total number of indices in the scene.

  // Keeps track of CPU and GPU profiling information.
  nvutils::ProfilerManager                m_profiler;
  nvutils::ProfilerTimeline*              m_profilerTimeline = nullptr;
  nvvk::ProfilerGpuTimer                  m_profilerGPU;
  std::shared_ptr<nvapp::ElementProfiler> m_profilerGUI;

  // We make these constants so that we can create their render passes without
  // creating the images yet.
  const VkFormat m_oitWeightedColorFormat  = VK_FORMAT_R16G16B16A16_SFLOAT;
  const VkFormat m_oitWeightedRevealFormat = VK_FORMAT_R16_SFLOAT;
  const VkFormat m_viewportColorFormat     = VK_FORMAT_B8G8R8A8_UNORM;

public:
  Sample(nvvk::Context* ctx) { m_ctx = ctx; }

  /////////////////////////////////////////////////////////////////////////////
  // Callbacks                                                               //
  /////////////////////////////////////////////////////////////////////////////

  // Sets up the sample. Exits if setup failed.
  void onAttach(nvapp::Application* app) override;

  // Tear down the sample, essentially by running creation in reverse
  void onDetach() override;

  void onResize(VkCommandBuffer cmd, const VkExtent2D& size) override;

  // Draws the GUI. This includes the settings pane, and the instruction
  // for ImGui to composite our color buffer onto the screen.
  void onUIRender() override;

  // Main rendering
  void onRender(VkCommandBuffer cmd) override;

  // Handles UI for the top menu bar.
  void onUIMenu() override;

  /////////////////////////////////////////////////////////////////////////////
  // Object Creation, Destruction, and Recreation                            //
  /////////////////////////////////////////////////////////////////////////////

  // Compares m_state to m_lastState. If m_state changed, then
  // it updates the parts of the rendering system that need to change, such
  // as by reloading shaders and by regenerating internal buffers.
  // It also essentially tracks which objects depend on which parameters.
  void updateRendererFromState(bool swapchainSizeChanged, bool forceRebuildAll);

  void destroyTextureSampler();

  // This function is intended to only be called once.
  // Device must not be using resource when called.
  void createTextureSampler();

  // Device must not be using resource when called.
  void destroyUniformBuffers();

  // Depends only on the number of images in the swapchain.
  // Device must not be using resource when called.
  void createUniformBuffers();

  // Destroys the vertex and index buffers used for the scene.
  // Device must not be using resource when called.
  void destroyScene();

  // Recomputes the geometry used for the scene (which is a single mesh, described by
  // m_bufferVertices and m_bufferIndices).
  // Device must not be using resource when called.
  void initScene();

  // Device must not be using resource when called.
  void destroyFrameImages();

  // Creates the intermediate buffers used for order-independent transparency -
  // these are all of the IMG_* textures referenced in common.h. Unlike static
  // textures, their contents are recomputed each frame.
  // Device must not be using resource when called.
  void createFrameImages(VkCommandBuffer cmd);

  // Device must not be using resource when called.
  void destroyDescriptorSets();

  // This needs to be recreated whenever the algorithm changes to or from
  // OIT_LOOP64, as that algorithm uses a different descriptor type for
  // the A-buffer.
  // Device must not be using resource when called.
  void createDescriptorSets();

  // This needs to be called whenever our buffers change. This will basically
  // cause VkCmdBindDescriptorSets to bind all of the textures we need at once.
  void updateAllDescriptorSets();

  // Device must not be using resource when called.
  void destroyRenderPasses();

  // Creates or recreates all render passes.
  void createRenderPasses();

  // Device must not be using resource when called.
  void destroyFramebuffers();

  // Device must not be using resource when called.
  void createFramebuffers();

  // Device must not be using resource when called.
  void destroyShaderModules();

  // Call this function whenever you need to update the shader definitions or
  // when the algorithm changes - this will create or reload only the shader
  // modules that are needed.
  // The basic idea is that recompiling all of the shader modules every time
  // would take a lot of time, but we can speed it up by parsing and recompiling
  // only the shader modules we need.
  void createOrReloadShaderModules();

  // Device must not be using resource when called.
  void destroyGraphicsPipelines();

  // Destroys all graphics pipelines, and creates only the graphics pipeline
  // objects we need for a given algorithm.
  // Device must not be using resource when called.
  void createGraphicsPipelines();

  // Creates a graphics pipeline, exposing only the features that are needed.
  //   vertShaderModule and fragShaderModule: The vertex and fragment shaders
  //   blendMode: An enum selecting how blending and depth writing work.
  //   usesVertexInput: Specifies whether or not we read from a vertex buffer.
  //     E.g. this is true for drawing spheres and false for fullscreen triangles.
  //   renderPass and subpass: The render pass and subpass in which this graphics pipeline will be used.
  VkPipeline createGraphicsPipeline(const std::string&   debugName,
                                    const VkShaderModule vertShaderModule,
                                    const VkShaderModule fragShaderModule,
                                    BlendMode            blendMode,
                                    bool                 usesVertexInput,
                                    bool                 isDoubleSided,
                                    VkRenderPass         renderPass,
                                    uint32_t             subpass = 0);

  /////////////////////////////////////////////////////////////////////////////
  // Main rendering logic                                                    //
  /////////////////////////////////////////////////////////////////////////////

  // Returns max(1, m_app->getViewportSize()). This is so that we always have
  // a valid size we can use to construct an image.
  VkExtent2D getViewportSize() const;

  void updateUniformBuffer(uint32_t currentImage, double time);

  // Blit the offscreen color buffer to the main buffer, resolving MSAA
  // samples and downscaling in the process.
  // Note that this will only do a box filter - more complex antialiasing
  // filters require using a custom compute shader.
  void copyOffscreenToBackBuffer(VkCommandBuffer cmd);

  void clearTransparentSimple(VkCommandBuffer& cmdBuffer);

  // Draws the first numObjects objects using a simple OIT method.
  // Assumes that the right render pass has already been started, and that the
  // index and vertex buffers for the mesh and descriptors are already good to go.
  void drawTransparentSimple(VkCommandBuffer& cmdBuffer, int numObjects);

  void clearTransparentLinkedList(VkCommandBuffer& cmdBuffer);

  // Draws the first numObjects objects using an OIT method where each fragment
  // has a linked list of fragments (using the A-buffer as a large pool of memory).
  // Assumes that the right render pass has already been started, and that the
  // index and vertex buffers for the mesh and descriptors are already good to go.
  void drawTransparentLinkedList(VkCommandBuffer& cmdBuffer, int numObjects);

  void clearTransparentLoop(VkCommandBuffer& cmdBuffer);

  // Draws the first numObjects objects using the two-pass depth sorting OIT
  // method. Assumes that the right render pass has already been started, and
  // that the index and vertex buffers for the mesh and drescriptors are already
  // good to go.
  void drawTransparentLoop(VkCommandBuffer& cmdBuffer, int numObjects);

  void clearTransparentLoop64(VkCommandBuffer& cmdBuffer);

  // A variant of OIT_LOOP that uses one less draw pass when the GPU supports
  // 64-bit atomics. Assumes that the right render pass has already been started, and
  // that the index and vertex buffers for the mesh and drescriptors are already
  // good to go.
  void drawTransparentLoop64(VkCommandBuffer& cmdBuffer, int numObjects);

  void clearTransparentLock(VkCommandBuffer& cmdBuffer, bool useInterlock);

  // The interlock and spinlock algorithms both attempt to sort the frontmost
  // OIT_LAYERS fragments and tailblend the rest, but both do it in two passes
  // (as opposed to OIT_LOOP's 3) by making use of critical sections. Spinlock
  // (useInterlock == false) uses a manual spin-wait version of a mutex, while
  // Interlock (useInterlock == true) uses the GL_NV_fragment_shader_interlock
  // or GL_ARB_fragment_shader_interlock extensions to implement a
  // critical section.
  void drawTransparentLock(VkCommandBuffer& cmdBuffer, int numObjects, bool useInterlock);

  // Weighted, Blended Order-Independent Transparency doesn't use an A-buffer
  // and is an approximate technique; instead, it uses two intermediate render
  // targets, which we implement using a render pass (see the creation of the
  // render pass for more information as to how that's set up).
  void drawTransparentWeighted(VkCommandBuffer& cmdBuffer, int numObjects);
};
