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

#pragma once

// Contains the declaration of the main sample class.
// Its functions are defined in oit.cpp (resource creation for OIT
// specifically), oitRender.cpp (main command buffer rendering, without GUI),
// oitGui.cpp (GUI), and main.cpp (other resource creation and main()).

#include <imgui/imgui_helper.h>

#include <nvh/cameracontrol.hpp>
#include <nvh/fileoperations.hpp>
#include <nvh/geometry.hpp>
#include <nvmath/nvmath.h>
#include <nvmath/nvmath_glsltypes.h>
#include <nvvk/appwindowprofiler_vk.hpp>
#include <nvvk/commands_vk.hpp>
#include <nvvk/descriptorsets_vk.hpp>
#include <nvvk/memallocator_vk.hpp>
#include <nvvk/memorymanagement_vk.hpp>
#include <nvvk/shadermodulemanager_vk.hpp>
#include <nvvk/shaders_vk.hpp>
#include <nvvk/swapchain_vk.hpp>

#include "common.h"
#include "utilities_vk.h"

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
  uint32_t linkedListAllocatedPerElement = 10;
  uint32_t percentTransparent            = 100;
  bool     tailBlend                     = true;
  uint32_t numObjects                    = 1024;
  uint32_t subdiv                        = 16;
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

class Sample : public nvvk::AppWindowProfilerVK
{
public:
  // Renderer state
  nvvk::BatchSubmission       m_submission;
  nvvk::RingFences            m_ringFences;
  nvvk::RingCommandPool       m_ringCmdPool;
  nvvk::ResourceAllocatorDma  m_allocatorDma;
  nvvk::DebugUtil             m_debug = nvvk::DebugUtil();
  bool                        m_submissionWaitForRead;
  // Per-frame objects
  std::vector<nvvk::Buffer> m_uniformBuffers;
  // We only need one of each of these resources, since only one draw operation will run at once.
  VkViewport    m_viewportGUI               = {};
  VkRect2D      m_scissorGUI                = {};
  VkFramebuffer m_mainColorDepthFramebuffer = nullptr;
  VkFramebuffer m_weightedFramebuffer       = nullptr;
  VkFramebuffer m_guiFramebuffer            = nullptr;
  ImageAndView  m_depthImage;
  ImageAndView  m_colorImage;
  BufferAndView m_oitABuffer;
  ImageAndView  m_oitAuxImage;
  ImageAndView  m_oitAuxSpinImage;
  ImageAndView  m_oitAuxDepthImage;
  ImageAndView  m_oitCounterImage;
  ImageAndView  m_oitWeightedColorImage;
  ImageAndView  m_oitWeightedRevealImage;
  ImageAndView m_downsampleImage;  // A 1spp image with the same format as m_colorImage used for resolving m_colorImage.
  ImageAndView m_guiCompositeImage;  // A 1spp image with the same format as the swapchain.
  VkSampler    m_pointSampler = nullptr;
  nvvk::Buffer m_vertexBuffer;
  nvvk::Buffer m_indexBuffer;
  // Shaders
  nvvk::ShaderModuleManager m_shaderModuleManager;
  nvvk::ShaderModuleID      m_shaderSceneVert;
  nvvk::ShaderModuleID      m_shaderOpaqueFrag;
  nvvk::ShaderModuleID      m_shaderFullScreenTriangleVert;
  nvvk::ShaderModuleID      m_shaderSimpleColorFrag;
  nvvk::ShaderModuleID      m_shaderSimpleCompositeFrag;
  nvvk::ShaderModuleID      m_shaderLinkedListColorFrag;
  nvvk::ShaderModuleID      m_shaderLinkedListCompositeFrag;
  nvvk::ShaderModuleID      m_shaderLoopDepthFrag;
  nvvk::ShaderModuleID      m_shaderLoopColorFrag;
  nvvk::ShaderModuleID      m_shaderLoopCompositeFrag;
  nvvk::ShaderModuleID      m_shaderLoop64ColorFrag;
  nvvk::ShaderModuleID      m_shaderLoop64CompositeFrag;
  nvvk::ShaderModuleID      m_shaderInterlockColorFrag;
  nvvk::ShaderModuleID      m_shaderInterlockCompositeFrag;
  nvvk::ShaderModuleID      m_shaderSpinlockColorFrag;
  nvvk::ShaderModuleID      m_shaderSpinlockCompositeFrag;
  nvvk::ShaderModuleID      m_shaderWeightedColorFrag;
  nvvk::ShaderModuleID      m_shaderWeightedCompositeFrag;
  // Descriptors
  // Contains a layout, a pipeline layout, some reflection information, and a
  // pool for a number of VkDescriptorSets created using the same layout.
  nvvk::DescriptorSetContainer m_descriptorInfo;
  // Render passes
  VkRenderPass m_renderPassColorDepthClear = nullptr;
  VkRenderPass m_renderPassWeighted        = nullptr;
  VkRenderPass m_renderPassGUI             = nullptr;
  // Graphics pipelines (organized by the algorithms that use them)
  VkPipeline m_pipelineOpaque              = nullptr;
  VkPipeline m_pipelineSimpleColor         = nullptr;
  VkPipeline m_pipelineSimpleComposite     = nullptr;
  VkPipeline m_pipelineLinkedListColor     = nullptr;
  VkPipeline m_pipelineLinkedListComposite = nullptr;
  VkPipeline m_pipelineLoopDepth           = nullptr;
  VkPipeline m_pipelineLoopColor           = nullptr;
  VkPipeline m_pipelineLoopComposite       = nullptr;
  VkPipeline m_pipelineLoop64Color         = nullptr;
  VkPipeline m_pipelineLoop64Composite     = nullptr;
  VkPipeline m_pipelineInterlockColor      = nullptr;
  VkPipeline m_pipelineInterlockComposite  = nullptr;
  VkPipeline m_pipelineSpinlockColor       = nullptr;
  VkPipeline m_pipelineSpinlockComposite   = nullptr;
  VkPipeline m_pipelineWeightedColor       = nullptr;
  VkPipeline m_pipelineWeightedComposite   = nullptr;

  // GUI-specific variables
  ImGuiH::Registry m_imGuiRegistry;  // Helper class that tracks IDs for dear imgui
  double           m_uiTime = 0;

  // Application state
  State              m_state;              // This frame's state
  State              m_lastState;          // Last frame's state
  bool               m_lastVsync = false;  // Last frame's vsync state
  nvh::CameraControl m_cameraControl;      // A controllable camera
  SceneData          m_sceneUbo;           // Uniform Buffer Object for the scene, depends on m_cameraControl.
  uint32_t           m_objectTriangleIndices = 0;  // The number of indices used in each sphere. (All objects have the same number of indices.)
  uint32_t           m_sceneTriangleIndices = 0;  // The total number of indices in the scene.

  // We make these constants so that we can create their render passes without
  // creating the images yet.
  const VkFormat m_oitWeightedColorFormat  = VK_FORMAT_R16G16B16A16_SFLOAT;
  const VkFormat m_oitWeightedRevealFormat = VK_FORMAT_R16_SFLOAT;
  const VkFormat m_guiCompositeColorFormat = VK_FORMAT_B8G8R8A8_UNORM;

  uint32_t m_frame = 0;

public:
  Sample()
      : AppWindowProfilerVK(false)
  {
#if defined(NDEBUG)
    setVsync(false);
#endif
  }

  /////////////////////////////////////////////////////////////////////////////
  // Callbacks                                                               //
  /////////////////////////////////////////////////////////////////////////////
  void resize(int width, int height) override;
  bool mouse_pos(int x, int y) override;
  bool mouse_button(int button, int action) override;
  bool mouse_wheel(int wheel) override;
  bool key_char(int key) override;
  bool key_button(int button, int action, int mods) override;

  /////////////////////////////////////////////////////////////////////////////
  // Object Creation, Destruction, and Recreation                            //
  /////////////////////////////////////////////////////////////////////////////

  // Sets up the sample. Returns whether or not setup succeeded.
  bool begin() override;

  // Immediately creates and executes a command buffer that updates the state
  // of the renderer. Usually called from cmdUpdateRendererFromState. Should
  // usually be only called when m_state != m_lastState.
  void updateRendererImmediate(bool swapchainSizeChanged, bool forceRebuildAll);

  // Compares m_state to m_lastState. If m_state changed, then
  // it updates the parts of the rendering system that need to change, such
  // as by reloading shaders and by regenerating internal buffers.
  // It also essentially tracks which objects depend on which parameters.
  void cmdUpdateRendererFromState(VkCommandBuffer cmdBuffer, bool swapchainSizeChanged, bool forceRebuildAll);

  // Tear down the sample, essentially by running creation in reverse
  void end() override;

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
  // m_bufferVertices and m_bufferIndices). Then adds upload instructions to the command buffer.
  // Device must not be using resource when called.
  void initScene(VkCommandBuffer commandBuffer);

  // Device must not be using resource when called.
  void destroyFrameImages();

  // Creates the intermediate buffers used for order-independent transparency -
  // these are all of the IMG_* textures referenced in common.h. Unlike static
  // textures, their contents are recomputed each frame.
  // Device must not be using resource when called.
  void createFrameImages(VkCommandBuffer cmdBuffer);

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
  void destroyGUIRenderPass();

  // Creates the ImGui render pass. This should not be called twice.
  void createGUIRenderPass();

  // Device must not be using resource when called.
  void destroyNonGUIRenderPasses();

  // Creates or recreates all non-ImGui render passes.
  void createNonGUIRenderPasses();

  // Device must not be using resource when called.
  void destroyFramebuffers();

  // Device must not be using resource when called.
  void createFramebuffers();

  // Updates global shader define; all shaders have to be recompiled after setting this.
  void updateShaderDefinitions();

  // Helper function to add new shader module to m_shadermoduleManager if
  // `shaderModule` isn't already set, or to reload `shaderModule` if it is set.
  //   shaderStage: Which shader stage this shader will be used for, e.g. fragment or vertex
  //   filename: The file containing the GLSL code for the shader.
  //   prepend: Additional text to be placed after the #version directive,
  //     such as preprocessor defines.
  void createOrReloadShaderModule(nvvk::ShaderModuleID& shaderModule,
                                  VkShaderStageFlags    shaderStage,
                                  const std::string&    filename,
                                  const std::string&    prepend = "");

  // Call this function whenever you need to update the shader definitions or
  // when the algorithm changes - this will create or reload only the shader
  // modules that are needed.
  // The basic idea is that recompiling all of the shader modules every time
  // would take a lot of time, but we can speed it up by parsing and recompiling
  // only the shader modules we need.
  void createOrReloadShaderModules();

  void destroyGraphicsPipeline(VkPipeline& pipeline);

  // Device must not be using resource when called.
  void destroyGraphicsPipelines();

  // Destroys all graphics pipelines, and creates only the graphics pipeline
  // objects we need for a given algorithm.
  // Device must not be using resource when called.
  void createGraphicsPipelines();

  // Creates a graphics pipeline, exposing only the features that are needed.
  //   vertShaderModule and fragShaderModule: The vertex and fragment shader
  // NVVK module IDs to use, respectively.
  //   blendMode: An enum selecting how blending and depth writing work.
  //   usesVertexInput: Specifies whether or not we read from a vertex buffer.
  //     E.g. this is true for drawing spheres and false for fullscreen triangles.
  //   renderPass and subpass: The render pass and subpass in which this graphics pipeline will be used.
  VkPipeline createGraphicsPipeline(const nvvk::ShaderModuleID& vertShaderModuleID,
                                    const nvvk::ShaderModuleID& fragShaderModuleID,
                                    BlendMode                   blendMode,
                                    bool                        usesVertexInput,
                                    bool                        isDoubleSided,
                                    VkRenderPass                renderPass,
                                    uint32_t                    subpass = 0);

  // Creates and begins a command buffer that will only be submitted once.
  VkCommandBuffer createTempCmdBuffer();

  // Sets m_viewportGUI and m_scissorGUI based on the current screen size.
  void setUpViewportsAndScissors();

  /////////////////////////////////////////////////////////////////////////////
  // GUI                                                                     //
  /////////////////////////////////////////////////////////////////////////////

  // If the cursor was hovering over the last item, displays a tooltip.
  void LastItemTooltip(const char* text);

  // If the object exists, draws ImGui text like
  // m_oitABuffer: 67000000 bytes
  void DoObjectSizeText(BufferAndView bv, const char* name);

  // If the object exists, draws ImGui text like
  // m_oitAuxImage: 1200 x 1024, 2 layers.
  void DoObjectSizeText(ImageAndView iv, const char* name);

  // Displays the Dear ImGui interface.
  // This interface includes tooltips for each of the elements, and also shows
  // or hides fields based on the current OIT algorithm.
  void DoGUI(int width, int height, double time);

  /////////////////////////////////////////////////////////////////////////////
  // Main rendering logic                                                    //
  /////////////////////////////////////////////////////////////////////////////

  void updateUniformBuffer(uint32_t currentImage, double time);

  // Blit the offscreen color buffer to the main buffer, resolving MSAA
  // samples and downscaling in the process.
  // Note that this will only do a box filter - more complex antialiasing
  // filters require using a custom compute shader.
  void copyOffscreenToBackBuffer(int winWidth, int winHeight, ImDrawData* imguiDrawData);

  // Performs a queue submission.
  void submissionExecute(VkFence fence = NULL, bool useImageReadWait = false, bool useImageWriteSignals = false);

  // Main loop
  void think(double time) override;

  // Renders the scene including transparency to m_colorImage.
  void render(VkCommandBuffer& cmdBuffer);

  // Adds calls to bind vertex and index buffers and draw numObjects objects, starting
  // with firstObject. (In this sample, an object is a single sphere).
  // Assumes that a render pass has already been started, and that the bound pipeline
  // state object is compatible with the given vertex layout.
  void drawSceneObjects(VkCommandBuffer& cmdBuffer, int firstObject, int numObjects);

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
