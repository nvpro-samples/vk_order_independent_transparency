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

// This file contains the implementation of the GUI for this sample.

#include "oit.h"

#include <nvgui/camera.hpp>
#include <nvgui/file_dialog.hpp>
#include <nvgui/property_editor.hpp>
#include <nvgui/tooltip.hpp>

#include <array>

void Sample::onUIMenu()
{
  bool saveViewport  = false;
  bool saveScreen    = false;
  bool reloadShaders = false;
  bool vsync         = m_app->isVsync();

  if(ImGui::BeginMenu("Tools"))
  {
    saveViewport |= ImGui::MenuItem("Save Viewport...", "Ctrl+Shift+S");
    saveScreen |= ImGui::MenuItem("Save Screen...", "Ctrl+Alt+Shift+S");
    reloadShaders |= ImGui::MenuItem("Reload", "Ctrl+R");
    ImGui::MenuItem("V-Sync", "Ctrl+Shift+V", &vsync);
    ImGui::EndMenu();
  }

  saveViewport |= ImGui::IsKeyChordPressed(ImGuiMod_Ctrl | ImGuiMod_Shift | ImGuiKey_S);
  saveScreen |= ImGui::IsKeyChordPressed(ImGuiMod_Ctrl | ImGuiMod_Alt | ImGuiMod_Shift | ImGuiKey_S);
  reloadShaders |= ImGui::IsKeyChordPressed(ImGuiMod_Ctrl | ImGuiKey_R);
  vsync ^= ImGui::IsKeyChordPressed(ImGuiMod_Ctrl | ImGuiMod_Shift | ImGuiKey_V);

  if(saveViewport)
  {
    const std::filesystem::path filename =
        nvgui::windowSaveFileDialog(m_app->getWindowHandle(), "Save Viewport", "PNG(.png),JPG(.jpg)|*.png;*.jpg;*.jpeg");
    if(!filename.empty())
    {
      m_app->saveImageToFile(m_viewportImage.getColorImage(), m_viewportImage.getSize(), filename);
    }
  }

  if(saveScreen)
  {
    const std::filesystem::path filename = nvgui::windowSaveFileDialog(m_app->getWindowHandle(), "Save Screen Including UI",
                                                                       "PNG(.png),JPG(.jpg)|*.png;*.jpg;*.jpeg");
    if(!filename.empty())
    {
      m_app->screenShot(filename);
    }
  }

  if(reloadShaders)
  {
    vkDeviceWaitIdle(m_app->getDevice());
    destroyShaderModules();
    updateRendererFromState(true, true);
  }

  if(m_app->isVsync() != vsync)
  {
    m_app->setVsync(vsync);
  }
}

// If the object exists, draws ImGui text like
// m_oitABuffer: 67000000 bytes
static void DoObjectSizeText(BufferAndView bv, const char* name)
{
  if(bv.buffer.buffer)
  {
    ImGui::Text("%s: %zu bytes", name, bv.size);
  }
}

// If the object exists, draws ImGui text like
// m_oitAuxImage: 1200 x 1024, 2 layers.
static void DoObjectSizeText(ImageAndView iv, const char* name)
{
  if(iv.image.image)
  {
    ImGui::Text("%s: %u x %u, %u layer%s",
                name,                           //
                iv.getWidth(), iv.getHeight(),  //
                iv.getLayers(), (iv.getLayers() != 1 ? "s" : ""));
  }
}

// Draws the GUI. This includes the settings pane, and the instruction
// for ImGui to composite our color buffer onto the screen.
void Sample::onUIRender()
{
  // We use nvgui::PropertyEditor for the custom nvpro-samples ImGui style.
  namespace PE = nvgui::PropertyEditor;

  // Settings pane
  if(ImGui::Begin(kUiPaneSettingsName))
  {
    // Algorithm settings
    if(ImGui::CollapsingHeader("Algorithm Settings", ImGuiTreeNodeFlags_DefaultOpen))
    {
      PE::begin();  // Set up nvpro-samples ImGui styles

      std::array<const char*, NUM_ALGORITHMS> algorithmNames;
      algorithmNames[OIT_SIMPLE]     = "simple";
      algorithmNames[OIT_LINKEDLIST] = "linkedlist";
      algorithmNames[OIT_LOOP]       = "loop32 two pass";
      algorithmNames[OIT_LOOP64]     = "loop64";
      algorithmNames[OIT_SPINLOCK]   = "spinlock";
      algorithmNames[OIT_INTERLOCK]  = "interlock";
      algorithmNames[OIT_WEIGHTED]   = "weighted blend";

      std::array<const char*, NUM_ALGORITHMS> algorithmDescriptions;
      algorithmDescriptions[OIT_SIMPLE] =
          "A simple A-buffer method.\n"
          "\n"
          "Each pixel or sample stores the first "
          "OIT_LAYERS fragments it processes, and tail-blends the rest. It "
          "then sorts these fragments by depth and blends the result onto "
          "the opaque objects.";
      algorithmDescriptions[OIT_LINKEDLIST] =
          "Uses the A-buffer as a single large block of memory.\n"
          "\n"
          "By atomically incrementing a counter (here a 1x1 image), each pixel "
          "or sample can construct a linked list of its fragments in parallel. "
          "When they run out of space in the A-buffer, threads tail-blend their "
          "remaining fragments. For each pixel or sample, the compositing shader "
          "then iterates over its linked list, sorts the frontmost OIT_LAYERS "
          "fragments by depth, and tail-blends the rest.";
      algorithmDescriptions[OIT_LOOP] =
          "A three-shader A-buffer method that does not support MSAA.\n"
          "\n"
          "Each sample first sorts the depths of its frontmost OIT_LAYERS "
          "fragments, which it can do in parallel using 32-bit atomics. "
          "Then it orders the colors of its fragments by matching them "
          "to their depths in this array, and tail-blends the rest. "
          "The compositing shader then blends the sorted fragments together.";
      algorithmDescriptions[OIT_LOOP64] =
          "A two-shader A-buffer method that does not support MSAA.\n"
          "\n"
          "This algorithm only appears if your device supports 64-bit atomics. "
          "We can pack the 32-bit depth and 8-bit-per-channel color together "
          "into a 64-bit integer. Each sample then sorts the frontmost "
          "OIT_LAYERS fragments together in parallel using 64-bit atomics. "
          "The compositing shader then blends the sorted fragments together.";
      algorithmDescriptions[OIT_INTERLOCK] =
          "A two-shader A-buffer method with a critical section.\n"
          "\n"
          "Instead of using spinlocks, we can use the "
          "GL_ARB_fragment_shader_interlock or GL_NV_fragment_shader_interlock "
          "extension (Vulkan's version of rasterizer order views) to make sure "
          "that at most one fragment shader invocation per pixel or sample "
          "inserts data into the respective part of the A-buffer at a time. "
          "It tail-blends fragments that don't make it into the A-buffer, and "
          "the compositing shader then blends the sorted fragments together.";
      algorithmDescriptions[OIT_SPINLOCK] =
          "A two-shader A-buffer method with a critical section.\n"
          "\n"
          "By using atomic operations to mimic spinlocks, each pixel or sample "
          "can sort its frontmost OIT_LAYERS fragments (including sample masks) "
          "by only allowing one instantiation to insert a value into the "
          "relevant part of the A-buffer at a time. It tail-blends fragments "
          "that don't make it into the A-buffer, and the compositing shader "
          "then blends the sorted fragments together.";
      algorithmDescriptions[OIT_WEIGHTED] =  //
          "Weighted, Blended Order-Independent Transparency is an "
          "approximate OIT algorithm that does not use an A-buffer. That is, "
          "it uses less memory and is usually faster than the other "
          "algorithms, but the other algorithms converge to the ground truth "
          "given enough memory.\n"
          "\n"
          "For a pixel or sample, let its fragments be numbered from i=1 to N. "
          "The algorithm chooses a weight w_i for each fragment, then computes\n"
          "\n"
          "    float4 accum = sum(w_i * rgba_i, i = 1...N)\n"
          "    float reveal = product(1 - a_i, i = 1...N).\n"
          "\n"
          "If all the fragments were blended together, they would have opacity "
          "1-reveal. So the algorithm then essentially composites a single "
          "RGBA color,\n"
          "    float4 color = float4(accum.rgb / accum.a, 1 - reveal.a)\n"
          "onto the opaque image. This sample implements this using two "
          "render pass subpasses.";

      // Algorithm combobox
      PE::entry("Algorithm", [&] {
        if(ImGui::BeginCombo("##Algorithm", algorithmNames[m_state.algorithm]))
        {
          for(uint32_t alg = 0; alg < NUM_ALGORITHMS; alg++)
          {
            // If this algorithm isn't supported, continue on to the
            // next one.
            if(OIT_LOOP64 == alg && VK_FALSE == m_ctx->getPhysicalDeviceFeatures12().shaderBufferInt64Atomics)
            {
              continue;
            }
            else if(OIT_INTERLOCK == alg && !m_ctx->hasExtensionEnabled(VK_EXT_FRAGMENT_SHADER_INTERLOCK_EXTENSION_NAME))
            {
              continue;
            }

            if(ImGui::Selectable(algorithmNames[alg], (m_state.algorithm == alg)))
            {
              m_state.algorithm = alg;
            }
            nvgui::tooltip(algorithmDescriptions[alg]);
          }
          ImGui::EndCombo();
        }
        // Normally this would return whether the value changed, but since
        // we detect changes by looking at the entire m_state struct instead,
        // we can return any value here.
        return true;
      });
      nvgui::tooltip(algorithmDescriptions[m_state.algorithm]);


      if(m_state.algorithm != OIT_WEIGHTED)
      {
        PE::Checkbox("Tail blend", &m_state.tailBlend,
                     "Chooses whether to discard fragments that cannot fit "
                     "into the A-buffer, or to blend them out-of-order using standard "
                     "transparency blending instead.");
      }
      if(m_state.algorithm == OIT_INTERLOCK)
      {
        PE::Checkbox("Interlock is ordered", &m_state.interlockIsOrdered,
                     "If checked, the 'interlock' algorithm uses ordered interlock "
                     "(layout(sample_interlock_ordered) and layout(pixel_interlock_ordered)), "
                     "which means that fragments will be processed in primitive order. "
                     "In particular, this makes it so that tail-blended fragments are "
                     "blended in a consistent order. When this is unchecked, the "
                     "interlock algorithm uses unordered interlock instead.");
      }

      if(m_state.algorithm != OIT_WEIGHTED && m_state.algorithm != OIT_LINKEDLIST)
      {
        constexpr uint32_t MAX_LAYERS_LOG2 = 5;
        PE::entry(
            "Layers",
            [&] {
              if(ImGui::BeginCombo("##layers", std::to_string(m_state.oitLayers).c_str()))
              {
                for(uint32_t log2 = 0; log2 <= MAX_LAYERS_LOG2; log2++)
                {
                  const uint32_t layers = 1U << log2;
                  if(ImGui::Selectable(std::to_string(layers).c_str(), (m_state.oitLayers == layers)))
                  {
                    m_state.oitLayers = layers;
                  }
                }
                ImGui::EndCombo();
              }
              return true;
            },
            "How many slots in the A-buffer to reserve for each pixel "
            "or sample. Each pixel or sample has its own space, and tail-blends "
            "its remaining fragments once it runs out of space.");
      }

      if(m_state.algorithm == OIT_LINKEDLIST)
      {
        PE::InputInt("List: Allocated per pixel", &m_state.linkedListAllocatedPerElement, 1, 8, ImGuiInputTextFlags_None,
                     "How many A-buffer slots to allocate per pixel or sample on average (since the "
                     "linked-list algorithm uses the A-buffer as a single block of memory)."
                     "Once the A-buffer runs out of space, the remaining fragments are tail-blended.");
        // Make sure at least 1 is allocated
        m_state.linkedListAllocatedPerElement = std::max(1, m_state.linkedListAllocatedPerElement);
      }

      // Anti-aliasing
      std::array<const char*, NUM_AATYPES> antialiasingNames;
      antialiasingNames[AA_NONE]     = "none";
      antialiasingNames[AA_MSAA_4X]  = "msaa 4x pixel-shading";
      antialiasingNames[AA_SSAA_4X]  = "msaa 4x sample-shading";
      antialiasingNames[AA_SUPER_4X] = "super 4x";
      antialiasingNames[AA_MSAA_8X]  = "msaa 8x pixel-shading";
      antialiasingNames[AA_SSAA_8X]  = "msaa 8x sample-shading";

      std::array<const char*, NUM_AATYPES> antialiasingDescriptions;
      antialiasingDescriptions[AA_NONE]     = "No antialiasing.";
      antialiasingDescriptions[AA_MSAA_4X]  = "MSAA using 4 samples per pixel. Processes fragments per-pixel.";
      antialiasingDescriptions[AA_SSAA_4X]  = "MSAA using 4 samples per pixel. Processes fragments per-sample.";
      antialiasingDescriptions[AA_SUPER_4X] = "Renders at twice the resolution and height.";
      antialiasingDescriptions[AA_MSAA_8X]  = "MSAA using 8 samples per pixel. Processes fragments per-pixel.";
      antialiasingDescriptions[AA_SSAA_8X]  = "MSAA using 8 samples per pixel. Processes fragments per-sample.";

      PE::entry("Anti-aliasing", [&] {
        if(ImGui::BeginCombo("##aa", antialiasingNames[m_state.aaType]))
        {
          for(uint32_t aaType = 0; aaType < NUM_AATYPES; aaType++)
          {
            if(ImGui::Selectable(antialiasingNames[aaType], (m_state.aaType == aaType)))
            {
              m_state.aaType = aaType;
            }
            nvgui::tooltip(antialiasingDescriptions[aaType]);
          }
          ImGui::EndCombo();
        }
        return true;
      });
      nvgui::tooltip(antialiasingDescriptions[m_state.aaType]);

      PE::end();
    }

    if(ImGui::CollapsingHeader("Object Sizes", ImGuiTreeNodeFlags_DefaultOpen))
    {
      DoObjectSizeText(m_oitABuffer, "A-buffer");
      DoObjectSizeText(m_oitAuxImage, "Aux image");
      DoObjectSizeText(m_oitAuxSpinImage, "Spinlock image");
      DoObjectSizeText(m_oitAuxDepthImage, "Furthest depths");
      DoObjectSizeText(m_oitCounterImage, "Atomic counter");
      DoObjectSizeText(m_oitWeightedColorImage, "Weighted color");
      DoObjectSizeText(m_oitWeightedRevealImage, "Reveal image");
    }

    if(ImGui::CollapsingHeader("Scene", ImGuiTreeNodeFlags_DefaultOpen))
    {
      PE::begin();

      PE::InputInt("Number of objects", &m_state.numObjects, 128, 1024);
      m_state.numObjects = std::max(1, m_state.numObjects);
      nvgui::tooltip("The number of spheres in the mesh.");

      PE::InputInt("Percent transparent", &m_state.percentTransparent);
      m_state.percentTransparent = std::max(0, std::min(m_state.percentTransparent, 100));
      nvgui::tooltip(
          "The percentage of spheres in the scene that are transparent. "
          "(Internally, the scene is 1 mesh; this controls the number of triangles "
          "that are drawn with the opaque vs. the transparent shader.)");

      PE::SliderFloat("Alpha min", &m_sceneUbo.alphaMin, 0.0f, 1.0f);
      nvgui::tooltip("The lower bound of object opacities.");

      PE::SliderFloat("Alpha width", &m_sceneUbo.alphaWidth, 0.0f, 1.0f);
      nvgui::tooltip(
          "How large a range the object opacities can span over. "
          "Opacities are always within the range [alphaMin, alphaMin+alphaWidth].");

      PE::InputInt("Subdivision level", &m_state.subdiv, 1, 8);
      m_state.subdiv = std::max(2, std::min(m_state.subdiv, 32));
      nvgui::tooltip(
          "How finely to subdivide the spheres. The number of triangles "
          "corresponds quadratically with this parameter.");

      PE::SliderFloat("Scale min", &m_state.scaleMin, 0.0f, 4.0f);
      nvgui::tooltip("The radius of the smallest spheres.");

      PE::SliderFloat("Scale width", &m_state.scaleWidth, 0.0f, 4.0f);
      nvgui::tooltip("How much the radii of the spheres can vary.");

      PE::end();
    }

    // Camera widget
    if(ImGui::CollapsingHeader("Camera", ImGuiTreeNodeFlags_DefaultOpen))
    {
      nvgui::CameraWidget(m_cameraControl);
    }
  }
  ImGui::End();

  // This code for the main viewport tells ImGui to composite our color image
  // to the screen once the window class calls ImGui::Render().
  ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0.0F, 0.0F));
  if(ImGui::Begin(kUiPaneViewportName))
  {
    // Display the G-Buffer image
    ImGui::Image(ImTextureID(m_viewportImage.getDescriptorSet()), ImGui::GetContentRegionAvail());
  }
  ImGui::End();
  ImGui::PopStyleVar();
}
