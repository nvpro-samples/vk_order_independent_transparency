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


// This file contains the implementation of the GUI for this sample.

#include "oit.h"

// If the cursor was hovering over the last item, displays a tooltip.
void Sample::LastItemTooltip(const char* text)
{
  if(ImGui::IsItemHovered())
  {
    ImGui::BeginTooltip();
    ImGui::PushTextWrapPos(ImGui::GetFontSize() * 35.0f);
    ImGui::TextUnformatted(text);
    ImGui::PopTextWrapPos();
    ImGui::EndTooltip();
  }
}

// If the object exists, draws ImGui text like
// m_oitABuffer: 67000000 bytes
void Sample::DoObjectSizeText(BufferAndView bv, const char* name)
{
  if(bv.buffer.buffer)
  {
    ImGui::Text("%s: %zu bytes", name, bv.size);
  }
}

// If the object exists, draws ImGui text like
// m_oitAuxImage: 1200 x 1024, 2 layers.
void Sample::DoObjectSizeText(ImageAndView iv, const char* name)
{
  if(iv.view)
  {
    ImGui::Text("%s: %u x %u, %u layer%s",
                name,                     //
                iv.c_width, iv.c_height,  //
                iv.c_layers, (iv.c_layers != 1 ? "s" : ""));
  }
}

void Sample::DoGUI(int width, int height, double time)
{
  ImGui::GetIO().DeltaTime   = static_cast<float>(time - m_uiTime);
  ImGui::GetIO().DisplaySize = ImVec2(static_cast<float>(width), static_cast<float>(height));

  m_uiTime = time;

  ImGui::NewFrame();

  ImGui::SetNextWindowPos(ImVec2(5, 5), ImGuiCond_FirstUseEver);
  ImGui::SetNextWindowSize(ImGuiH::dpiScaled(350, 0), ImGuiCond_FirstUseEver);

  if(ImGui::Begin("NVIDIA " PROJECT_NAME, nullptr))
  {
    ImGui::PushItemWidth(ImGuiH::dpiScaled(150));

    // Algorithm combobox
    m_imGuiRegistry.enumCombobox(GUI_ALGORITHM, "algorithm", &m_state.algorithm);
    const char* algorithmDescriptions[NUM_ALGORITHMS];
    algorithmDescriptions[OIT_SIMPLE] =
        "A simple A-buffer method. Each pixel or sample stores the first "
        "OIT_LAYERS fragments it processes, and tail-blends the rest. It "
        "then sorts these fragments by depth and blends the result onto "
        "the opaque objects.";
    algorithmDescriptions[OIT_LINKEDLIST] =
        "Uses the A-buffer as a single large block of memory. By atomically "
        "incrementing a counter (here a 1x1 image), each pixel or sample can "
        "construct a linked list of its fragments in parallel. When they "
        "run out of space in the A-buffer, threads tail-blend their "
        "remaining fragments. For each pixel or sample, the compositing shader "
        "then iterates over its linked list, sorts the frontmost OIT_LAYERS "
        "fragments by depth, and tail-blends the rest.";
    algorithmDescriptions[OIT_LOOP] =
        "A three-shader A-buffer method that does not support MSAA. "
        "Each sample first sorts the depths of its frontmost OIT_LAYERS "
        "fragments, which it can do in parallel using 32-bit atomics. "
        "Then it orders the colors of its fragments by matching them "
        "to their depths in this array, and tail-blends the rest. "
        "The compositing shader then blends the sorted fragments together.";
    algorithmDescriptions[OIT_LOOP64] =
        "A two-shader A-buffer method that does not support MSAA. "
        "This algorithm only appears if your device supports 64-bit atomics. "
        "We can pack the 32-bit depth and 8-bit-per-channel color together "
        "into a 64-bit integer. Each sample then sorts the frontmost "
        "OIT_LAYERS fragments together in parallel using 64-bit atomics. "
        "The compositing shader then blends the sorted fragments together.";
    algorithmDescriptions[OIT_INTERLOCK] =
        "A two-shader A-buffer method with a critical section. Instead of "
        "using spinlocks, we can use the GL_ARB_fragment_shader_interlock "
        "or GL_NV_fragment_shader_interlock extension (Vulkan's version of "
        "rasterizer order views) to make sure that at most one fragment "
        "shader invocation per pixel or sample inserts data into the "
        "respective part of the A-buffer at a time. It tail-blends "
        "fragments that don't make it into the A-buffer, and the "
        "compositing shader then blends the sorted fragments together.";
    algorithmDescriptions[OIT_SPINLOCK] =
        "A two-shader A-buffer method with a critical section. By using "
        "atomic operations to mimic spinlocks, each pixel or sample can "
        "sort its frontmost OIT_LAYERS fragments (including sample masks) "
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
        "For a pixel or sample, let its fragments be numbered from i=1 to N. "
        "The algorithm chooses a weight w_i for each fragment, then computes\n"
        "    float4 accum = sum(w_i * rgba_i, i = 1...N)\n"
        "    float reveal = product(1 - a_i, i = 1...N).\n"
        "If all the fragments were blended together, they would have opacity "
        "1-reveal. So the algorithm then essentially composites a single "
        "RGBA color,\n"
        "    float4 color = float4(accum.rgb / accum.a, 1 - reveal.a)\n"
        "onto the opaque image. This sample implements this using two "
        "render pass subpasses.";
    LastItemTooltip(algorithmDescriptions[m_state.algorithm]);

    ImGuiH::InputIntClamped("Percent transparent", &m_state.percentTransparent, 0, 100);
    LastItemTooltip(
        "The percentage of spheres in the scene that are transparent. "
        "(Internally, the scene is 1 mesh; this controls the number of triangles "
        "that are drawn with the opaque vs. the transparent shader.)");
    ImGui::SliderFloat("Alpha min", &m_sceneUbo.alphaMin, 0.0f, 1.0f);
    LastItemTooltip("The lower bound of object opacities.");
    ImGui::SliderFloat("Alpha width", &m_sceneUbo.alphaWidth, 0.0f, 1.0f);
    LastItemTooltip(
        "How large a range the object opacities can span over. "
        "Opacities are always within the range [alphaMin, alphaMin+alphaWidth].");
    if(m_state.algorithm != OIT_WEIGHTED)
    {
      ImGui::Checkbox("Tail blend", &m_state.tailBlend);
      LastItemTooltip(
          "Chooses whether to discard fragments that cannot fit "
          "into the A-buffer, or to blend them out-of-order using standard "
          "transparency blending instead.");
    }
    if(m_state.algorithm == OIT_INTERLOCK)
    {
      ImGui::Checkbox("Interlock is ordered", &m_state.interlockIsOrdered);
      LastItemTooltip(
          "If checked, the 'interlock' algorithm uses ordered interlock "
          "(layout(sample_interlock_ordered) and layout(pixel_interlock_ordered)), "
          "which means that fragments will be processed in primitive order. "
          "In particular, this makes it so that tail-blended fragments are "
          "blended in a consistent order. When this is unchecked, the "
          "interlock algorithm uses unordered interlock instead.");
    }

    if(m_state.algorithm != OIT_WEIGHTED && m_state.algorithm != OIT_LINKEDLIST)
    {
      m_imGuiRegistry.enumCombobox(GUI_OITSAMPLES, "layers", &m_state.oitLayers);
      LastItemTooltip(
          "How many slots in the A-buffer to reserve for each pixel "
          "or sample. Each pixel or sample has its own space, and tail-blends "
          "its remaining fragments once it runs out of space.");
    }

    if(m_state.algorithm == OIT_LINKEDLIST)
    {
      ImGuiH::InputIntClamped("List: Allocated per pixel", &m_state.linkedListAllocatedPerElement, 1, 128, 1, 8);
      LastItemTooltip(
          "How many A-buffer slots to allocate per pixel or sample on average (since the "
          "linked-list algorithm uses the A-buffer as a single block of memory)."
          "Once the A-buffer runs out of space, the remaining fragments are tail-blended.");
    }

    // Anti-aliasing
    m_imGuiRegistry.enumCombobox(GUI_AA, "anti-aliasing", &m_state.aaType);
    const char* antialiasingDescriptions[NUM_AATYPES];
    antialiasingDescriptions[AA_NONE]     = "No antialiasing.";
    antialiasingDescriptions[AA_MSAA_4X]  = "MSAA using 4 samples per pixel. Processes fragments per-pixel.";
    antialiasingDescriptions[AA_MSAA_8X]  = "MSAA using 8 samples per pixel. Processes fragments per-pixel.";
    antialiasingDescriptions[AA_SSAA_4X]  = "MSAA using 4 samples per pixel. Processes fragments per-sample.";
    antialiasingDescriptions[AA_SSAA_8X]  = "MSAA using 8 samples per pixel. Processes fragments per-sample.";
    antialiasingDescriptions[AA_SUPER_4X] = "Renders at twice the resolution and height.";
    LastItemTooltip(antialiasingDescriptions[m_state.aaType]);

    ImGui::Separator();
    ImGui::Text("Scene");

    ImGuiH::InputIntClamped("Number of objects", &m_state.numObjects, 1, 65536, 128, 1024);
    LastItemTooltip("The number of spheres in the mesh.");
    ImGuiH::InputIntClamped("Subdivision level", &m_state.subdiv, 2, 32, 1, 8);
    LastItemTooltip(
        "How finely to subdivide the spheres. The number of triangles "
        "corresponds quadratically with this parameter.");
    ImGui::SliderFloat("Scale min", &m_state.scaleMin, 0.1f, 4.0f);
    LastItemTooltip("The radius of the smallest spheres.");
    ImGui::SliderFloat("Scale width", &m_state.scaleWidth, 0, 4.0f);
    LastItemTooltip("How much the radii of the spheres can vary.");

    ImGui::Separator();
    ImGui::Text("Object Sizes");
    DoObjectSizeText(m_oitABuffer, "A-buffer");
    DoObjectSizeText(m_oitAuxImage, "Aux image");
    DoObjectSizeText(m_oitAuxSpinImage, "Spinlock image");
    DoObjectSizeText(m_oitAuxDepthImage, "Furthest depths");
    DoObjectSizeText(m_oitCounterImage, "Atomic counter");
    DoObjectSizeText(m_oitWeightedColorImage, "Weighted color");
    DoObjectSizeText(m_oitWeightedRevealImage, "Reveal image");
  }
  ImGui::End();
}