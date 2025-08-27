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


// This file contains implementations of the main OIT drawing functions from
// oit.h, excluding GUI and resolving from m_colorImage to the swapchain.

#include "oit.h"

#include <GLFW/glfw3.h>

void Sample::onRender(VkCommandBuffer cmd)
{
  m_profilerTimeline->frameAdvance();
  NVVK_DBG_SCOPE(cmd);
  auto profilerRangeRender = m_profilerGPU.cmdFrameSection(cmd, __FUNCTION__);

  // If elements of m_state have changed, this reinitializes parts of the renderer
  updateRendererFromState(false, false);

  // Update the GPU's uniform buffer
  updateUniformBuffer(m_app->getFrameCycleIndex(), glfwGetTime());

  // Record this frame's command buffer
  // Clear auxiliary buffers before we even start a render pass - this
  // reduces the number of render passes we need to use by 1.
  switch(m_state.algorithm)
  {
    case OIT_SIMPLE:
      clearTransparentSimple(cmd);
      break;
    case OIT_LINKEDLIST:
      clearTransparentLinkedList(cmd);
      break;
    case OIT_LOOP:
      clearTransparentLoop(cmd);
      break;
    case OIT_LOOP64:
      clearTransparentLoop64(cmd);
      break;
    case OIT_INTERLOCK:
    case OIT_SPINLOCK:
      clearTransparentLock(cmd, (m_state.algorithm == OIT_INTERLOCK));
      break;
    case OIT_WEIGHTED:
      // Its render pass clears OIT_WEIGHTED for us
      break;
    default:
      assert(!"Algorithm case not called in switch statement!");
  }

  // We'll make the first m_state.percentTransparent percent of our spheres transparent;
  // the rest, at the end, will be opaque. Since we only have one mesh, we can do this
  // by drawing the last range of triangles using an opaque shader, and then drawing
  // the first using our OIT methods.
  const int numObjects     = m_sceneTriangleIndices / m_objectTriangleIndices;
  int       numTransparent = (numObjects * m_state.percentTransparent) / 100;
  if(numTransparent > numObjects)
  {
    numTransparent = numObjects;
  }
  const int numOpaque = numObjects - numTransparent;

  // Start the main render pass
  {
    auto profilerMainRenderPass = m_profilerGPU.cmdFrameSection(cmd, "Main Render Pass");

    // Transition the color image to work as a color attachment, in case it
    // was set to VK_IMAGE_LAYOUT_GENERAL.
    m_colorImage.transitionTo(cmd, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);

    // Set up the render pass
    std::array<VkClearValue, 2> clearValues = {};
    clearValues[0].color                    = {0.2f, 0.2f, 0.2f, 0.2f};  // Background color, in linear space
    clearValues[1].depthStencil             = {1.0f, 0};                 // Clear depth

    const VkRenderPassBeginInfo renderPassInfo{.sType       = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
                                               .renderPass  = m_renderPassColorDepthClear,
                                               .framebuffer = m_mainColorDepthFramebuffer,
                                               .renderArea = {.extent = {m_colorImage.getWidth(), m_colorImage.getHeight()}},
                                               .clearValueCount = uint32_t(clearValues.size()),
                                               .pClearValues    = clearValues.data()};

    vkCmdBeginRenderPass(cmd, &renderPassInfo, VK_SUBPASS_CONTENTS_INLINE);

    // Bind the vertex and index buffers
    const VkDeviceSize offset = 0;
    vkCmdBindVertexBuffers(cmd, 0, 1, &m_vertexBuffer.buffer, &offset);
    vkCmdBindIndexBuffer(cmd, m_indexBuffer.buffer, 0, VK_INDEX_TYPE_UINT32);

    // Bind the descriptor set (constant buffers, images)
    // Pipeline layout depends only on descriptor set layout.
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_pipelineLayout, 0, 1,
                            m_descriptorPack.getSetPtr(m_app->getFrameCycleIndex()), 0, nullptr);

    // Draw all of the opaque objects
    if(numOpaque > 0)
    {
      auto profilerOpaque = m_profilerGPU.cmdFrameSection(cmd, "Opaque");

      // Bind the graphics pipeline state object (shaders, configuration)
      vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_pipelines[+PassIndex::eOpaque]);

      // Draw!
      vkCmdDrawIndexed(cmd, numOpaque * m_objectTriangleIndices, 1, numTransparent * m_objectTriangleIndices, 0, 0);
    }

    // Now, draw the transparent objects.
    switch(m_state.algorithm)
    {
      case OIT_SIMPLE:
        drawTransparentSimple(cmd, numTransparent);
        break;
      case OIT_LINKEDLIST:
        drawTransparentLinkedList(cmd, numTransparent);
        break;
      case OIT_LOOP:
        drawTransparentLoop(cmd, numTransparent);
        break;
      case OIT_LOOP64:
        drawTransparentLoop64(cmd, numTransparent);
        break;
      case OIT_INTERLOCK:
      case OIT_SPINLOCK:
        drawTransparentLock(cmd, numTransparent, (m_state.algorithm == OIT_INTERLOCK));
        break;
      case OIT_WEIGHTED:
        drawTransparentWeighted(cmd, numTransparent);
        break;
      default:
        assert(!"Algorithm case not called in switch statement!");
    }

    vkCmdEndRenderPass(cmd);
  }

  copyOffscreenToBackBuffer(cmd);
}

void Sample::clearTransparentSimple(VkCommandBuffer& cmd)
{
  auto section = m_profilerGPU.cmdFrameSection(cmd, "SimpleClear");

  // Clear the base mip and layer of m_oitAuxImage
  const VkClearColorValue       auxClearColor{.uint32 = {0}};  // Since m_oitAux is R32UINT
  const VkImageSubresourceRange auxClearRanges{
      .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT, .levelCount = 1, .layerCount = m_oitAuxImage.getLayers()};
  vkCmdClearColorImage(cmd,                        // Command buffer
                       m_oitAuxImage.image.image,  // The VkImage
                       VK_IMAGE_LAYOUT_GENERAL,    // The current image layout
                       &auxClearColor,             // The color to clear it with
                       1,                          // The number of VkImageSubresourceRanges below
                       &auxClearRanges             // Range of mipmap levels, array layers, and aspects to be cleared
  );

  // Make sure this completes before using m_oitAuxImage again.
  cmdTransferBarrierSimple(cmd);
}

void Sample::drawTransparentSimple(VkCommandBuffer& cmd, int numObjects)
{
  // COLOR
  // Stores the first OIT_LAYERS fragments per pixel or sample in the A-buffer,
  // and tail-blends the rest.
  {
    auto section = m_profilerGPU.cmdFrameSection(cmd, "SimpleColor");
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_pipelines[+PassIndex::eSimpleColor]);
    // Draw all objects
    vkCmdDrawIndexed(cmd, numObjects * m_objectTriangleIndices, 1, 0, 0, 0);
  }

  // Make sure the color pass completes before the composite pass
  cmdFragmentBarrierSimple(cmd);

  // COMPOSITE
  // Sorts the stored fragments per pixel or sample and composites them onto the color image.
  {
    auto section = m_profilerGPU.cmdFrameSection(cmd, "SimpleComposite");
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_pipelines[+PassIndex::eSimpleComposite]);
    // Draw a full-screen triangle:
    vkCmdDraw(cmd, 3, 1, 0, 0);
  }
}

void Sample::clearTransparentLinkedList(VkCommandBuffer& cmd)
{

  auto section = m_profilerGPU.cmdFrameSection(cmd, "LinkedListClear");

  // Sets the atomic counter (really a 1x1 image) to 0, and set imgAux to 0.
  const VkClearColorValue auxClearColor{.uint32 = {0}};  // Since m_oitAux is R32UINT
  VkImageSubresourceRange auxClearRanges{
      .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT, .levelCount = 1, .layerCount = m_oitAuxImage.getLayers()};
  vkCmdClearColorImage(cmd, m_oitAuxImage.image.image, m_oitAuxImage.getLayout(), &auxClearColor, 1, &auxClearRanges);
  auxClearRanges.layerCount = 1;
  vkCmdClearColorImage(cmd, m_oitCounterImage.image.image, m_oitCounterImage.getLayout(), &auxClearColor, 1, &auxClearRanges);

  // Make sure this completes before using these images again.
  cmdTransferBarrierSimple(cmd);
}

void Sample::drawTransparentLinkedList(VkCommandBuffer& cmd, int numObjects)
{
  // COLOR
  // Constructs the linked lists.
  {
    auto section = m_profilerGPU.cmdFrameSection(cmd, "LinkedListColor");
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_pipelines[+PassIndex::eLinkedListColor]);
    // Draw all objects
    vkCmdDrawIndexed(cmd, numObjects * m_objectTriangleIndices, 1, 0, 0, 0);
  }

  // Make sure the color pass completes before the composite pass
  cmdFragmentBarrierSimple(cmd);

  // COMPOSITE
  // Iterates through the linked lists and sorts and tail-blends fragments.
  {
    auto section = m_profilerGPU.cmdFrameSection(cmd, "LinkedListComposite");
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_pipelines[+PassIndex::eLinkedListComposite]);
    // Draw a full-screen triangle
    vkCmdDraw(cmd, 3, 1, 0, 0);
  }
}

void Sample::clearTransparentLoop(VkCommandBuffer& cmd)
{
  auto section = m_profilerGPU.cmdFrameSection(cmd, "LoopClear");

  // Set all depth values in m_oitABuffer to 0xFFFFFFFF.

  // This makes sure to only overwrite the depth portion of the A-buffer, which
  // should improve bandwidth. See the memory layout described in oitScene.frag.glsl
  // for more information.

  const size_t clearSize = m_sceneUbo.viewport.z * sizeof(uint32_t) * m_state.oitLayers;

  for(size_t i = 0; i < (m_state.sampleShading ? m_state.msaa : 1); i++)
  {
    vkCmdFillBuffer(cmd,                         // Command buffer
                    m_oitABuffer.buffer.buffer,  // Buffer
                    i * clearSize * 2,           // Offset
                    clearSize,                   // Size
                    0xFFFFFFFFu);                // Data
  }

  // Make sure this completes before using m_oitABuffer again.
  cmdTransferBarrierSimple(cmd);
}

void Sample::drawTransparentLoop(VkCommandBuffer& cmd, int numObjects)
{
  // DEPTH
  // Sorts the frontmost OIT_LAYERS depths per sample.
  {
    auto section = m_profilerGPU.cmdFrameSection(cmd, "LoopDepth");
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_pipelines[+PassIndex::eLoopDepth]);
    // Draw all objects
    vkCmdDrawIndexed(cmd, numObjects * m_objectTriangleIndices, 1, 0, 0, 0);
  }

  // Make sure the depth pass completes before the composite pass
  cmdFragmentBarrierSimple(cmd);

  // COLOR
  // Uses the sorted depth information to sort colors into layers
  {
    auto section = m_profilerGPU.cmdFrameSection(cmd, "LoopColor");
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_pipelines[+PassIndex::eLoopColor]);
    // Draw all objects
    vkCmdDrawIndexed(cmd, numObjects * m_objectTriangleIndices, 1, 0, 0, 0);
  }

  // Make sure the color pass completes before the composite pass
  cmdFragmentBarrierSimple(cmd);

  // COMPOSITE
  // Blends the sorted colors together.
  {
    auto section = m_profilerGPU.cmdFrameSection(cmd, "LoopComposite");
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_pipelines[+PassIndex::eLoopComposite]);
    // Draw a full-screen triangle
    vkCmdDraw(cmd, 3, 1, 0, 0);
  }
}

void Sample::clearTransparentLoop64(VkCommandBuffer& cmd)
{
  auto section = m_profilerGPU.cmdFrameSection(cmd, "Loop64Clear");
  // Sets all values in m_oitABuffer to 0xFFFFFFFF (depth), 0xFFFFFFFF (color)
  vkCmdFillBuffer(cmd, m_oitABuffer.buffer.buffer, 0, VK_WHOLE_SIZE, 0xFFFFFFFFu);

  // Make sure this completes before using m_oitABuffer again.
  cmdTransferBarrierSimple(cmd);
}

void Sample::drawTransparentLoop64(VkCommandBuffer& cmd, int numObjects)
{
  // (DEPTH +) COLOR
  // Sorts the frontmost OIT_LAYERS (depth, color) pairs per sample.
  {
    auto section = m_profilerGPU.cmdFrameSection(cmd, "Loop64Color");
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_pipelines[+PassIndex::eLoop64Color]);
    // Draw all objects
    vkCmdDrawIndexed(cmd, numObjects * m_objectTriangleIndices, 1, 0, 0, 0);
  }

  // Make sure the depth + color pass completes before the composite pass
  cmdFragmentBarrierSimple(cmd);

  // COMPOSITE
  // Blends the sorted colors together
  {
    auto section = m_profilerGPU.cmdFrameSection(cmd, "Loop64Composite");
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_pipelines[+PassIndex::eLoop64Composite]);
    // Draw a full-screen triangle
    vkCmdDraw(cmd, 3, 1, 0, 0);
  }
}

void Sample::clearTransparentLock(VkCommandBuffer& cmd, bool useInterlock)
{
  auto section = m_profilerGPU.cmdFrameSection(cmd, "LockClear");
  // Sets the values in IMG_AUX to 0 and IMG_AUXDEPTH to 0xFFFFFFFF.
  // If using spinlock, sets the values in IMG_AUXSPIN to 0 as well.

  const VkClearColorValue       auxClearColor0{.uint32 = {0}};  // // Since m_oitAux is R32UINT
  const VkClearColorValue       auxClearColorF{.uint32 = {0xFFFFFFFFu}};
  const VkImageSubresourceRange auxClearRanges{
      .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT, .levelCount = 1, .layerCount = m_oitAuxDepthImage.getLayers()};

  vkCmdClearColorImage(cmd, m_oitAuxDepthImage.image.image, m_oitAuxDepthImage.getLayout(), &auxClearColorF, 1, &auxClearRanges);
  vkCmdClearColorImage(cmd, m_oitAuxImage.image.image, m_oitAuxImage.getLayout(), &auxClearColor0, 1, &auxClearRanges);
  if(!useInterlock)
  {
    // Also clear m_oitAuxSpinImage
    vkCmdClearColorImage(cmd, m_oitAuxSpinImage.image.image, m_oitAuxSpinImage.getLayout(), &auxClearColor0, 1, &auxClearRanges);
  }
  cmdTransferBarrierSimple(cmd);
}

void Sample::drawTransparentLock(VkCommandBuffer& cmd, int numObjects, bool useInterlock)
{
  // COLOR
  // Sorts the frontmost OIT_LAYERS (depth, color) pairs per pixel.
  {
    auto section = m_profilerGPU.cmdFrameSection(cmd, "LockColor");
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                      m_pipelines[+(useInterlock ? PassIndex::eInterlockColor : PassIndex::eSpinlockColor)]);
    // Draw all objects
    vkCmdDrawIndexed(cmd, numObjects * m_objectTriangleIndices, 1, 0, 0, 0);
  }

  // Make sure the color pass completes before the composite pass
  cmdFragmentBarrierSimple(cmd);

  // COMPOSITE
  // Blends the sorted colors together
  {
    auto section = m_profilerGPU.cmdFrameSection(cmd, "LockComposite");
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                      m_pipelines[+(useInterlock ? PassIndex::eInterlockComposite : PassIndex::eSpinlockComposite)]);
    // Draw a full-screen triangle
    vkCmdDraw(cmd, 3, 1, 0, 0);
  }
}

void Sample::drawTransparentWeighted(VkCommandBuffer& cmd, int numObjects)
{
  // Swap out the render pass for WBOIT's render pass
  vkCmdEndRenderPass(cmd);

  auto section = m_profilerGPU.cmdFrameSection(cmd, "WeightedBlendedOIT");

  // Transition the color image to work as an attachment
  m_colorImage.transitionTo(cmd, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);

  std::array<VkClearValue, 2> clearValues;
  clearValues[0].color = VkClearColorValue{.float32 = {0.0f, 0.0f, 0.0f, 0.0f}};
  // Initially, all pixels show through all the way (reveal = 100%)
  clearValues[1].color = VkClearColorValue{.float32 = {1.0f}};
  const VkRenderPassBeginInfo renderPassInfo{.sType           = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
                                             .renderPass      = m_renderPassWeighted,
                                             .framebuffer     = m_weightedFramebuffer,
                                             .renderArea      = {.extent = {.width  = m_oitWeightedColorImage.getWidth(),
                                                                            .height = m_oitWeightedColorImage.getHeight()}},
                                             .clearValueCount = uint32_t(clearValues.size()),
                                             .pClearValues    = clearValues.data()};

  vkCmdBeginRenderPass(cmd, &renderPassInfo, VK_SUBPASS_CONTENTS_INLINE);

  // COLOR PASS
  // Computes the weighted sum and reveal factor.
  {
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_pipelines[+PassIndex::eWeightedColor]);
    // Draw all objects
    vkCmdDrawIndexed(cmd, numObjects * m_objectTriangleIndices, 1, 0, 0, 0);
  }

  // Move to the next subpass
  vkCmdNextSubpass(cmd, VK_SUBPASS_CONTENTS_INLINE);
  // COMPOSITE PASS
  // Averages out the summed colors (in some sense) to get the final transparent color.
  {
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_pipelines[+PassIndex::eWeightedComposite]);
    // Draw a full-screen triangle
    vkCmdDraw(cmd, 3, 1, 0, 0);
  }
}
