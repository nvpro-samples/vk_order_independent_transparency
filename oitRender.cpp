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


// This file contains implementations of the main OIT drawing functions from
// oit.h, excluding GUI and resolving from m_colorImage to the swapchain.

#include "oit.h"

void Sample::render(VkCommandBuffer& cmdBuffer)
{
  // Clear auxiliary buffers before we even start a render pass - this
  // reduces the number of render passes we need to use by 1.
  switch(m_state.algorithm)
  {
    case OIT_SIMPLE:
      clearTransparentSimple(cmdBuffer);
      break;
    case OIT_LINKEDLIST:
      clearTransparentLinkedList(cmdBuffer);
      break;
    case OIT_LOOP:
      clearTransparentLoop(cmdBuffer);
      break;
    case OIT_LOOP64:
      clearTransparentLoop64(cmdBuffer);
      break;
    case OIT_INTERLOCK:
    case OIT_SPINLOCK:
      clearTransparentLock(cmdBuffer, (m_state.algorithm == OIT_INTERLOCK));
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
    const nvvk::ProfilerVK::Section scopedTimer(m_profilerVK, "Main", cmdBuffer);

    // Transition the color image to work as a color attachment, in case it
    // was set to VK_IMAGE_LAYOUT_GENERAL.
    m_colorImage.transitionTo(cmdBuffer,                                 // Command buffer
                              VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,  // New layout
                              VK_ACCESS_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT);

    // Set up the render pass
    VkRenderPassBeginInfo renderPassInfo    = {VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO};
    renderPassInfo.renderPass               = m_renderPassColorDepthClear;
    renderPassInfo.framebuffer              = m_mainColorDepthFramebuffer;
    renderPassInfo.renderArea.offset        = {0, 0};
    renderPassInfo.renderArea.extent.width  = m_colorImage.c_width;
    renderPassInfo.renderArea.extent.height = m_colorImage.c_height;

    std::array<VkClearValue, 2> clearValues = {};
    clearValues[0].color                    = {0.2f, 0.2f, 0.2f, 0.2f};  // Background color, in linear space
    clearValues[1].depthStencil             = {1.0f, 0};                 // Clear depth
    renderPassInfo.clearValueCount          = static_cast<uint32_t>(clearValues.size());
    renderPassInfo.pClearValues             = clearValues.data();

    vkCmdBeginRenderPass(cmdBuffer, &renderPassInfo, VK_SUBPASS_CONTENTS_INLINE);

    // Draw all of the opaque objects
    {
      // Bind the descriptor set (constant buffers, images)
      // Pipeline layout depends only on descriptor set layout.
      VkDescriptorSet descriptorSet = m_descriptorInfo.getSet(m_swapChain.getActiveImageIndex());
      vkCmdBindDescriptorSets(cmdBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, m_descriptorInfo.getPipeLayout(), 0, 1,
                              &descriptorSet, 0, nullptr);

      drawSceneObjects(cmdBuffer, numTransparent, numOpaque);
    }

    // Now, draw the transparent objects.
    switch(m_state.algorithm)
    {
      case OIT_SIMPLE:
        drawTransparentSimple(cmdBuffer, numTransparent);
        break;
      case OIT_LINKEDLIST:
        drawTransparentLinkedList(cmdBuffer, numTransparent);
        break;
      case OIT_LOOP:
        drawTransparentLoop(cmdBuffer, numTransparent);
        break;
      case OIT_LOOP64:
        drawTransparentLoop64(cmdBuffer, numTransparent);
        break;
      case OIT_INTERLOCK:
      case OIT_SPINLOCK:
        drawTransparentLock(cmdBuffer, numTransparent, (m_state.algorithm == OIT_INTERLOCK));
        break;
      case OIT_WEIGHTED:
        drawTransparentWeighted(cmdBuffer, numTransparent);
        break;
      default:
        assert(!"Algorithm case not called in switch statement!");
    }

    vkCmdEndRenderPass(cmdBuffer);
  }
}

void Sample::drawSceneObjects(VkCommandBuffer& cmdBuffer, int firstObject, int numObjects)
{
  // Bind the vertex and index buffers
  VkBuffer     vertexBuffers[] = {m_vertexBuffer.buffer};
  VkDeviceSize offsets[]       = {0};
  vkCmdBindVertexBuffers(cmdBuffer, 0, 1, vertexBuffers, offsets);

  vkCmdBindIndexBuffer(cmdBuffer, m_indexBuffer.buffer, 0, VK_INDEX_TYPE_UINT32);

  // Bind the graphics pipeline state object (shaders, configuration)
  vkCmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, m_pipelineOpaque);

  // Draw!
  vkCmdDrawIndexed(cmdBuffer, numObjects * m_objectTriangleIndices, 1, firstObject * m_objectTriangleIndices, 0, 0);
}

void Sample::clearTransparentSimple(VkCommandBuffer& cmdBuffer)
{
  // Clears all values in m_oitAux to 0.
  const nvvk::ProfilerVK::Section scopedTimer(m_profilerVK, "ClearSimple", cmdBuffer);

  // Clear the base mip and layer of m_oitAuxImage
  VkClearColorValue auxClearColor;
  auxClearColor.uint32[0] = 0;  // Since m_oitAux is R32UINT
  VkImageSubresourceRange auxClearRanges;
  auxClearRanges.aspectMask     = VK_IMAGE_ASPECT_COLOR_BIT;
  auxClearRanges.baseArrayLayer = 0;
  auxClearRanges.baseMipLevel   = 0;
  auxClearRanges.layerCount     = m_oitAuxImage.c_layers;
  auxClearRanges.levelCount     = 1;
  vkCmdClearColorImage(cmdBuffer,                  // Command buffer
                       m_oitAuxImage.image.image,  // The VkImage
                       VK_IMAGE_LAYOUT_GENERAL,    // The current image layout
                       &auxClearColor,             // The color to clear it with
                       1,                          // The number of VkImageSubresourceRanges below
                       &auxClearRanges             // Range of mipmap levels, array layers, and aspects to be cleared
  );

  // Make sure this completes before using m_oitAuxImage again.
  cmdTransferBarrierSimple(cmdBuffer);
}

void Sample::drawTransparentSimple(VkCommandBuffer& cmdBuffer, int numObjects)
{
  // COLOR
  // Stores the first OIT_LAYERS fragments per pixel or sample in the A-buffer,
  // and tail-blends the rest.
  {
    vkCmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, m_pipelineSimpleColor);
    // Draw all objects
    vkCmdDrawIndexed(cmdBuffer, numObjects * m_objectTriangleIndices, 1, 0, 0, 0);
  }

  // Make sure the color pass completes before the composite pass
  cmdFragmentBarrierSimple(cmdBuffer);

  // COMPOSITE
  // Sorts the stored fragments per pixel or sample and composites them onto the color image.
  {
    vkCmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, m_pipelineSimpleComposite);
    // Draw a full-screen triangle:
    vkCmdDraw(cmdBuffer, 3, 1, 0, 0);
  }
}

void Sample::clearTransparentLinkedList(VkCommandBuffer& cmdBuffer)
{
  // Sets the atomic counter (really a 1x1 image) to 0, and set imgAux to 0.
  const nvvk::ProfilerVK::Section scopedTimer(m_profilerVK, "ClearLinkedList", cmdBuffer);

  VkClearColorValue auxClearColor;
  auxClearColor.uint32[0] = 0;  // Since m_oitAux is R32UINT
  VkImageSubresourceRange auxClearRanges;
  auxClearRanges.aspectMask     = VK_IMAGE_ASPECT_COLOR_BIT;
  auxClearRanges.baseArrayLayer = 0;
  auxClearRanges.baseMipLevel   = 0;
  auxClearRanges.layerCount     = m_oitAuxImage.c_layers;
  auxClearRanges.levelCount     = 1;
  vkCmdClearColorImage(cmdBuffer, m_oitAuxImage.image.image, m_oitAuxImage.currentLayout, &auxClearColor, 1, &auxClearRanges);
  auxClearRanges.layerCount = 1;
  vkCmdClearColorImage(cmdBuffer, m_oitCounterImage.image.image, m_oitCounterImage.currentLayout, &auxClearColor, 1, &auxClearRanges);

  // Make sure this completes before using these images again.
  cmdTransferBarrierSimple(cmdBuffer);
}

void Sample::drawTransparentLinkedList(VkCommandBuffer& cmdBuffer, int numObjects)
{
  // COLOR
  // Constructs the linked lists.
  {
    vkCmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, m_pipelineLinkedListColor);
    // Draw all objects
    vkCmdDrawIndexed(cmdBuffer, numObjects * m_objectTriangleIndices, 1, 0, 0, 0);
  }

  // Make sure the color pass completes before the composite pass
  cmdFragmentBarrierSimple(cmdBuffer);

  // COMPOSITE
  // Iterates through the linked lists and sorts and tail-blends fragments.
  {
    vkCmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, m_pipelineLinkedListComposite);
    // Draw a full-screen triangle
    vkCmdDraw(cmdBuffer, 3, 1, 0, 0);
  }
}

void Sample::clearTransparentLoop(VkCommandBuffer& cmdBuffer)
{
  // Set all depth values in m_oitABuffer to 0xFFFFFFFF.
  const nvvk::ProfilerVK::Section scopedTimer(m_profilerVK, "ClearLoop", cmdBuffer);

  // This makes sure to only overwrite the depth portion of the A-buffer, which
  // should improve bandwidth. See the memory layout described in oitScene.frag.glsl
  // for more information.

  const size_t clearSize = m_sceneUbo.viewport.z * sizeof(uint32_t) * m_state.oitLayers;

  for(size_t i = 0; i < (m_state.sampleShading ? m_state.msaa : 1); i++)
  {
    vkCmdFillBuffer(cmdBuffer,                   // Command buffer
                    m_oitABuffer.buffer.buffer,  // Buffer
                    i * clearSize * 2,           // Offset
                    clearSize,                   // Size
                    0xFFFFFFFFu);                // Data
  }

  // Make sure this completes before using m_oitABuffer again.
  cmdTransferBarrierSimple(cmdBuffer);
}

void Sample::drawTransparentLoop(VkCommandBuffer& cmdBuffer, int numObjects)
{
  // DEPTH
  // Sorts the frontmost OIT_LAYERS depths per sample.
  {
    vkCmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, m_pipelineLoopDepth);
    // Draw all objects
    vkCmdDrawIndexed(cmdBuffer, numObjects * m_objectTriangleIndices, 1, 0, 0, 0);
  }

  // Make sure the depth pass completes before the composite pass
  cmdFragmentBarrierSimple(cmdBuffer);

  // COLOR
  // Uses the sorted depth information to sort colors into layers
  {
    vkCmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, m_pipelineLoopColor);
    // Draw all objects
    vkCmdDrawIndexed(cmdBuffer, numObjects * m_objectTriangleIndices, 1, 0, 0, 0);
  }

  // Make sure the color pass completes before the composite pass
  cmdFragmentBarrierSimple(cmdBuffer);

  // COMPOSITE
  // Blends the sorted colors together.
  {
    vkCmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, m_pipelineLoopComposite);
    // Draw a full-screen triangle
    vkCmdDraw(cmdBuffer, 3, 1, 0, 0);
  }
}

void Sample::clearTransparentLoop64(VkCommandBuffer& cmdBuffer)
{
  // Sets all values in m_oitABuffer to 0xFFFFFFFF (depth), 0xFFFFFFFF (color)
  const nvvk::ProfilerVK::Section scopedTimer(m_profilerVK, "ClearLoop64", cmdBuffer);
  vkCmdFillBuffer(cmdBuffer, m_oitABuffer.buffer.buffer, 0, VK_WHOLE_SIZE, 0xFFFFFFFFu);

  // Make sure this completes before using m_oitABuffer again.
  cmdTransferBarrierSimple(cmdBuffer);
}

void Sample::drawTransparentLoop64(VkCommandBuffer& cmdBuffer, int numObjects)
{
  // (DEPTH +) COLOR
  // Sorts the frontmost OIT_LAYERS (depth, color) pairs per sample.
  {
    vkCmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, m_pipelineLoop64Color);
    // Draw all objects
    vkCmdDrawIndexed(cmdBuffer, numObjects * m_objectTriangleIndices, 1, 0, 0, 0);
  }

  // Make sure the depth + color pass completes before the composite pass
  cmdFragmentBarrierSimple(cmdBuffer);

  // COMPOSITE
  // Blends the sorted colors together
  {
    vkCmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, m_pipelineLoop64Composite);
    // Draw a full-screen triangle
    vkCmdDraw(cmdBuffer, 3, 1, 0, 0);
  }
}

void Sample::clearTransparentLock(VkCommandBuffer& cmdBuffer, bool useInterlock)
{
  // Sets the values in IMG_AUX to 0 and IMG_AUXDEPTH to 0xFFFFFFFF.
  // If using spinlock, sets the values in IMG_AUXSPIN to 0 as well.
  const nvvk::ProfilerVK::Section scopedTimer(m_profilerVK, "ClearLock", cmdBuffer);

  VkClearColorValue auxClearColor0;
  auxClearColor0.uint32[0] = 0;  // Since m_oitAux is R32UINT
  VkClearColorValue auxClearColorF;
  auxClearColorF.uint32[0] = 0xFFFFFFFFu;
  VkImageSubresourceRange auxClearRanges;
  auxClearRanges.aspectMask     = VK_IMAGE_ASPECT_COLOR_BIT;
  auxClearRanges.baseArrayLayer = 0;
  auxClearRanges.baseMipLevel   = 0;
  auxClearRanges.layerCount     = m_oitAuxDepthImage.c_layers;
  auxClearRanges.levelCount     = 1;

  vkCmdClearColorImage(cmdBuffer, m_oitAuxDepthImage.image.image, m_oitAuxDepthImage.currentLayout, &auxClearColorF, 1, &auxClearRanges);
  vkCmdClearColorImage(cmdBuffer, m_oitAuxImage.image.image, m_oitAuxImage.currentLayout, &auxClearColor0, 1, &auxClearRanges);
  if(!useInterlock)
  {
    // Also clear m_oitAuxSpinImage
    vkCmdClearColorImage(cmdBuffer, m_oitAuxSpinImage.image.image, m_oitAuxSpinImage.currentLayout, &auxClearColor0, 1, &auxClearRanges);
  }
  cmdTransferBarrierSimple(cmdBuffer);
}

void Sample::drawTransparentLock(VkCommandBuffer& cmdBuffer, int numObjects, bool useInterlock)
{
  // COLOR
  // Sorts the frontmost OIT_LAYERS (depth, color) pairs per pixel.
  {
    vkCmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, (useInterlock ? m_pipelineInterlockColor : m_pipelineSpinlockColor));
    // Draw all objects
    vkCmdDrawIndexed(cmdBuffer, numObjects * m_objectTriangleIndices, 1, 0, 0, 0);
  }

  // Make sure the color pass completes before the composite pass
  cmdFragmentBarrierSimple(cmdBuffer);

  // COMPOSITE
  // Blends the sorted colors together
  {
    vkCmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS,
                      (useInterlock ? m_pipelineInterlockComposite : m_pipelineSpinlockComposite));
    // Draw a full-screen triangle
    vkCmdDraw(cmdBuffer, 3, 1, 0, 0);
  }
}

void Sample::drawTransparentWeighted(VkCommandBuffer& cmdBuffer, int numObjects)
{
  // Swap out the render pass for WBOIT's render pass
  vkCmdEndRenderPass(cmdBuffer);

  // Transition the color image to work as an attachment
  m_colorImage.transitionTo(cmdBuffer, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                            VK_ACCESS_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT);

  VkRenderPassBeginInfo renderPassInfo    = {VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO};
  renderPassInfo.renderPass               = m_renderPassWeighted;
  renderPassInfo.framebuffer              = m_weightedFramebuffer;
  renderPassInfo.renderArea.offset        = {0, 0};
  renderPassInfo.renderArea.extent.width  = m_oitWeightedColorImage.c_width;
  renderPassInfo.renderArea.extent.height = m_oitWeightedRevealImage.c_height;
  std::array<VkClearValue, 2> clearValues;
  clearValues[0].color.float32[0] = 0.0f;
  clearValues[0].color.float32[1] = 0.0f;
  clearValues[0].color.float32[2] = 0.0f;
  clearValues[0].color.float32[3] = 0.0f;
  clearValues[1].color.float32[0] = 1.0f;  // Initially, all pixels show through all the way (reveal = 100%)
  renderPassInfo.clearValueCount  = static_cast<uint32_t>(clearValues.size());
  renderPassInfo.pClearValues     = clearValues.data();

  vkCmdBeginRenderPass(cmdBuffer, &renderPassInfo, VK_SUBPASS_CONTENTS_INLINE);

  // COLOR PASS
  // Computes the weighted sum and reveal factor.
  {
    vkCmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, m_pipelineWeightedColor);
    // Draw all objects
    vkCmdDrawIndexed(cmdBuffer, numObjects * m_objectTriangleIndices, 1, 0, 0, 0);
  }

  // Move to the next subpass
  vkCmdNextSubpass(cmdBuffer, VK_SUBPASS_CONTENTS_INLINE);
  // COMPOSITE PASS
  // Averages out the summed colors (in some sense) to get the final transparent color.
  {
    vkCmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, m_pipelineWeightedComposite);
    // Draw a full-screen triangle
    vkCmdDraw(cmdBuffer, 3, 1, 0, 0);
  }
}