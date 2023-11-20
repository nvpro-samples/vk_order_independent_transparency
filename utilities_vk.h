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

// Contains utility classes for this sample.
// Many of these are specific to this sample and wouldn't fit in the more
// general NVVK helper library - for instance, Vertex specifies the vertex
// binding description and attribute description for the geometry that this
// sample specifically uses.

#include <array>
#include <nvh/geometry.hpp>
#include <glm/glm.hpp>
#include <nvvk/resourceallocator_vk.hpp>
#include <nvvk/buffers_vk.hpp>
#include <nvvk/context_vk.hpp>
#include <nvvk/debug_util_vk.hpp>
#include <nvvk/images_vk.hpp>
#include <nvvk/structs_vk.hpp>
#include <vulkan/vulkan_core.h>

// Vertex structure used for the main mesh.
struct Vertex
{
  glm::vec3 pos;
  glm::vec3 normal;
  glm::vec4 color;

  // Must have a constructor from nvh::geometry::Vertex in order for initScene
  // to work
  Vertex(const nvh::geometry::Vertex& vertex)
  {
    for(int i = 0; i < 3; i++)
    {
      pos[i]    = vertex.position[i];
      normal[i] = vertex.normal[i];
    }
    color = glm::vec4(1.0f);
  }

  static VkVertexInputBindingDescription getBindingDescription()
  {
    VkVertexInputBindingDescription bindingDescription = {};
    bindingDescription.binding                         = 0;
    bindingDescription.stride                          = sizeof(Vertex);
    bindingDescription.inputRate                       = VK_VERTEX_INPUT_RATE_VERTEX;

    return bindingDescription;
  }

  static std::array<VkVertexInputAttributeDescription, 3> getAttributeDescriptions()
  {
    std::array<VkVertexInputAttributeDescription, 3> attributeDescriptions = {};

    attributeDescriptions[0].binding  = 0;
    attributeDescriptions[0].location = 0;
    attributeDescriptions[0].format   = VK_FORMAT_R32G32B32_SFLOAT;
    attributeDescriptions[0].offset   = offsetof(Vertex, pos);

    attributeDescriptions[1].binding  = 0;
    attributeDescriptions[1].location = 1;
    attributeDescriptions[1].format   = VK_FORMAT_R32G32B32_SFLOAT;
    attributeDescriptions[1].offset   = offsetof(Vertex, normal);

    attributeDescriptions[2].binding  = 0;
    attributeDescriptions[2].location = 2;
    attributeDescriptions[2].format   = VK_FORMAT_R32G32B32A32_SFLOAT;
    attributeDescriptions[2].offset   = offsetof(Vertex, color);

    return attributeDescriptions;
  }
};

// A BufferAndView is an NVVK buffer (i.e. Vulkan buffer and underlying memory),
// together with a view that points to the whole buffer. It's a simplification
// that works for this sample!
struct BufferAndView
{
  nvvk::Buffer buffer;
  VkBufferView    view = nullptr;
  VkDeviceSize    size = 0;  // In bytes

  // Creates a buffer and view with the given size, usage, and view format.
  // The memory properties are always VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT.
  void create(nvvk::Context& context, nvvk::ResourceAllocatorDma& allocator, VkDeviceSize bufferSize, VkBufferUsageFlags bufferUsage, VkFormat viewFormat)
  {
    assert(buffer.buffer == nullptr);  // Destroy the buffer before recreating it, please!
    buffer = allocator.createBuffer(bufferSize, bufferUsage);
    if((bufferUsage & (VK_BUFFER_USAGE_STORAGE_TEXEL_BUFFER_BIT | VK_BUFFER_USAGE_UNIFORM_TEXEL_BUFFER_BIT)) != 0)
    {
      view = nvvk::createBufferView(context, nvvk::makeBufferViewCreateInfo(buffer.buffer, viewFormat, bufferSize));
    }
    size = bufferSize;
  }

  // To destroy the object, provide its context and allocator.
  void destroy(nvvk::Context& context, nvvk::ResourceAllocatorDma& allocator)
  {
    if(buffer.buffer != nullptr)
    {
      allocator.destroy(buffer);
    }

    if(view != nullptr)
    {
      vkDestroyBufferView(context.m_device, view, nullptr);
      view = nullptr;
    }

    size = 0;
  }

  void setName(nvvk::DebugUtil& util, const char* name)
  {
    util.setObjectName(buffer.buffer, name);
    if(view != nullptr)
    {
      util.setObjectName(view, name);
    }
  }
};

// Creates a simple texture with 1 mip, 1 array layer, 1 sample per texel, with
// optimal tiling, in an undefined layout, with the VK_IMAGE_USAGE_SAMPLED_BIT flag
// (and possibly additional flags), and accessible only from a single queue family.
inline nvvk::Image createImageSimple(nvvk::ResourceAllocatorDma& allocator,
                                        VkImageType         imageType,
                                        VkFormat            format,
                                        uint32_t            width,
                                        uint32_t            height,
                                        uint32_t            arrayLayers          = 1,
                                        VkImageUsageFlags   additionalUsageFlags = 0,
                                        uint32_t            numSamples           = 1)
{
  // There are several different ways to create images using the NVVK framework.
  // Here, we'll use AllocatorDma::createImage.

  VkImageCreateInfo imageInfo = nvvk::make<VkImageCreateInfo>();
  imageInfo.imageType         = imageType;
  imageInfo.extent.width      = width;
  imageInfo.extent.height     = height;
  imageInfo.extent.depth      = 1;
  imageInfo.mipLevels         = 1;
  imageInfo.arrayLayers       = arrayLayers;
  imageInfo.format            = format;
  imageInfo.tiling            = VK_IMAGE_TILING_OPTIMAL;
  imageInfo.initialLayout     = VK_IMAGE_LAYOUT_UNDEFINED;
  imageInfo.usage             = VK_IMAGE_USAGE_SAMPLED_BIT | additionalUsageFlags;
  imageInfo.samples           = static_cast<VkSampleCountFlagBits>(numSamples);
  imageInfo.sharingMode       = VK_SHARING_MODE_EXCLUSIVE;

  return allocator.createImage(imageInfo);
}

inline VkSampleCountFlagBits getSampleCountFlagBits(int msaa)
{
  return static_cast<VkSampleCountFlagBits>(msaa);
}

// A simple wrapper for writing a vkCmdPipelineBarrier for doing things such as
// image layout transitions.
inline void cmdImageTransition(VkCommandBuffer    cmd,
                               VkImage            img,
                               VkImageAspectFlags aspects,
                               VkAccessFlags      src,
                               VkAccessFlags      dst,
                               VkImageLayout      oldLayout,
                               VkImageLayout      newLayout)
{

  VkPipelineStageFlags srcPipe = nvvk::makeAccessMaskPipelineStageFlags(src);
  VkPipelineStageFlags dstPipe = nvvk::makeAccessMaskPipelineStageFlags(dst);
  VkImageMemoryBarrier barrier = nvvk::makeImageMemoryBarrier(img, src, dst, oldLayout, newLayout, aspects);

  vkCmdPipelineBarrier(cmd, srcPipe, dstPipe, VK_FALSE, 0, VK_NULL_HANDLE, 0, VK_NULL_HANDLE, 1, &barrier);
}

// An ImageAndView is an NVVK image (i.e. Vulkan image and underlying memory),
// together with a view that points to the whole image, and data to track its
// current state. It's a simplification that works for this sample!
struct ImageAndView
{
  nvvk::Image image;
  VkImageView    view = nullptr;
  // Information you might need, but please don't modify
  uint32_t c_width  = 0;                    // Should not be changed once the texture is created!
  uint32_t c_height = 0;                    // Should not be changed once the texture is created!
  uint32_t c_layers = 0;                    // Should not be changed once the texture is created!
  VkFormat c_format = VK_FORMAT_UNDEFINED;  // Should not be changed once the texture is created!

  // Information for pipeline transitions. These should generally only be
  // modified via transitionTo or when ending render passes.
  VkImageLayout currentLayout = VK_IMAGE_LAYOUT_UNDEFINED;  // The current layout of the image in GPU memory (e.g. GENERAL or COLOR_ATTACHMENT_OPTIMAL)
  VkAccessFlags currentAccesses = 0;  // How the memory of this texture can be accessed (e.g. shader read/write, color attachment read/write)

public:
  // Creates a simple texture and view with 1 mip and 1 array layer.
  // with optimal tiling, in an undefined layout, with the VK_IMAGE_USAGE_SAMPLED_BIT flag
  // (and possibly additional flags), and accessible only from a single queue family.
  void create(nvvk::Context&      context,
              nvvk::ResourceAllocatorDma& allocator,
              VkImageType         imageType,
              VkImageAspectFlags  viewAspect,
              VkFormat            format,
              uint32_t            width,
              uint32_t            height,
              uint32_t            arrayLayers          = 1,
              VkImageUsageFlags   additionalUsageFlags = 0,
              uint32_t            numSamples           = 1)
  {
    assert(view == nullptr);  // Destroy the image before recreating it, please!
    image = createImageSimple(allocator, imageType, format, width, height, arrayLayers, additionalUsageFlags, numSamples);
    VkImageViewCreateInfo viewInfo       = nvvk::makeImage2DViewCreateInfo(image.image, format, viewAspect);
    viewInfo.subresourceRange.layerCount = arrayLayers;
    viewInfo.viewType                    = (arrayLayers == 1 ? VK_IMAGE_VIEW_TYPE_2D : VK_IMAGE_VIEW_TYPE_2D_ARRAY);
    vkCreateImageView(context.m_device, &viewInfo, nullptr, &view);

    c_width  = width;
    c_height = height;
    c_layers = arrayLayers;
    c_format = format;
  }

  // To destroy the object, provide its context and allocator.
  void destroy(nvvk::Context& context, nvvk::ResourceAllocatorDma& allocator)
  {
    if(view != nullptr)
    {
      vkDestroyImageView(context.m_device, view, nullptr);
      allocator.destroy(image);
      view            = nullptr;
      currentLayout   = VK_IMAGE_LAYOUT_UNDEFINED;
      currentAccesses = 0;
    }
  }

  void transitionTo(VkCommandBuffer cmdBuffer,
                    VkImageLayout   dstLayout,  // How the image will be laid out in memory.
                    VkAccessFlags   dstAccesses)  // The ways that the app will be able to access the image.
  {
    // Note that in larger applications, we could batch together pipeline
    // barriers for better performance!

    // Maps to barrier.subresourceRange.aspectMask
    VkImageAspectFlags aspectMask = 0;
    if(dstLayout == VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL)
    {
      aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;
      if(c_format == VK_FORMAT_D32_SFLOAT_S8_UINT || c_format == VK_FORMAT_D24_UNORM_S8_UINT)
      {
        aspectMask |= VK_IMAGE_ASPECT_STENCIL_BIT;
      }
    }
    else
    {
      aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    }

    cmdImageTransition(cmdBuffer, image.image, aspectMask, currentAccesses, dstAccesses, currentLayout, dstLayout);

    // Update current layout, stages, and accesses
    currentLayout   = dstLayout;
    currentAccesses = dstAccesses;
  }

  // Should be called to keep track of the image's current layout when a render
  // pass that includes a image layout transition finishes.
  void endRenderPass(VkImageLayout dstLayout) { currentLayout = dstLayout; }

  void setName(nvvk::DebugUtil& util, const char* name)
  {
    util.setObjectName(image.image, name);
    util.setObjectName(view, name);
  }
};

// Adds a simple command that ensures that all transfer writes have finished before all
// subsequent fragment shader reads and writes (in the current scope).
// Note that on NV hardware, unless you need a layout transition, there's little benefit to using
// memory barriers for each of the individual objects (and in fact may run into issues with the
// Vulkan specification).
// The dependency flags are BY_REGION_BIT by default, since most calls to cmdBarrier come from
// dependencies inside render passes, which require this (according to section 6.6.1).
inline void cmdTransferBarrierSimple(VkCommandBuffer cmdBuffer)
{
  const VkPipelineStageFlags srcStageFlags = VK_PIPELINE_STAGE_TRANSFER_BIT;
  const VkPipelineStageFlags dstStageFlags = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;

  VkMemoryBarrier barrier = nvvk::make<VkMemoryBarrier>();
  barrier.srcAccessMask   = VK_ACCESS_TRANSFER_WRITE_BIT;
  barrier.dstAccessMask   = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;

  vkCmdPipelineBarrier(cmdBuffer, srcStageFlags, dstStageFlags, VK_DEPENDENCY_BY_REGION_BIT,  //
                       1, &barrier,                                                           //
                       0, VK_NULL_HANDLE,                                                     //
                       0, VK_NULL_HANDLE);
}

// Adds a simple command that ensures that all fragment shader reads writes have finished before all
// subsequent fragment shader reads and writes (in the current scope).
// Note that on NV hardware, unless you need a layout transition, there's little benefit to using
// memory barriers for each of the individual objects (and in fact may run into issues with the
// Vulkan specification).
// The dependency flags are BY_REGION_BIT by default, since most calls to cmdBarrier come from
// dependencies inside render passes, which require this (according to section 6.6.1).
inline void cmdFragmentBarrierSimple(VkCommandBuffer cmdBuffer)
{
  const VkPipelineStageFlags stageFlags = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;

  VkMemoryBarrier barrier = nvvk::make<VkMemoryBarrier>();
  barrier.srcAccessMask   = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
  barrier.dstAccessMask   = barrier.srcAccessMask;

  vkCmdPipelineBarrier(cmdBuffer, stageFlags, stageFlags, VK_DEPENDENCY_BY_REGION_BIT,  //
                       1, &barrier,                                  //
                       0, VK_NULL_HANDLE,                            //
                       0, VK_NULL_HANDLE);
}