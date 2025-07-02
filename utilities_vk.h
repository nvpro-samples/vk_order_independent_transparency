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

// Contains utility classes for this sample.
// Many of these are specific to this sample and wouldn't fit in the more
// general NVVK helper library - for instance, Vertex specifies the vertex
// binding description and attribute description for the geometry that this
// sample specifically uses.

#include <nvutils/file_operations.hpp>
#include <nvutils/hash_operations.hpp>
#include <nvutils/primitives.hpp>
#include <nvvk/barriers.hpp>
#include <nvvk/check_error.hpp>
#include <nvvk/debug_util.hpp>
#include <nvvk/resource_allocator.hpp>
#include <nvvk/resources.hpp>
#include <nvvkglsl/glsl.hpp>

#include <glm/glm.hpp>
#include <vulkan/vulkan_core.h>

#include <array>
#include <filesystem>
#include <string>
#include <tuple>
#include <vector>

// Vertex structure used for the main mesh.
struct Vertex
{
  glm::vec3 pos;
  glm::vec3 normal;
  glm::vec4 color;

  // Must have a constructor from nvh::geometry::Vertex in order for initScene
  // to work
  Vertex(const nvutils::PrimitiveVertex& vertex)
  {
    pos    = vertex.pos;
    normal = vertex.nrm;
    color  = glm::vec4(1.0f);
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

    attributeDescriptions[0] = VkVertexInputAttributeDescription{
        .location = 0, .binding = 0, .format = VK_FORMAT_R32G32B32_SFLOAT, .offset = offsetof(Vertex, pos)};

    attributeDescriptions[1] = VkVertexInputAttributeDescription{
        .location = 1, .binding = 0, .format = VK_FORMAT_R32G32B32_SFLOAT, .offset = offsetof(Vertex, normal)};

    attributeDescriptions[2] = VkVertexInputAttributeDescription{
        .location = 2, .binding = 0, .format = VK_FORMAT_R32G32B32A32_SFLOAT, .offset = offsetof(Vertex, color)};

    return attributeDescriptions;
  }
};

// A BufferAndView is an NVVK buffer (i.e. Vulkan buffer and underlying memory),
// together with a view that points to the whole buffer. It's a simplification
// that works for this sample!
struct BufferAndView
{
  nvvk::Buffer buffer;
  VkBufferView view = VK_NULL_HANDLE;
  VkDeviceSize size = 0;  // In bytes

  // Creates a buffer and view with the given size, usage, and view format.
  // The memory properties are always VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT.
  void init(VkDevice device, nvvk::ResourceAllocator& allocator, VkDeviceSize bufferSize, VkBufferUsageFlags bufferUsage, VkFormat viewFormat)
  {
    assert(buffer.buffer == VK_NULL_HANDLE);  // Destroy the buffer before recreating it, please!
    NVVK_CHECK(allocator.createBuffer(buffer, bufferSize, bufferUsage));
    if((bufferUsage & (VK_BUFFER_USAGE_STORAGE_TEXEL_BUFFER_BIT | VK_BUFFER_USAGE_UNIFORM_TEXEL_BUFFER_BIT)) != 0)
    {
      VkBufferViewCreateInfo info{.sType  = VK_STRUCTURE_TYPE_BUFFER_VIEW_CREATE_INFO,  //
                                  .buffer = buffer.buffer,                              //
                                  .format = viewFormat,                                 //
                                  .range  = bufferSize};
      NVVK_CHECK(vkCreateBufferView(device, &info, nullptr, &view));
    }
    size = bufferSize;
  }

  // To destroy the object, provide its context and allocator.
  void deinit(VkDevice device, nvvk::ResourceAllocator& allocator)
  {
    if(view != VK_NULL_HANDLE)
    {
      vkDestroyBufferView(device, view, nullptr);
      view = VK_NULL_HANDLE;
    }

    if(buffer.buffer != VK_NULL_HANDLE)
    {
      allocator.destroyBuffer(buffer);
    }

    size = 0;
  }

  void setName(nvvk::DebugUtil& util, const char* name)
  {
    util.setObjectName(buffer.buffer, name);
    if(view != VK_NULL_HANDLE)
    {
      util.setObjectName(view, name);
    }
  }
};

// An ImageAndView is an NVVK image (i.e. Vulkan image and underlying memory),
// together with a view that points to the whole image, and data to track its
// current state. It's a simplification that works for this sample!
struct ImageAndView
{
  nvvk::Image image;

public:
  // Creates a simple texture and view with 1 mip and 1 array layer.
  // with optimal tiling, in an undefined layout, with the VK_IMAGE_USAGE_SAMPLED_BIT flag
  // (and possibly additional flags), and accessible only from a single queue family.
  void init(VkDevice                 device,
            nvvk::ResourceAllocator& allocator,
            VkImageType              imageType,
            VkImageAspectFlags       viewAspect,
            VkFormat                 format,
            uint32_t                 width,
            uint32_t                 height,
            uint32_t                 arrayLayers          = 1,
            VkImageUsageFlags        additionalUsageFlags = 0,
            uint32_t                 numSamples           = 1)
  {
    assert(image.image == VK_NULL_HANDLE);  // Destroy the image before recreating it, please!


    VkImageCreateInfo imageInfo = {.sType         = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
                                   .imageType     = imageType,
                                   .format        = format,
                                   .extent        = VkExtent3D{.width = width, .height = height, .depth = 1},
                                   .mipLevels     = 1,
                                   .arrayLayers   = arrayLayers,
                                   .samples       = static_cast<VkSampleCountFlagBits>(numSamples),
                                   .tiling        = VK_IMAGE_TILING_OPTIMAL,
                                   .usage         = VK_IMAGE_USAGE_SAMPLED_BIT | additionalUsageFlags,
                                   .sharingMode   = VK_SHARING_MODE_EXCLUSIVE,
                                   .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED};

    VkImageViewCreateInfo viewInfo{.sType    = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
                                   .viewType = (arrayLayers == 1 ? VK_IMAGE_VIEW_TYPE_2D : VK_IMAGE_VIEW_TYPE_2D_ARRAY),
                                   .format   = format,
                                   .subresourceRange = {.aspectMask = viewAspect, .levelCount = 1, .layerCount = arrayLayers}};

    NVVK_CHECK(allocator.createImage(image, imageInfo, viewInfo));
  }

  uint32_t      getWidth() const { return image.extent.width; }
  uint32_t      getHeight() const { return image.extent.height; }
  uint32_t      getLayers() const { return image.arrayLayers; }
  VkImageLayout getLayout() const { return image.descriptor.imageLayout; }
  VkFormat      getFormat() const { return image.format; }
  VkImageView   getView() const { return image.descriptor.imageView; }

  // To destroy the object, provide its context and allocator.
  void deinit(VkDevice device, nvvk::ResourceAllocator& allocator) { allocator.destroyImage(image); }

  void transitionTo(VkCommandBuffer cmdBuffer,
                    VkImageLayout   dstLayout)  // How the image will be laid out in memory.
  {
    // Note that in larger applications, we could batch together pipeline
    // barriers for better performance!

    // We always want to transition the whole image. Choose aspectMask based
    // on the image format:
    VkImageAspectFlags aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    if(VK_FORMAT_D32_SFLOAT_S8_UINT == image.format || VK_FORMAT_D24_UNORM_S8_UINT == image.format)
    {
      aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT;
    }
    else if(VK_FORMAT_D32_SFLOAT == image.format)
    {
      aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;
    }

    const nvvk::ImageMemoryBarrierParams params{.image            = image.image,
                                                .oldLayout        = image.descriptor.imageLayout,
                                                .newLayout        = dstLayout,
                                                .subresourceRange = VkImageSubresourceRange{
                                                    .aspectMask     = aspectMask,
                                                    .baseMipLevel   = 0,
                                                    .levelCount     = VK_REMAINING_MIP_LEVELS,
                                                    .baseArrayLayer = 0,
                                                    .layerCount     = VK_REMAINING_ARRAY_LAYERS,
                                                }};
    nvvk::cmdImageMemoryBarrier(cmdBuffer, params);

    // Update current layout
    image.descriptor.imageLayout = dstLayout;
  }

  // Should be called to keep track of the image's current layout when a render
  // pass that includes a image layout transition finishes.
  void endRenderPass(VkImageLayout dstLayout) { image.descriptor.imageLayout = dstLayout; }

  void setName(nvvk::DebugUtil& util, const char* name)
  {
    util.setObjectName(image.image, name);
    util.setObjectName(image.descriptor.imageView, name);
  }
};

// Adds a simple command that ensures that all transfer writes have finished before all
// subsequent fragment shader reads and writes (in the current scope).
// Note that on NV hardware, unless you need a layout transition, there's little benefit to using
// memory barriers for each of the individual objects (and in fact may run into issues with the
// Vulkan specification).
// The dependency flags are BY_REGION_BIT by default, since most calls to this function come from
// dependencies inside render passes, which require this (according to section 6.6.1).
inline void cmdTransferBarrierSimple(VkCommandBuffer cmdBuffer)
{
  const VkMemoryBarrier2 barrier{
      .sType         = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2,
      .srcStageMask  = VK_PIPELINE_STAGE_2_TRANSFER_BIT,
      .srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT,
      .dstStageMask  = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT,
      .dstAccessMask = VK_ACCESS_2_SHADER_READ_BIT | VK_ACCESS_2_SHADER_WRITE_BIT,
  };

  const VkDependencyInfo dependency{.sType              = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
                                    .dependencyFlags    = VK_DEPENDENCY_BY_REGION_BIT,
                                    .memoryBarrierCount = 1,
                                    .pMemoryBarriers    = &barrier};

  vkCmdPipelineBarrier2(cmdBuffer, &dependency);
}

// Adds a simple command that ensures that all fragment shader reads and writes have finished before all
// subsequent fragment shader reads and writes (in the current scope).
// Note that on NV hardware, unless you need a layout transition, there's little benefit to using
// memory barriers for each of the individual objects (and in fact may run into issues with the
// Vulkan specification).
// The dependency flags are BY_REGION_BIT by default, since most calls to this function come from
// dependencies inside render passes, which require this (according to section 6.6.1).
inline void cmdFragmentBarrierSimple(VkCommandBuffer cmdBuffer)
{
  const VkMemoryBarrier2 barrier{
      .sType         = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2,
      .srcStageMask  = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
      .srcAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT,
      .dstStageMask  = barrier.srcStageMask,
      .dstAccessMask = barrier.srcAccessMask,
  };

  const VkDependencyInfo dependency{.sType              = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
                                    .dependencyFlags    = VK_DEPENDENCY_BY_REGION_BIT,
                                    .memoryBarrierCount = 1,
                                    .pMemoryBarriers    = &barrier};

  vkCmdPipelineBarrier2(cmdBuffer, &dependency);
}

//-----------------------------------------------------------------------------
// CachingShaderCompiler

// TODO: Move this to a .cpp file; it's expensive!

using CompileDefines = std::vector<std::pair<std::string, std::string>>;

struct CompileInput
{
  shaderc_shader_kind   shader_kind;
  std::filesystem::path filename;
  CompileDefines        defines;
};

inline bool operator==(const CompileInput& lhs, const CompileInput& rhs)
{
  return (lhs.shader_kind == rhs.shader_kind) && (lhs.filename == rhs.filename) && (lhs.defines == rhs.defines);
}

// Hash function for CompileInput
template <>
struct std::hash<CompileInput>
{
  std::size_t operator()(const CompileInput& s) const noexcept
  {
    std::size_t hash = 0;
    // Since std::hash<std::filesystem::path> doesn't yet exist in VS 2019,
    // we use s.filename.native(), which can be hashed.
    nvutils::hashCombine(hash, s.filename.native());
    for(const auto& define : s.defines)
    {
      nvutils::hashCombine(hash, define.first, define.second);
    }
    nvutils::hashCombine(hash, s.shader_kind);
    return hash;
  }
};

struct ShaderCacheValue
{
  VkShaderModule                  module = VK_NULL_HANDLE;
  std::filesystem::file_time_type modified_time;
};

// A wrapper around ShaderC that outputs VkShaderModules and caches its results.
// It makes some simplifying assumptions around compilation settings.
struct CachingShaderCompiler
{
public:
  void addSearchPaths(const std::vector<std::filesystem::path>& paths) { m_compiler.addSearchPaths(paths); }

  VkShaderModule compile(VkDevice device, const CompileInput& input)
  {
    // Queue up the file modification time query
    const std::filesystem::path& absolutePath = nvutils::findFile(input.filename, m_compiler.searchPaths());
    if(absolutePath.empty())
    {
      // File not found
      return VK_NULL_HANDLE;
    }

    const std::filesystem::file_time_type modified_time = std::filesystem::last_write_time(absolutePath);

    // Is this file in our cache?
    const auto& it = m_cache.find(input);
    if(m_cache.end() != it)
    {
      // If the file hasn't been modified since then, we can use it directly.
      if(it->second.modified_time >= modified_time)
      {
        return it->second.module;
      }
    }

    // Missing or out-of-date from the cache. Compile it anew:
    ShaderCacheValue result{.modified_time = modified_time};

    {
      m_compiler.clearOptions();
      shaderc::CompileOptions& options = m_compiler.options();
      options.SetGenerateDebugInfo();
      for(const auto& define : input.defines)
      {
        options.AddMacroDefinition(define.first, define.second);
      }

      const shaderc::SpvCompilationResult compileResult = m_compiler.compileFile(absolutePath, input.shader_kind);
      const uint32_t*                     spirv         = m_compiler.getSpirv(compileResult);
      if(spirv)
      {
        VkShaderModuleCreateInfo shaderInfo{.sType    = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
                                            .codeSize = m_compiler.getSpirvSize(compileResult),
                                            .pCode    = spirv};
        NVVK_CHECK(vkCreateShaderModule(device, &shaderInfo, nullptr, &result.module));
        nvvk::DebugUtil::getInstance().setObjectName(result.module, input.filename.string());
      }
    }

    // Update the cache:
    m_cache[input] = result;
    return result.module;
  }

  void clear(VkDevice device)
  {
    for(const auto& kvp : m_cache)
    {
      vkDestroyShaderModule(device, kvp.second.module, nullptr);
    }
    m_cache.clear();
  }

  void deinit(VkDevice device) { clear(device); }

private:
  nvvkglsl::GlslCompiler                             m_compiler;
  std::unordered_map<CompileInput, ShaderCacheValue> m_cache;
};
