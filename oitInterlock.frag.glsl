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


// OIT_INTERLOCK supports MSAA.
//
// The color pass sorts the frontmost OIT_LAYERS (depth, color) pairs per pixel
// in the A-buffer, tail blending colors that make it in.
// To do this, we insert the first OIT_LAYERS fragments; any further fragments
// then test to see if they're in the frontmost OIT_LAYERS fragments so far, and
// if so, replace the furthest fragment.
//
// If OIT_INTERLOCK_IS_ORDERED is set to 1, then insertion attempts are done in
// primitive order, so the selection of the fragment to tail blend in each
// invocation is guaranteed to be stable between frames. Additionally, this
// improves stability even without tail blending: if multiple fragments share
// the same depth, the one that's blended first is consistently defined.
//
// The resolve pass then sorts and blends the fragments from front to back.

#version 460
#extension GL_GOOGLE_include_directive : enable

#include "shaderCommon.glsl"

////////////////////////////////////////////////////////////////////////////////
// Color                                                                      //
////////////////////////////////////////////////////////////////////////////////
#if PASS == PASS_COLOR

#include "oitColorDepthDefines.glsl"

#extension GL_NV_fragment_shader_interlock : enable
#extension GL_ARB_fragment_shader_interlock : enable

#if GL_NV_fragment_shader_interlock || GL_ARB_fragment_shader_interlock

#if OIT_SAMPLE_SHADING
// Enables interlock on individual samples.
#if OIT_INTERLOCK_IS_ORDERED
layout(sample_interlock_ordered) in;
#else   // #if OIT_INTERLOCK_IS_ORDERED
layout(sample_interlock_unordered) in;
#endif  // #if OIT_INTERLOCK_IS_ORDERED
#else   // #if OIT_SAMPLE_SHADING
#if OIT_INTERLOCK_IS_ORDERED
layout(pixel_interlock_ordered) in;
#else   // #if OIT_INTERLOCK_IS_ORDERED
layout(pixel_interlock_unordered) in;
#endif  // #if OIT_INTERLOCK_IS_ORDERED
#endif  // #if OIT_SAMPLE_SHADING

#if GL_NV_fragment_shader_interlock
#define beginInvocationInterlock beginInvocationInterlockNV
#define endInvocationInterlock endInvocationInterlockNV
#else  // #if GL_NV_fragment_shader_interlock
#define beginInvocationInterlock beginInvocationInterlockARB
#define endInvocationInterlock endInvocationInterlockARB
#endif  // #if GL_NV_fragment_shader_interlock

#else  // #if GL_NV_fragment_shader_interlock || GL_ARB_fragment_shader_interlock
#pragma error "OIT_INTERLOCK requires GL_NV_fragment_shader_interlock or GL_ARB_fragment_shader_interlock!"
#endif  // #if GL_NV_fragment_shader_interlock || GL_ARB_fragment_shader_interlock

layout(binding = IMG_ABUFFER, abufferType) uniform coherent uimageBuffer imgAbuffer;
// Stores the number of fragments that have been processed by this pixel.
layout(binding = IMG_AUX, r32ui) uniform coherent uimage2DUsed imgAux;
// Stores the depth of the furthest fragment that was inserted into the A-buffer.
layout(binding = IMG_AUXDEPTH, r32ui) uniform coherent uimage2DUsed imgDepth;

layout(location = 0) in Interpolants IN;
layout(location = 0, index = 0) out vec4 outColor;

void main()
{
  // Get the unpremultiplied linear-space RGBA color of this pixel
  vec4 color = shading(IN);
  // Convert to unpremultiplied sRGB for 8-bit storage
  const vec4 sRGBColor = unPremultLinearToSRGB(color);

  // Compute index in the A-buffer
  const int viewSize = scene.viewport.z;
  const int listPos  = viewSize * OIT_LAYERS * sampleID + (coord.y * scene.viewport.x + coord.x);

  uvec4 storeValue = uvec4(packUnorm4x8(sRGBColor), floatBitsToUint(gl_FragCoord.z), storeMask, 0);

  // Critical section --
  beginInvocationInterlock();
#if USE_EARLYDEPTH
  uint oldDepth = imageLoad(imgDepth, coord).r;
  if(storeValue.y <= oldDepth)
#endif  // #if USE_EARLYDEPTH
  {
    const uint oldCounter = imageLoad(imgAux, coord).r;
    imageStore(imgAux, coord, uvec4(oldCounter + 1));

    if(oldCounter < OIT_LAYERS)
    {
      imageStore(imgAbuffer, listPos + int(oldCounter) * viewSize, storeValue);

      // Inserted, so we won't tail-blend it:
      color = vec4(0);
    }
    else
    {
      // Find the furthest element
      int  furthest = 0;
      uint maxDepth = 0;

      for(int i = 0; i < OIT_LAYERS; i++)
      {
        const uint testDepth = imageLoad(imgAbuffer, listPos + i * viewSize).g;
        if(testDepth > maxDepth)
        {
          maxDepth = testDepth;
          furthest = i;
        }
      }

      if(maxDepth > storeValue.g)
      {
        // Replace the furthest fragment, tail-blending it, with this fragment.
        color = unPremultSRGBToLinear(unpackUnorm4x8(imageLoad(imgAbuffer, listPos + furthest * viewSize).r));
        imageStore(imgAbuffer, listPos + furthest * viewSize, storeValue);
#if USE_EARLYDEPTH
        imageStore(imgDepth, coord, uvec4(maxDepth));
#endif  // #if USE_EARLYDEPTH
      }
    }
  }
  endInvocationInterlock();
// -- End critical section
#if OIT_TAILBLEND
  outColor = vec4(color.rgb * color.a, color.a);  // Premultiply the color
#endif                                            // #if OIT_TAILBLEND
}

#endif  // #if PASS == PASS_COLOR

////////////////////////////////////////////////////////////////////////////////
// Composite                                                                  //
////////////////////////////////////////////////////////////////////////////////
#if PASS == PASS_COMPOSITE

// We read from imgAbuffer and imgAux, and possibly write to imgColor.
// restrict means that different buffers will point to different locations in memory.
// A uimageBuffer is an unsigned storage texel buffer.
// We define uimage2DUsed in oitCompositeDefines.glsl - it's either uimage2DArray or uimage2D.

#include "oitCompositeDefines.glsl"

// Stores up to OIT_LAYERS fragments per (MSAA) sample and their depths.
// Im Vulkan, an imageBuffer maps to a Storage Texel Buffer.
layout(binding = IMG_ABUFFER, abufferType) uniform restrict readonly uimageBuffer imgAbuffer;
// Stores the number of fragments processed so far per (MSAA) sample.
layout(binding = IMG_AUX, r32ui) uniform restrict readonly uimage2DUsed imgAux;

layout(location = 0) out vec4 outColor;

void main()
{
  loadType array[OIT_LAYERS];

  vec4 color = vec4(0);

  // Get the number of pixels in the image.
  int viewSize = scene.viewport.z;
  // Get the index of the current sample at the current fragment.
  int listPos = viewSize * OIT_LAYERS * sampleID + coord.y * scene.viewport.x + coord.x;

  // Load the number of fragments for the given sample. Then load those
  // fragments and sort them.

  // The number of fragments for this sample.
  int fragments = int(imageLoad(imgAux, coord).r);
  fragments     = min(OIT_LAYERS, fragments);

  for(int i = 0; i < fragments; i++)
  {
    array[i] = loadOp(imageLoad(imgAbuffer, listPos + i * viewSize));
  }

  bubbleSort(array, fragments);

  vec4 colorSum = vec4(0);  // Initially completely transparent

#if OIT_COVERAGE_SHADING
  // Compute the blended color of each MSAA sample from the fragments stored
  // in the A-buffer. For each MSAA sample, we loop through the fragments
  // and see which fragments covered this sample.

  for(int s = 0; s < OIT_MSAA; s++)
  {
    vec4 sColor = vec4(0);
    for(int i = 0; i < fragments; i++)
    {
      if((array[i].b & (1 << s)) != 0)
      {
        doBlendPacked(sColor, array[i].r);
      }
    }
    colorSum += sColor;
  }
  colorSum /= OIT_MSAA;
#else
  // Blend all of the fragments together:
  for(int i = 0; i < fragments; i++)
  {
    doBlendPacked(colorSum, array[i].x);
  }
#endif

  outColor = colorSum;
}

#endif  // #if PASS == PASS_COMPOSITE