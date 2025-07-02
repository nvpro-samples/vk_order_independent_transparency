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


// OIT_SPINLOCK supports MSAA.
// The color pass sorts the frontmost OIT_LAYERS (depth, color) pairs per pixel
// in the A-buffer, unordered, tail blending colors that make it in. To do this,
// we insert the first OIT_LAYERS fragments; any further fragments then test to
// see if they're in the frontmost OIT_LAYERS fragments so far, and if so,
// replace the furthest fragment.
// The resolve pass then sorts and blends the fragments from front to back.

#version 460
#extension GL_GOOGLE_include_directive : enable

#include "shaderCommon.glsl"

////////////////////////////////////////////////////////////////////////////////
// Color                                                                      //
////////////////////////////////////////////////////////////////////////////////
#if PASS == PASS_COLOR

#include "oitColorDepthDefines.glsl"

layout(abufferType, binding = IMG_ABUFFER) uniform coherent uimageBuffer imgAbuffer;
layout(r32ui, binding = IMG_AUX) uniform coherent uimage2DUsed imgAux;
layout(r32ui, binding = IMG_AUXSPIN) uniform coherent uimage2DUsed imgSpin;
layout(r32ui, binding = IMG_AUXDEPTH) uniform coherent uimage2DUsed imgDepth;

layout(location = 0) in Interpolants IN;
layout(location = 0, index = 0) out vec4 outColor;

void main()
{
  // Get the unpremultiplied linear-space RGBA color of this ixel
  vec4 color = shading(IN);
  // Convert to unpremultiplied sRGB for 8-bit storage
  const vec4 sRGBColor = unPremultLinearToSRGB(color);

  // Compute index in the A-buffer
  const int viewSize = scene.viewport.z;
  const int listPos  = viewSize * OIT_LAYERS * sampleID + (coord.y * scene.viewport.x + coord.x);

  uvec4 storeValue = uvec4(packUnorm4x8(sRGBColor), floatBitsToUint(gl_FragCoord.z), storeMask, 0);

  // gl_order_independent_transparency has an #if for a different version of a
  // spinlock here, but since it's unstable (it flickers) and is disabled by
  // default, we don't implement it here.

#if USE_EARLYDEPTH
  uint oldDepth = imageLoad(imgDepth, coord).r;
  if(storeValue.y <= oldDepth)
#endif  // #if USE_EARLYDEPTH
  {
    // `done` tracks whether we've managed to complete the spinlock.
    // If the current thread is a helper thread, there's nothing to do.
    bool done = gl_SampleMaskIn[0] == 0;

    while(!done)
    {
      // Atomically set the value of imgSpin at coord to 1 ("in use").
      // If the original value was 0 (i.e. "this was the first thread to set it
      // to 1"), then we can enter the critical section.
      uint old = imageAtomicExchange(imgSpin, coord, 1u);
      if(old == 0u)
      {
        // Critical section --

        // See if there's enough space to avoid having to evict another fragment.
        const uint oldCounter = imageLoad(imgAux, coord).r;
        imageStore(imgAux, coord, uvec4(oldCounter + 1));

        if(oldCounter < OIT_LAYERS)
        {
          imageStore(imgAbuffer, listPos + int(oldCounter) * viewSize, storeValue);
          color = vec4(0);  // Inserted, so won't be tailblended
        }
        else
        {
          // Find the furthest element
          int  furthest = 0;
          uint maxDepth = 0;
          for(int i = 0; i < OIT_LAYERS; i++)
          {
            uint testDepth = imageLoad(imgAbuffer, listPos + i * viewSize).g;
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
        // -- End critical section
        imageAtomicExchange(imgSpin, coord, 0u);
        done = true;
      }
    }
  }

#if OIT_TAILBLEND
  outColor = vec4(color.rgb * color.a, color.a);  // Premultiply the color
#endif
}

#endif // #if PASS == PASS_COLOR

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

#endif // #if PASS == PASS_COMPOSITE