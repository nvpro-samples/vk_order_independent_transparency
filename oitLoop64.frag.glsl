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


// OIT_LOOP64 is a variant of OIT_LOOP that combines the depth and color shader
// passes into one pass if the GPU supports 64-bit atomics. It does not support
// MSAA at the moment.
// The color pass sorts the frontmost OIT_LAYERS (depth, color) pairs per pixel
// in the A-buffer, in order from nearest to furthest, tail blending colors
// that didn't make it in. The resolve pass then blends the fragments from front
// to back.

// This relies on how for positive floating-point numbers x and y, x > y iff
// floatBitsToUint(x) > floatBitsToUint(y). As such, this depends on the
// viewport depths always being positive.

// The A-buffer is laid out like this:
// for each SSAA sample...
//   for each OIT layer...
//     for each pixel...
//       a r32ui depth value (via floatBitsToUint, cleared to 0xffffffff)
//       a r32ui packed sRGB unpremultiplied alpha color

#version 460
#extension GL_GOOGLE_include_directive : enable

#include "shaderCommon.glsl"

////////////////////////////////////////////////////////////////////////////////
// Color                                                                      //
////////////////////////////////////////////////////////////////////////////////
#if PASS == PASS_COLOR

#include "oitColorDepthDefines.glsl"

#extension GL_NV_shader_atomic_int64 : require
#extension GL_ARB_gpu_shader_int64 : require  // For uint64_t

// Note that this is now bound as a storage buffer, instead of a
// storage texel buffer.
layout(binding = IMG_ABUFFER, std430) coherent buffer ssboAbuffer
{
  uint64_t abuffer[];
};

layout(location = 0) in Interpolants IN;
layout(location = 0, index = 0) out vec4 outColor;

void main()
{
  // Get the unpremultiplied linear-space RGBA color of this pixel
  vec4 color = shading(IN);
  // Convert to unpremultiplied sRGB for 8-bit storage
  const vec4 sRGBColor = unPremultLinearToSRGB(color);

  // Compute base index in the A-buffer
  const int viewSize = scene.viewport.z;
  const int listPos  = viewSize * OIT_LAYERS * sampleID + (coord.y * scene.viewport.x + coord.x);

  bool canInsert = true;  // If false, canot be inserted into the A-buffer.

  // Store the color in the least significant bits and the depth in the most significant bits.
  uint64_t zcur = packUint2x32(uvec2(packUnorm4x8(sRGBColor), floatBitsToUint(gl_FragCoord.z)));
  int      i    = 0;  // Current position in the array

#if USE_EARLYDEPTH
  // Do some early tests to minimize the amount of insertion-sorting work we
  // have to do.
  // If the fragment is further away than the last depth fragment, skip it:
  uint64_t pretest = abuffer[listPos + (OIT_LAYERS - 1) * viewSize];
  if(zcur > pretest)
  {
    canInsert = false;
  }
  else
  {
    // Check to see if the fragment can be inserted in the latter half of the
    // depth array:
    pretest = abuffer[listPos + (OIT_LAYERS / 2) * viewSize];
    if(zcur > pretest)
    {
      i = (OIT_LAYERS / 2);
    }
  }
#endif

  bool evict = true;
  if(canInsert)
  {
    // Try to insert zcur in the place of the first element of the array that
    // is greater than or equal to it. In the former case, shift all of the
    // remaining elements in the array down.
    for(; i < OIT_LAYERS; i++)
    {
      uint64_t ztest = atomicMin(abuffer[listPos + i * viewSize], zcur);

      if(ztest == packUint2x32(uvec2(0xFFFFFFFFu, 0xFFFFFFFFu)))
      {
        // We just inserted zcur into an empty space in the array.
        evict = false;
        break;
      }

      zcur = (ztest > zcur) ? ztest : zcur;
    }
  }

  if(!evict)
  {
    // Inserted without having to evict, so make this color transparent:
    outColor = vec4(0);
  }
  else
  {
#if OIT_TAILBLEND
    // Unpack the color of the fragment that cannot fit into the A-buffer and
    // premultiply it
    const uvec2 current      = unpackUint2x32(zcur);
    const vec4  currentColor = unPremultSRGBToLinear(unpackUnorm4x8(current.x));
    outColor                 = vec4(currentColor.rgb * currentColor.a, currentColor.a);
#else   // #if OIT_TAILBLEND
    outColor = vec4(0);
#endif  // #if OIT_TAILBLEND
  }
}

#endif // #if PASS == PASS_COLOR

////////////////////////////////////////////////////////////////////////////////
// Composite                                                                  //
////////////////////////////////////////////////////////////////////////////////
#if PASS == PASS_COMPOSITE

// Gets the colors in the A-buffer (which are already sorted
// front to back) and bends them together.

#include "oitCompositeDefines.glsl"

layout(binding = IMG_ABUFFER, std430) restrict buffer ssboAbuffer
{
  uvec2 abuffer[];
};

layout(location = 0) out vec4 outColor;

void main()
{
  vec4 color = vec4(0);

  const int viewSize = scene.viewport.z;
  const int listPos  = viewSize * OIT_LAYERS * sampleID + (coord.y * scene.viewport.x + coord.x);

  for(int i = 0; i < OIT_LAYERS; i++)
  {
    uvec2 stored = abuffer[listPos + i * viewSize];
    if(stored.y != 0xFFFFFFFFu)
    {
      doBlendPacked(color, stored.x);
    }
    else
    {
      break;
    }
  }

  outColor = color;
}

#endif // #if PASS == PASS_COMPOSITE