/* Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

// A simple order-independent transparency technique that has an area in the
// A-buffer to store the first OIT_LAYERS fragments. The composite pass then
// sorts these fragments.

#version 460
#extension GL_GOOGLE_include_directive : enable

#include "shaderCommon.glsl"

////////////////////////////////////////////////////////////////////////////////
// Color                                                                      //
////////////////////////////////////////////////////////////////////////////////
#if PASS == PASS_COLOR

// We write to imgAbuffer and imgAux.
// The coherent keyword enforces safe coherency of writes and reads.
// A uimageBuffer is an unsigned storage texel buffer.
// We redefined uimage2DUsed above - it's either uimage2DArray or uimage2D.

#include "oitColorDepthDefines.glsl"

// Stores up to OIT_LAYERS fragments per (MSAA) sample and their depths.
layout(binding = IMG_ABUFFER, abufferType) uniform coherent uimageBuffer imgAbuffer;
// Stores the number of fragments processed so far per (MSAA) sample.
layout(binding = IMG_AUX, r32ui) uniform coherent uimage2DUsed imgAux;

layout(location = 0) in Interpolants IN;
layout(location = 0, index = 0) out vec4 outColor;

void main()
{
  // Get the unpremultiplied linear-space RGBA color of this pixel
  vec4 color = shading(IN);
  // Convert to unpremultiplied sRGB for 8-bit storage
  const vec4 sRGBColor = unPremultLinearToSRGB(color);

  // Get the number of pixels in the image
  const int viewSize = scene.viewport.z;  // The number of pixels in the image
  // Get the index of the current sample at the current fragment
  int listPos = viewSize * OIT_LAYERS * sampleID + coord.y * scene.viewport.x + coord.x;

  // For the first OIT_LAYERS fragments for this sample, store them in the
  // A-buffer. For the rest, blend them together using normal blending
  // if OIT_TAILBLEND is enabled, and ignore them otherwise.

  // We'll sort the elements in the A-buffer against the second component here;
  // the first and third components act as a payload. When using MSAA with
  // coverage shading, the third component let us know what MSAA samples this
  // element of the A-buffer covers.
  uvec4 storeValue = uvec4(packUnorm4x8(sRGBColor), floatBitsToUint(gl_FragCoord.z), storeMask, 0);

  // Get the previous number of fragments stored in the A-buffer for this sample,
  // and increment it.
  uint oldCounter = imageAtomicAdd(imgAux, coord, 1u);
  if(oldCounter < OIT_LAYERS)
  {
    imageStore(imgAbuffer, listPos + int(oldCounter) * viewSize, storeValue);

    // Inserted, so make this fragment transparent:
    outColor = vec4(0);
  }
  else
  {
#if OIT_TAILBLEND
    // Premultiply alpha
    outColor = vec4(color.rgb * color.a, color.a);
#else   // #if OIT_TAILBLEND
    outColor = vec4(0);  // Ignore tail-blended values
#endif  // #if OIT_TAILBLEND
  }
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