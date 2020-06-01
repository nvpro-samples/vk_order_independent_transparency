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

// OIT_LINKEDLIST builds a linked list of fragments for each pixel.
// (It supports MSAA.) It treats the A-buffer as a single large array, so the
// pointers in the linked list are indexes in this array.
// imgAux stores the head of each linked list per pixel. When we need to store
// a new fragment, we find an empty space in the array using an atomic counter,
// add a new linked list node pointing to the previous head, and set the head
// to the new linked list node. 0 represents nullptr here, and is used as a
// list terminator.

#version 460
#extension GL_GOOGLE_include_directive : enable

#include "shaderCommon.glsl"

////////////////////////////////////////////////////////////////////////////////
// Color                                                                      //
////////////////////////////////////////////////////////////////////////////////
#if PASS == PASS_COLOR

#include "oitColorDepthDefines.glsl"

layout(binding = IMG_ABUFFER, rgba32ui) uniform coherent uimageBuffer imgAbuffer;
layout(binding = IMG_AUX, r32ui) uniform coherent uimage2DUsed imgAux;
// One major difference from the OpenGL version is that we use a 1x1 image here
// instead of an atomic counter variable.
layout(binding = IMG_COUNTER, r32ui) uniform uimage2D imgCounter;

layout(location = 0) in Interpolants IN;
layout(location = 0, index = 0) out vec4 outColor;

void main()
{
  // +1 as 0 is used as a list terminator. Adds 1 to imgCounter[0,0] and
  // returns the original value.
  const uint newOffset = imageAtomicAdd(imgCounter, ivec2(0), 1) + 1;
  // Get the unpremultiplied linear-space RGBA color of this pixel
  const vec4 color = shading(IN);

  if(newOffset >= scene.linkedListAllocatedPerElement)
  {
    // we ran out of memory, so tail-blend using premultiplied alpha if allowed
#if OIT_TAILBLEND
    outColor = vec4(color.rgb * color.a, color.a);  // Premultiply alpha
#else
    outColor = vec4(0); // Make the fragment transparent
#endif  // #if OIT_TAILBLEND
    return;
  }

  // Note that this is indeed thread-safe! The order in which threads
  // reach this line determines the order of the fragments in the linked list.
  const uint oldOffset = imageAtomicExchange(imgAux, coord, newOffset);

  // Convert to unpremultiplied sRGB for 8-bit storage
  const vec4 sRGBColor = unPremultLinearToSRGB(color);

  const uvec4 storeValue = uvec4(packUnorm4x8(sRGBColor),          //
                                 floatBitsToUint(gl_FragCoord.z),  //
                                 storeMask,                        //
                                 oldOffset);

  imageStore(imgAbuffer, int(newOffset), storeValue);

  outColor = vec4(0);
}

#endif // #if PASS == PASS_COLOR

////////////////////////////////////////////////////////////////////////////////
// Composite                                                                  //
////////////////////////////////////////////////////////////////////////////////
#if PASS == PASS_COMPOSITE

// See oitScene.frag.glsl for a description of this technique.
// For each coordinate, we iterate over the linked list, then like OIT_SIMPLE,
// sort and blend the first OIT_LAYERS fragments. The difference is that remaining
// elements in the linked list are tail blended.

#include "oitCompositeDefines.glsl"

layout(binding = IMG_ABUFFER, abufferType) uniform restrict readonly uimageBuffer imgAbuffer;
layout(binding = IMG_AUX, r32ui) uniform restrict readonly uimage2DUsed imgAux;

layout(location = 0) out vec4 outColor;

void main()
{
  loadType array[OIT_LAYERS];

  vec4 color     = vec4(0);
  int  fragments = 0;  // The number of fragments for this sample.

  uint startOffset = imageLoad(imgAux, coord).r;

  // Traverse the linked list:
  while(startOffset != uint(0) && fragments < OIT_LAYERS)
  {
    const uvec4 stored = imageLoad(imgAbuffer, int(startOffset));
    array[fragments]   = loadOp(stored);
    fragments++;

    startOffset = stored.a;
  }

  // Sort the fragments:
  bubbleSort(array, fragments);

  // Process the remaining fragments
  vec4 tailColor = vec4(0);

  while(startOffset != uint(0))
  {
    uvec4 stored = imageLoad(imgAbuffer, int(startOffset));
// Push the value into the array and tail-blend the furthest value
// that comes out:
#if OIT_TAILBLEND
    loadType tail = insertionSortTail(array, loadOp(stored));
    doBlendPacked(tailColor, tail.r);
#else   // #if OIT_TAILBLEND
    insertionSort(array, loadOp(stored));
#endif  // #if OIT_TAILBLEND
    startOffset = stored.a;
  }

  vec4 colorSum = vec4(0);

#if OIT_COVERAGE_SHADING
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

  // Finally, blend the frontmost fragments over the tail-blended fragments
  doBlend(colorSum, tailColor);
  outColor = colorSum;
}

#endif // #if PASS == PASS_COMPOSITE