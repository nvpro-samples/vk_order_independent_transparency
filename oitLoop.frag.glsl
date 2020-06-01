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

// OIT_LOOP does not support MSAA at the moment.
// It uses two passes and a resolve pass; the first stores the depths of the
// frontmost OIT_LAYERS fragments per pixel in the A-buffer, in order from
// nearest to farthest. Then the second pass writes the sorted colors into
// another section of the A-buffer, and tail blends colors that didn't make it in.
// The resolve pass then blends the fragments from front to back.

// This relies on how for positive floating-point numbers x and y, x > y iff
// floatBitsToUint(x) > floatBitsToUint(y). As such, this depends on the
// viewport depths always being positive.

// The A-buffer is laid out like this:
// for each SSAA sample...
//   for each OIT layer...
//     for each pixel...
//       a r32ui depth value (via floatBitsToUint, cleared to 0xffffffff)
//     for each pixel...
//       a packed color in a uvec4

#version 460
#extension GL_GOOGLE_include_directive : enable

#include "shaderCommon.glsl"

////////////////////////////////////////////////////////////////////////////////
// Depth sorting pass                                                         //
////////////////////////////////////////////////////////////////////////////////
#if PASS == PASS_DEPTH

#include "oitColorDepthDefines.glsl"

layout(binding = IMG_ABUFFER, r32ui) uniform coherent uimageBuffer imgAbuffer;

layout(location = 0) in Interpolants IN;
layout(location = 0, index = 0) out vec4 outColor;

void main()
{
  // The number of pixels in the image
  const int viewSize = scene.viewport.z;
  const int listPos  = viewSize * OIT_LAYERS * 2 * sampleID + (coord.y * scene.viewport.x + coord.x);

  // Insert the floating-point depth (reinterpreted as a uint) into the list of depths
  uint zcur = floatBitsToUint(gl_FragCoord.z);
  int  i    = 0;  // Current position in the array

#if USE_EARLYDEPTH
  // Do some early tests to minimize the amount of insertion-sorting work we
  // have to do.
  // If the fragment is further away than the last depth fragment, skip it:
  uint pretest = imageLoad(imgAbuffer, listPos + (OIT_LAYERS - 1) * viewSize).x;
  if(zcur > pretest)
    return;
  // Check to see if the fragment can be inserted in the latter half of the
  // depth array:
  pretest = imageLoad(imgAbuffer, listPos + (OIT_LAYERS / 2) * viewSize).x;
  if(zcur > pretest)
    i = (OIT_LAYERS / 2);
#endif  // #if USE_EARLYDEPTH

  // Try to insert zcur in the place of the first element of the array that
  // is greater than or equal to it. In the former case, shift all of the
  // remaining elements in the array down.
  for(; i < OIT_LAYERS; i++)
  {
    const uint ztest = imageAtomicMin(imgAbuffer, listPos + i * viewSize, zcur);
    if(ztest == 0xFFFFFFFFu || ztest == zcur)
    {
      // In the former case, we just inserted zcur into an empty space in the
      // array. In the latter case, we found a depth value that exactly matched.
      break;
    }
    zcur = max(ztest, zcur);
  }

  // Note that this line is necessary, since otherwise we'll get a warning from
  // the validation layer saying that undefined values will be written.
  // TODO: See if we can remove this
  outColor = vec4(0);
}

#endif // #if PASS == PASS_DEPTH

////////////////////////////////////////////////////////////////////////////////
// Color                                                                      //
////////////////////////////////////////////////////////////////////////////////
#if PASS == PASS_COLOR

// For each fragment, we look up its depth in the sorted array of depths.
// If we find a match, we write the fragment's color into the corresponding
// place in an array of colors. Otherwise, we tail blend it if enabled.

#include "oitColorDepthDefines.glsl"

layout(binding = IMG_ABUFFER, r32ui) uniform coherent uimageBuffer imgAbuffer;

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
  const int listPos = viewSize * OIT_LAYERS * 2 * sampleID + (coord.y * scene.viewport.x + coord.x);

  const uint zcur = floatBitsToUint(gl_FragCoord.z);

#if USE_EARLYDEPTH
  // If this fragment was behind the frontmost OIT_LAYERS fragments, it didn't
  // make it in, so tail blend it:
  if(imageLoad(imgAbuffer, listPos + (OIT_LAYERS - 1) * viewSize).x < zcur)
  {
#if OIT_TAILBLEND
    // Premultiply alpha
    outColor = vec4(color.rgb * color.a, color.a);
#else   // #if OIT_TAILBLEND
    outColor = vec4(0);
#endif  // #if OIT_TAILBLEND
    return;
  }
#endif  // #if USE_EARLYDEPTH

  // Use binary search to determine which index this depth value corresponds to
  // At each step, we know that it'll be in the closed interval [start, end].
  int start = 0;
  int end = (OIT_LAYERS - 1);
  uint ztest;
  while(start < end)
  {
    int mid = (start + end) / 2;
    ztest = imageLoad(imgAbuffer, listPos + mid * viewSize).x;
    if(ztest < zcur)
    {
      start = mid + 1;  // in [mid + 1, end]
    }
    else
    {
      end = mid;  // in [start, mid]
    }
  }

  // We now have start == end. Insert the packed color into the A-buffer at
  // this index.
  imageStore(imgAbuffer, listPos + (OIT_LAYERS + start) * viewSize, uvec4(packUnorm4x8(sRGBColor)));

  // Inserted, so make this color transparent:
  outColor = vec4(0);
}

#endif // #if PASS == PASS_COLOR

////////////////////////////////////////////////////////////////////////////////
// Composite                                                                  //
////////////////////////////////////////////////////////////////////////////////
#if PASS == PASS_COMPOSITE

// Gets the colors in the second part of the A-buffer (which are already sorted
// front to back) and bends them together.

#include "oitCompositeDefines.glsl"

layout(binding = IMG_ABUFFER, r32ui) uniform restrict readonly uimageBuffer imgAbuffer;

layout(location = 0) out vec4 outColor;

void main()
{
  vec4 color = vec4(0);

  const int viewSize = scene.viewport.z;
  int       listPos  = viewSize * OIT_LAYERS * 2 * sampleID + (coord.y * scene.viewport.x + coord.x);

  // Count the number of fragments for this pixel
  int fragments = 0;
  for(int i = 0; i < OIT_LAYERS; i++)
  {
    const uint ztest = imageLoad(imgAbuffer, listPos + i * viewSize).r;
    if(ztest != 0xFFFFFFFFu)
    {
      fragments++;
    }
    else
    {
      break;
    }
  }

  // Jump ahead to the color portion of the A-buffer
  listPos += viewSize * OIT_LAYERS;

  for(int i = 0; i < fragments; i++)
  {
    doBlendPacked(color, imageLoad(imgAbuffer, listPos + i * viewSize).r);
  }

  outColor = color;
}

#endif // #if PASS == PASS_COMPOSITE