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

// Implements Weighted, Blended Order-Independent Transparency,
// from http://casual-effects.blogspot.de/2014/03/weighted-blended-order-independent.html.
// This is an approximate order-independent transparency method.
// The idea is that we assign each fragment an arbitrarily chosen weight,
// here based on its depth, transparency, and color. Then we compute the
// following quantities, where color0, color1, ... are premultiplied:
// outColor: (weight0 * color0.rgba) + (weight1 * color1.rgba) + ...
//   (i.e. the weighted premultiplied sum, and)
// outReveal: (1-color0.a) * (1-color1.a) * ...
//   (i.e. 1 minus the opacity of the result).
// Then in the resolve pass, get the average weighted RGB color,
// outColor.rgb/outColor.a, and blend it onto the image with the opacity
// of the result. There's one more trick here; assuming it's being blended
// onto an opaque surface, we can use the GL blend mode
// GL_ONE_MINUS_SRC_ALPHA, GL_SRC_ALPHA
// using outReveal (instead of 1-outReveal) as the alpha channel to blend
// onto the image.

#version 460
#extension GL_GOOGLE_include_directive : enable

#include "shaderCommon.glsl"

////////////////////////////////////////////////////////////////////////////////
// Color                                                                      //
////////////////////////////////////////////////////////////////////////////////
#if PASS == PASS_COLOR

layout(location = 0) in Interpolants IN;
layout(location = 0) out vec4 outColor;
layout(location = 1) out float outReveal;

void main()
{
  vec4 color = shading(IN);
  color.rgb *= color.a;  // Premultiply it

  // Insert your favorite weighting function here. The color-based factor
  // avoids color pollution from the edges of wispy clouds. The z-based
  // factor gives precedence to nearer surfaces.

  // The depth functions in the paper want a camera-space depth of 0.1 < z < 500,
  // but the scene at the moment uses a range of about 0.01 to 50, so multiply
  // by 10 to get an adjusted depth:
  const float depthZ = -IN.depth * 10.0f;

  const float distWeight = clamp(0.03 / (1e-5 + pow(depthZ / 200, 4.0)), 1e-2, 3e3);

  float alphaWeight = min(1.0, max(max(color.r, color.g), max(color.b, color.a)) * 40.0 + 0.01);
  alphaWeight *= alphaWeight;

  const float weight = alphaWeight * distWeight;

  // GL Blend function: GL_ONE, GL_ONE
  outColor = color * weight;

  // GL blend function: GL_ZERO, GL_ONE_MINUS_SRC_ALPHA
  outReveal = color.a;
}

#endif // #if PASS == PASS_COLOR

////////////////////////////////////////////////////////////////////////////////
// Composite                                                                  //
////////////////////////////////////////////////////////////////////////////////
#if PASS == PASS_COMPOSITE

#if OIT_MSAA != 1
layout(input_attachment_index = 0, binding = IMG_WEIGHTED_COLOR) uniform subpassInputMS texColor;
layout(input_attachment_index = 1, binding = IMG_WEIGHTED_REVEAL) uniform subpassInputMS texWeights;
#else
layout(input_attachment_index = 0, binding = IMG_WEIGHTED_COLOR) uniform subpassInput texColor;
layout(input_attachment_index = 1, binding = IMG_WEIGHTED_REVEAL) uniform subpassInput texWeights;
#endif

layout(location = 0) out vec4 outColor;

void main()
{
#if OIT_MSAA != 1
  vec4  accum  = subpassLoad(texColor, gl_SampleID);
  float reveal = subpassLoad(texWeights, gl_SampleID).r;
#else
  vec4 accum = subpassLoad(texColor);
  float reveal = subpassLoad(texWeights).r;
#endif
  // GL blend function: GL_ONE_MINUS_SRC_ALPHA, GL_SRC_ALPHA
  outColor = vec4(accum.rgb / max(accum.a, 1e-5), reveal);
}

#endif // #if PASS == PASS_COMPOSITE