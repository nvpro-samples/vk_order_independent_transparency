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

#extension GL_GOOGLE_include_directive : enable

#include "common.h"

struct Interpolants
{
  vec3  pos;     // World-space vertex position
  vec3  normal;  // World-space vertex normal
  vec4  color;   // Linear-space color
  float depth;   // Z coordinate after applying the view matrix (larger = further away)
};

// Gooch shading!
// Interpolates between white and a cooler color based on the angle
// between the normal and the light.
vec3 goochLighting(vec3 normal)
{
  // Light direction
  vec3 light = normalize(vec3(-1, 2, 1));
  // cos(theta), remapped into [0,1]
  float warmth = dot(normalize(normal), light) * 0.5 + 0.5;
  // Interpolate between warm and cool colors (alpha here will be ignored)
  return mix(vec3(0, 0.25, 0.75), vec3(1, 1, 1), warmth);
}

// Applies Gooch shading to a surface with color and alpha and returns
// an unpremultiplied RGBA color.
vec4 shading(const Interpolants its)
{
  vec3 colorRGB = its.color.rgb * goochLighting(its.normal);

  // Calculate transparency in [alphaMin, alphaMin+alphaWidth]
  float alpha = clamp(scene.alphaMin + its.color.a * scene.alphaWidth, 0, 1);

  return vec4(colorRGB, alpha);
}

// Converts an unpremultiplied scalar from linear space to sRGB. Note that
// this does not match the standard behavior outside [0,1].
float unPremultLinearToSRGB(float c)
{
  if(c < 0.0031308f)
  {
    return c * 12.92f;
  }
  else
  {
    return (pow(c, 1.0f / 2.4f) * 1.055f) - 0.055f;
  }
}

// Converts an unpremultiplied RGB color from linear space to sRGB. Note that
// this does not match the standard behavior outside [0,1].
vec4 unPremultLinearToSRGB(vec4 c)
{
  c.r = unPremultLinearToSRGB(c.r);
  c.g = unPremultLinearToSRGB(c.g);
  c.b = unPremultLinearToSRGB(c.b);
  return c;
}

// Converts an unpremultiplied scalar from sRGB to linear space. Note that
// this does not match the standard behavior outside [0,1].
float unPremultSRGBToLinear(float c)
{
  if(c < 0.04045f)
  {
    return c / 12.92f;
  }
  else
  {
    return pow((c + 0.055f) / 1.055f, 2.4f);
  }
}

// Converts an unpremultiplied RGBA color from sRGB to linear space. Note that
// this does not match the standard behavior outside [0,1].
vec4 unPremultSRGBToLinear(vec4 c)
{
  c.r = unPremultSRGBToLinear(c.r);
  c.g = unPremultSRGBToLinear(c.g);
  c.b = unPremultSRGBToLinear(c.b);
  return c;
}

// Sets color to the result of blending color over baseColor.
// Color and baseColor are both premultiplied colors.
void doBlend(inout vec4 color, vec4 baseColor)
{
  color.rgb += (1 - color.a) * baseColor.rgb;
  color.a += (1 - color.a) * baseColor.a;
}

// Sets color to the result of blending color over fragment.
// Color and fragment are both premultiplied colors; fragment
// is an rgba8 sRGB unpremultiplied color packed in a 32-bit uint.
void doBlendPacked(inout vec4 color, uint fragment)
{
  vec4 unpackedColor = unpackUnorm4x8(fragment);
  // Convert from unpremultiplied sRGB to premultiplied alpha
  unpackedColor = unPremultSRGBToLinear(unpackedColor);
  unpackedColor.rgb *= unpackedColor.a;
  doBlend(color, unpackedColor);
}