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

// Common file shared between C++ and GLSL.

// Vertex shader attribute indexes, so that we don't reuse them:
#define VERTEX_POS 0
#define VERTEX_NORMAL 1
#define VERTEX_COLOR 2

// Uniform buffer object indexes
#define UBO_SCENE 0

#define IMG_ABUFFER 1
#define IMG_AUX 2
#define IMG_AUXSPIN 3
#define IMG_AUXDEPTH 4
#define IMG_COUNTER 5
#define IMG_COLOR 6
#define IMG_WEIGHTED_COLOR 7
#define IMG_WEIGHTED_REVEAL 8

// Although these are formally enums, we use #defines here to make them
// compatible with GLSL.

// OIT algorithms
#define OIT_SIMPLE 0
#define OIT_LINKEDLIST 1
#define OIT_LOOP 2
#define OIT_LOOP64 3
#define OIT_SPINLOCK 4
#define OIT_INTERLOCK 5
#define OIT_WEIGHTED 6
#define NUM_ALGORITHMS 7

// OIT passes
#define PASS_DEPTH 0
#define PASS_COLOR 1
#define PASS_COMPOSITE 2

#define AA_NONE 0
#define AA_MSAA_4X 1
#define AA_SSAA_4X 2
#define AA_SUPER_4X 3
#define AA_MSAA_8X 4
#define AA_SSAA_8X 5
#define NUM_AATYPES 6

// Affects several techniques, does a coarse depth-test to avoid
// longer-lasting actions (helps when many layers are used)
#define USE_EARLYDEPTH 1

// SceneData Uniform Buffer Object
#ifdef __cplusplus
// Note: This assumes that <nvmath/nvmath_glsltypes.h> has already been included.
using namespace nvmath;  // Make nvmath::mat4 correspond to mat4, e.g.
#endif                   // #ifdef __cplusplus

struct SceneData
{
  // Vectors are multiplied on the right.
  mat4 projViewMatrix;
  mat4 viewMatrix;
  mat4 viewMatrixInverseTranspose;

  ivec3 viewport;  // (width, height, width*height)
  // For SIMPLE, INTERLOCK, SPINLOCK, LOOP, and LOOP64, the number of OIT layers;
  // for LINKEDLIST, the total number of elements in the A-buffer.
  uint linkedListAllocatedPerElement;

  float alphaMin;
  float alphaWidth;
  vec2  _pad1;
};

// GLSL-only code
#ifndef __cplusplus

// Uniform buffer object for scene data
layout(std140, binding = UBO_SCENE) uniform sceneBuffer
{
  SceneData scene;
};

#ifndef OIT_LAYERS
#define OIT OIT_INTERLOCK
#define OIT_LAYERS 8
#define OIT_LOOP_DEPTH
#define OIT_TAILBLEND 1
#define OIT_MSAA 8
#define OIT_SAMPLE_SHADING 1
#endif

// When using MSAA, we can either use the coverage shading technique (not
// coverage-to-alpha! This stores the coverage (i.e. MSAA sample mask) of each
// fragment in the A-buffer) or sample shading (lower-level supersampling; each
// sample gets its own array in the A-buffer).
// We want to use coverage shading if using MSAA and not using sample shading.
#define OIT_COVERAGE_SHADING ((OIT_MSAA != 1) && (OIT_SAMPLE_SHADING == 0))

#endif  // #ifndef __cplusplus