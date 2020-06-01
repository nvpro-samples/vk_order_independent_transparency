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

// Includes defines used for color and depth pases, including a statement
// that forces early depth testing.

// Important!
//
// This forces the depth/stencil test to be run prior to executing the shader.
// Otherwise, shaders that make use of shader writes could have fragments that
// execute even though the depth/stencil pass fails - in this case, transparent
// surfaces behind opaque objects (that we don't want to include in OIT)
// See https://www.khronos.org/opengl/wiki/Early_Fragment_Test
// In addition to that, post_depth_coverage also makes it so that gl_SampleMaskIn[]
// reflects the sample mask after depth testing, instead of before. This fixes
// problems with transparent and opaque objects in MSAA, and implicitly enables
// early_fragment_tests.
// See https://www.khronos.org/registry/OpenGL/extensions/ARB/ARB_post_depth_coverage.txt
#extension GL_ARB_post_depth_coverage : enable
layout(post_depth_coverage) in;

// If OIT_COVERAGE_SHADING is used, then the a-buffer uses three components;
// otherwise, it uses two.
#if OIT_COVERAGE_SHADING
#define abufferType rgba32ui
#define storeMask gl_SampleMaskIn[0]
#else  // #if OIT_COVERAGE_SHADING
#define abufferType rg32ui
#define storeMask 0
#endif  // #if OIT_COVERAGE_SHADING

#if OIT_SAMPLE_SHADING && OIT != OIT_WEIGHTED
#define uimage2DUsed uimage2DArray
#define sampleID gl_SampleID
ivec3 coord = ivec3(gl_FragCoord.xy, gl_SampleID);
#else  // #if OIT_SAMPLE_SHADING && OIT != OIT_WEIGHTED
#define uimage2DUsed uimage2D
#define sampleID 0
ivec2 coord = ivec2(gl_FragCoord.xy);
#endif  // #if OIT_SAMPLE_SHADING && OIT != OIT_WEIGHTED