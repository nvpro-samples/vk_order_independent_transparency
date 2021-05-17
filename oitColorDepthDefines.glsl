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