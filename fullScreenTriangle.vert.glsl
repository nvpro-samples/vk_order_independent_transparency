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


#version 460

// Draws a full-screen triangle.
// This is used for full-screen passes over images,
// such as during the resolve step.
// We could also do this using a compute shader instead.

void main()
{
  // 0---^-----------2
  // |   |   |     /
  // <---.---|---/---> x+
  // |   |   | /
  // |-------/
  // |   | /
  // |   /
  // | / |
  // 1   V
  //     y+
  vec4 pos = vec4((float((gl_VertexIndex >> 1U) & 1U)) * 4.0 - 1.0, (float(gl_VertexIndex & 1U)) * 4.0 - 1.0, 0, 1.0);
  gl_Position = pos;
}