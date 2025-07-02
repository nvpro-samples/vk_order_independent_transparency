/*
 * Copyright (c) 2020-2025, NVIDIA CORPORATION.  All rights reserved.
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
 * SPDX-FileCopyrightText: Copyright (c) 2020-2025 NVIDIA CORPORATION
 * SPDX-License-Identifier: Apache-2.0
 */


#version 460
#extension GL_GOOGLE_include_directive : enable

#include "shaderCommon.glsl"

layout(location = VERTEX_POS) in vec3 inPosition;
layout(location = VERTEX_NORMAL) in vec3 inNormal;
layout(location = VERTEX_COLOR) in vec4 inColor;

layout(location = 0) out Interpolants OUT;

void main()
{
  gl_Position = scene.projViewMatrix * vec4(inPosition, 1.0);
  OUT.depth   = (scene.viewMatrix * vec4(inPosition, 1.0)).z;
  OUT.pos     = inPosition;
  OUT.normal  = inNormal;
  OUT.color   = inColor;
}