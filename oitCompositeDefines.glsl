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

// Includes defines used for composite passes, as well as sorting functions
// that depend upon these defines.

// If OIT_COVERAGE_SHADING is used, then the a-buffer uses three components;
// otherwise, it uses two.
#if OIT_COVERAGE_SHADING
#define abufferType rgba32ui
#define loadType uvec3
#define loadOp(a) (a).rgb
#else  // #if OIT_COVERAGE_SHADING
#define abufferType rg32ui
#define loadType uvec2
#define loadOp(a) (a).rg
#endif  // #if OIT_COVERAGE_SHADING

#if OIT_SAMPLE_SHADING
#define uimage2DUsed uimage2DArray
#define sampleID gl_SampleID
ivec3 coord = ivec3(gl_FragCoord.xy, gl_SampleID);
#else  // #if OIT_SAMPLE_SHADING && OIT != OIT_WEIGHTED
#define uimage2DUsed uimage2D
#define sampleID 0
ivec2 coord = ivec2(gl_FragCoord.xy);
#endif  // #if OIT_SAMPLE_SHADING && OIT != OIT_WEIGHTED

// These sorting routines depend on loadType, so we define them here.

// Sorts the first n elements of array in ascending order
// according to their second components using bubble sort
// (this is O(n^2)!)
void bubbleSort(inout uvec2 array[OIT_LAYERS], int n)
{
#if OIT_LAYERS > 1
  for(int i = (n - 2); i >= 0; --i)
  {
    for(int j = 0; j <= i; ++j)
    {
      if(uintBitsToFloat(array[j].g) >= uintBitsToFloat(array[j + 1].g))
      {
        // Swap array[j] and array[j+1]
        uvec2 temp   = array[j + 1];
        array[j + 1] = array[j];
        array[j]     = temp;
      }
    }
  }
#endif  // #if OIT_LAYERS > 1
}

// Also define a 3-component version of bubbleSort if OIT_MSAA is defined
#if OIT_MSAA
void bubbleSort(inout uvec3 array[OIT_LAYERS], int n)
{
#if OIT_LAYERS > 1
  for(int i = (n - 2); i >= 0; --i)
  {
    for(int j = 0; j <= i; ++j)
    {
      if(uintBitsToFloat(array[j].g) >= uintBitsToFloat(array[j + 1].g))
      {
        // Swap array[j] and array[j+1]
        uvec3 temp   = array[j + 1];
        array[j + 1] = array[j];
        array[j]     = temp;
      }
    }
  }
#endif  // #if OIT_LAYERS > 1
}
#endif  // #if OIT_MSAA

// Inserts a new item into array so that array remains sorted in increasing
// order according to its second components.
void insertionSort(inout loadType array[OIT_LAYERS], loadType newitem)
{
  for(int i = 0; i < OIT_LAYERS; ++i)
  {
    if(uintBitsToFloat(newitem.g) < uintBitsToFloat(array[i].g))
    {
      // shift rest
      for(int j = OIT_LAYERS - 1; j > i; j--)
      {
        array[j] = array[j - 1];
      }
      array[i] = newitem;
      return;
    }
  }
}

// If we know that newitem will often be the largest element of the array,
// we can perform insertion sort more quickly by checking if we need to move
// anything.
// This inserts a new item into array so that the array remains sorted in
// increasing order according to its second components.
loadType insertionSortTail(inout loadType array[OIT_LAYERS], loadType newitem)
{
  loadType newlast = newitem;
  if(uintBitsToFloat(newitem.g) < uintBitsToFloat(array[OIT_LAYERS - 1].g))
  {
    for(int i = 0; i < OIT_LAYERS; ++i)
    {
      if(uintBitsToFloat(newitem.g) < uintBitsToFloat(array[i].g))
      {
        //newlast = oldlast;
        newlast = array[OIT_LAYERS - 1];
        // shift rest
        for(int j = OIT_LAYERS - 1; j > i; j--)
        {
          array[j] = array[j - 1];
        }
        array[i] = newitem;
        break;
      }
    }
  }

  return newlast;
}