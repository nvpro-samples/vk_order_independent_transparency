# vk_order_independent_transparency

Demonstrates seven different techniques for order-independent transparency (OIT) in Vulkan.

![Shows a thousand semitransparent spheres on a gray background with a user interface in the top-left corner.](doc/vk_order_independent_transparency.png)

## About

This sample demonstrates seven different algorithms for rendering transparent objects without requiring them to be sorted in advance. Six of these algorithms produce ground-truth images if given enough memory, while a seventh produces fast and memory-efficient but approximate results. (Note that sorting alone isn't enough to blend transparent objects correctly, since the [painter's algorithm](https://en.wikipedia.org/wiki/Painter%27s_algorithm) can fail, while these six approaches can blend objects correctly.) 

This is useful whether you're rendering skyscraper facades, automobile exteriors, or rows of glasses on a table. This sample shows these techniques applied to hundreds of overlapping transparent and opaque spheres. It also shows how they can be implemented in Vulkan, such as by using subpass inputs for Weighted, Blended Order-Independent Transparency.

These techniques were presented in Christoph Kubisch's GTC 2014 talk, "Order Independent Transparency In OpenGL 4.x", which you can find at http://on-demand.gputechconf.com/gtc/2014/presentations/S4385-order-independent-transparency-opengl.pdf.

You can also hover over any of the elements in the UI inside the sample to find out more about what they do.

## Algorithm Descriptions

### Overview

This sample implements seven OIT algorithms: Simple, Linked List, Loop32, Loop64, Spinlock, Interlock, and Weighted, Blended Order-Independent Transparency (WBOIT). These operate per sample or per pixel, depending on the antialiasing mode.

Six of these (all but WBOIT) sort each fragment's color information based on depth so long as they have space to store all of the separate pieces of information. The amount of space used to store fragment information can be configured using the GUI. When they run out of space, they tail blend the remaining fragments using normal, non-order-independent transparency directly onto the color buffer (using [premultiplied alpha](https://developer.nvidia.com/content/alpha-blending-pre-or-not-pre)). Then they blend the sorted fragments on top. However, while Linked List, Loop32, Loop64, Spinlock, and Interlock always sort the frontmost few fragments per pixel/sample (tail blending the backmost samples), Simple sorts the first fragments it processes per pixel/sample.

WBOIT, on the other hand, uses a constant amount of space per pixel/sample, but weights instead of sorts fragments by depth before blending them.

Here's a quick overview of the properties of each algorithm. See the algorithm descriptions below for more details:

| Name        | Correctness Bound By           | MSAA Support | Bytes per Pixel or Sample                                    | Sorts Front | Number of Transparent Draws | Additional Extensions Required |
| ----------- | ------------------------------ | ------------ | ------------------------------------------------------------ | ----------- | --------------------------- | ------------------------------ |
| Simple      | `OIT_LAYERS`, draw order       | Yes          | `8*OIT_LAYERS+4`, or `16*OIT_LAYERS+4` (with antialiasing masks) | No          | 1                           | No                             |
| Linked List | `OIT_LAYERS` and A-buffer size | Yes          | `16` (per element) + `4`                                     | Yes         | 1                           | No                             |
| Loop32      | `OIT_LAYERS`                   | No           | `8*OIT_LAYERS+4`                                             | Yes         | 2                           | No                             |
| Loop64      | `OIT_LAYERS`                   | No           | `8*OIT_LAYERS+4`                                             | Yes         | 1                           | Yes                            |
| Spinlock    | `OIT_LAYERS`                   | Yes          | `8*OIT_LAYERS+12`, or `16*OIT_LAYERS+12` (with antialiasing masks) | Yes         | 1                           | No                             |
| Interlock   | `OIT_LAYERS`                   | Yes          | `16*OIT_LAYERS+8`, or `32*OIT_LAYERS+8` (with antialiasing masks) | Yes         | 1                           | Yes                            |
| WBOIT       | Approximation                  | Yes          | `20`                                                         | Yes         | 1                           | No                             |

This sample stores the vertex and index data for all of its spheres in a single mesh. It draws the faces corresponding to the last `100 - percentTransparent`% of spheres using an opaque shader, then draws the first `percentTransparent`% of spheres using the algorithm's `drawTransparent` method.

### Simple

For each pixel or sample, the color shader stores the colors and depths of the first `OIT_LAYERS` fragments it receives, and tail blends any subsequent fragments. The composite pass then sorts these fragments.

For instance, suppose `OIT_LAYERS` = 2 with no antialiasing, and a thread processes four `(RGBA color, depth)` pairs, `(c4, 0.4), (c2, 0.2), (c1, 0.1), (c3, 0.3)` (where `c1`...`c4` are RGBA colors). The color shader would store `(c4, 0.4), (c2, 0.2)` in the A-buffer, and tail blend `(c1, 0.1)` followed by `(c3, 0.3)` onto the color buffer (out of order, which is generally the case in tail blending). The composite shader would then sort the A-buffer values to get `(c2, 0.2), (c4, 0.4)`, then blend the colors from back to front. Note that in this case the frontmost fragment, `(c1, 0.1)`, was drawn behind everything else, because there were more overlapping objects than the A-buffer had space for! This would usually result in visible artifacts, but this algorithm can also work well enough if there's minimal overlap, or if objects are sorted in advance.

### Linked List

This algorithm builds a linked list of fragments for each pixel. To do this, it uses a single contiguous block of memory, an image storing the index of the head of each list, and a 1x1 image acting as an atomic counter containing the index of the first empty element in the block of memory, using 0 as a list terminator.

Each thread running the color shader atomically increments the counter to get the index of the element. If there's space left in the buffer, it writes its data and a pointer to the previous head of its linked list into that location; otherwise, it tail blends the fragment. Each thread running the composite shader then iterates down the linked list, gathering and sorting the first `OIT_LAYERS` elements and tail-blending the rest.

For instance, suppose `OIT_LAYERS=2` with no antialiasing, the A-buffer is 4 elements long (including one space for 0, the list terminator), and  `(color, depth)` fragments corresponding to two pixels are processed as follows:

* Pixel 1: `(c4, 0.4)`
* Pixel 2: `(c1, 0.1)`
* Pixel 1: `(c2, 0.2)`
* Pixel 1: `(c3, 0.3)`

At the end of the color shader, the A-buffer will contain the following `(color, depth, old offset)` values:

| 0                       | 1            | 2            | 3            |
| ----------------------- | ------------ | ------------ | ------------ |
| Empty (list terminator) | (c4, 0.4, 0) | (c1, 0.1, 0) | (c2, 0.2, 1) |

`(c3, 0.3)` was tail blended. When compositing, Pixel 1's thread will start at element 3 and gather and sort `(c2, 0.2)` and `(c4, 0.4)`, while Pixel 2's thread will start at element 2 and gather and sort `(c1, 0.1, 0)`.

### Loop32

This algorithm uses 32-bit atomic operations to first sort the depths of each pixel's frontmost `OIT_LAYERS` fragments. It then matches colors to depths, and blends the fragments in order.

For instance, given the four fragments in the Simple example with `OIT_LAYERS` = 2 without antialiasing, the depth shader would compute that the frontmost sorted depths are `(0.1, 0.2)`. This step would also tail blend `(c4, 0.4)` and `(c3, 0.3)`. It would then match the colors to the depths to get `(c1, c2)`, and then blend this together.

### Loop64

Loop32 uses three shaders, and requires drawing transparent objects twice. If the device supports the `VK_KHR_shader_atomic_int64` extension, then we can pack colors and depths together into a 64-bit integer, and sort colors and depths together by sorting the 64-bit integers. This requires us to only draw transparent objects once.

For instance, given the four fragments in the Simple example with `OIT_LAYERS` = 2 without antialiasing, the first shader could compute that the frontmost sorted depths and colors are `((c1, 0.1), (c2, 0.2))`, tail blending `(c4, 0.4)` and `(c3, 0.3)`. It would then blend the sorted colors together.

### Spinlock

This algorithm maintains a sorted list of the frontmost `OIT_LAYERS` fragments per pixel or sample using insertion sort. However, inserting elements into a list (and pushing all of the other elements back) is not thread-safe. This algorithm solves this problem by implementing a spinlock per pixel using atomic operations, which permits only one thread per pixel to insert elements at a time.

For instance, imagine the following scenario with `OIT_LAYERS`=2 without antialiasing, for a single pixel. Each of the threads is being run by a different warp.

* Thread 1 starts processing the fragment `(c3, 0.3)`. It enters the critical section. The A-buffer area for this pixel is still empty, `((0,0,0,0), 1), ((0,0,0,0), 1)`.
* Thread 2 starts processing the fragment `(c1, 0.1`). It sees that the critical section is occupied and starts spin waiting.
* Thread 3 starts processing the fragment `(c2, 0.2`). It sees that the critical section is occupied and starts spin waiting.
* Thread 1 inserts `(c3, 0.3)` and leaves the critical section. The A-buffer area for this pixel is now `(c3, 0.3), ((0,0,0,0), 1)`.
* Thread 3 sees that the critical section is unoccupied and enters the critical section.
* Thread 2 sees that the critical section is occupied and keeps spin waiting.
* Thread 3 inserts `(c2, 0.2)` into the first position and leaves the critical section. The A-buffer area for this pixel is now `(c2, 0.2), (c3, 0.3)`.
* Thread 4 starts processing the fragment `(c4, 0.4)`. It sees that it would be behind the last fragment in the A-buffer and tail blends `(c4, 0.4)`, then exits.
* Thread 2 sees that the critical section is unoccupied and enters the critical section. It inserts `(c1, 0.1)` into the first position, removing and tail blending `(c3, 0.3)`. It then leaves the critical section, finishing execution. The A-buffer area for this pixel is now `(c1, 0.1), (c2, 0.2)`.

### Interlock

If the device supports the `VK_EXT_fragment_shader_interlock` extension, then we can use invocation interlocking to prevent multiple invocations from entering a critical section, without having to implement a spin lock (and without requiring the threads to spin while they wait for the critical section to be unoccupied). This is somewhat similar to rasterizer order views in Direct3D 11.3.

To do this, we call `beginInvocationInterlockARB` or `beginInvocationInterlockNV` before entering the critical section (depending on whether the GLSL code supports the `GL_ARB_fragment_shader_interlock` or `GL_NV_fragment_shader_interlock`extension), then call `endInvocationInterlockARB` or `endInvocationInterlockNV` to end the critical section.

This is similar to the example for Spinlock, except with spin locks replaced by invocation interlocks.

### Weighted, Blended Order-Independent Transparency

Weighted, Blended Order-Independent Transparency ([McGuire and Bavoil 2013](http://jcgt.org/published/0002/02/09/)) assigns a weight to each fragment, then commutatively blends their colors together. By assigning higher weights for more important pixels, it can emulate some of the effects of layered opacity - such as how closer fragments usually affect the final color more than further fragments - without having to sort the fragments. However, it can also diverge from the ground truth in scenarios where order strongly affects the result, such as when opacity is high.

Here, we compute a weight from each fragment's depth and RGBA color, as described in `oitWeighted.frag.glsl`. For each pixel, we then compute the following quantities, where `color_0`, `color_1`, ... are premultiplied RGBA colors, and `weight_0`, `weight_1`, ... are the floating-point weights of each fragment:

`outColor = (weight_0 * color_0) + (weight_1 * color_1) + ...`

i.e. the weighted premultiplied sum, and

`outReveal = (1 - color_0.a) * (1 - color_1.a) * ...`

i.e. one minus the opacity of the result. This can be done using blending modes. In the resolve pass, we then get the average weighted RGB color, `outColor.rgb/outColor.a`, and blend it onto the image with the opacity of the result, `1 - outReveal`, using a variant of premultiplied alpha to use `outReveal` directly.

## Code Layout

This sample's main class is declared in `oit.h`, which includes descriptions for most of its functions. Its function definitions are split into four files:

* `oitRender.cpp` contains the most important drawing code.
* `oit.cpp` shows the parts of Vulkan object creation that are important for OIT.
* `oitGui.cpp` implements the GUI.
* `main.cpp` contains the rest of the functions, most of which are not as important for OIT (such as framebuffer and generic graphics pipeline generation).

`utilities_vk.h` contains some Vulkan helper objects which are specific to this sample, but make object management a bit easier.

`common.h` contains defines shared between C++ and GLSL code.

The shader files are laid out as follows:

* `oitInterlock.frag.glsl`, `oitLinkedList.frag.glsl`, `oitLoop.frag.glsl`, `oitLoop64.frag.glsl`, `oitSimple.frag.glsl`, `oitSpinlock.frag.glsl`, and `oitWeighted.frag.glsl` contain the main shader code for each of the seven algorithms. They all use the same structure, so you can diff them to see the variations in each implementation.
* `fullScreenTriangle.vert.glsl` generates a full-screen triangle, used for screen-space passes.
* `object.vert.glsl` is the vertex shader for rendering objects.
* `opaque.frag.glsl` is the fragment shader for opaque objects, applying basic Gooch shading.
* `oitColorDepthDefines.glsl`, `oitCompositeDefines.glsl`, and `shaderCommon.glsl` contain common defines and functions used across GLSL files.

## Building

To build this sample, first install a recent [Vulkan SDK](https://www.lunarg.com/vulkan-sdk/). Then do one of the following:

* To clone all NVIDIA DesignWorks Samples, clone https://github.com/nvpro-samples/build_all, then run one of the `clone_all` scripts in that directory.
* Or to get the files for this sample without the other samples, clone this repository as well as https://github.com/nvpro-samples/shared_sources into a single directory. On Windows, you'll need to clone https://github.com/nvpro-samples/shared_external into the directory as well.

You can then use CMake to generate and subsequently build the project.

## Additional Notes

* Since the GTC 2014 talk, at least two new OIT techniques have been presented that are also worth considering:
  * [Moment-Based Order-Independent Transparency (Münstermann et al. 2018)](http://momentsingraphics.de/I3D2018.html) is a family of algorithms that operate somewhat like WBOIT, but use higher-order moments to produce a more accurate image.
  * It's also possible to create an image with correctly rendered semitransparent objects directly without sorting using ray tracing, whether by computing attenuation after each intersection, or by using stochastic transparency. For more information and for a tutorial of how to implement stochastic transparency, please see the [NVIDIA Vulkan Ray Tracing Tutorials](https://github.com/nvpro-samples/vk_raytracing_tutorial_KHR).
* The six A-buffer-based OIT algorithms implement antialiasing through manually blending MSAA sample masks, combining that with A-buffer storage per sample instead of per pixel, or through supersampling. However, there are also many other ways to implement antialiasing with order-independent transparency techniques, and both accuracy and performance should be considered in the context of application implementations.
* Other layouts for the A-buffer, such as using image arrays, could be more performant in terms of clearing and cache efficiency.

For further reading, please see:

*Multi-Layer Alpha Blending* by Marco Salvi and Karthik Vaidyanathan: https://software.intel.com/content/www/us/en/develop/articles/multi-layer-alpha-blending.html

*Efficient Layered Fragment Buffer Techniques* by Pyarelal Knowles, Geoff Leach, and Fabio Zambetta: http://openglinsights.com/bendingthepipeline.html

*Freepipe: programmable parallel rendering architecture for efficient multi-fragment effects* by Fang Liu, Mengcheng Huang, Xuehui Liu, and Enhua Wu: https://sites.google.com/site/hmcen0921/cudarasterizer

*k+-buffer: Fragment Synchronized k-buffer* by Andreas A. Vasilakis and Ioannis Fudos: www.cgrg.cs.uoi.gr/wp-content/uploads/bezier/publications/abasilak-ifudos-i3d2014/k-buffer.pdf

*Real-time concurrent linked list construction on the GPU* by Jason C. Yang, Justin Hensley, Holger Grün, and Nicolas Thibieroz: https://dl.acm.org/doi/10.1111/j.1467-8659.2010.01725.x

*Stochastic Transparency* by Eric Enderton, Erik Sintorn, Peter Shirley and David Luebke: http://enderton.org/eric/pub/stochtransp-tvcg.pdf

*Interactive Order-Independent Transparency* by Cass Everitt: http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.18.9286&rep=rep1&type=pdf