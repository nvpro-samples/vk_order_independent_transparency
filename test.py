# Smoke test runner (does it run?) for vk_order_independent_transparency.
#
# This calls the sample with a list of test cases in the form of a sequence
# script (if we ever need more than Windows' limit of 8191 characters we could
# save it to a temporary file) and then lets main() handle the rest.
#
# The main advantage of running test cases inside the sample is speed;
# we only need to initialize the Vulkan context once. The downside is that
# it'll stop on the first failure.
import subprocess, sys

# We start with the default, some special cases, and all the cases other than algorithms:
script = """SEQUENCE "init"
SEQUENCE "interlock_unordered"
--algorithm 5 --interlockIsOrdered 0
SEQUENCE "opaque"
--percentTransparent 0
SEQUENCE "opaque_msaa4"
--percentTransparent 0 --aaType 1
SEQUENCE "opaque_ssaa4"
--percentTransparent 0 --aaType 2
SEQUENCE "3objects"
--numObjects 3
SEQUENCE "lowsubdiv"
--subdiv 2
SEQUENCE "scaleMin"
--scaleMin 1.0
SEQUENCE "scaleWidth"
--scaleWidth 10.0
"""

# Now build the algorithm test matrix: each algorithm, with and without tail
# blending, in each of the different antialiasing modes.
algorithmNames = [
    "simple",
    "linkedlist",
    "loop",
    "loop64",
    "spinlock",
    "interlock",
    "weighted",
]
tailNames = ["notail", "tail"]
aaNames = ["noaa", "msaa4", "ssaa4", "super4", "msaa8", "ssaa8"]
for algorithm in range(len(algorithmNames)):
    for tail in range(len(tailNames)):
        for aa in range(len(aaNames)):
            script += f'SEQUENCE "{algorithmNames[algorithm]}_{tailNames[tail]}_{aaNames[aa]}"\n'
            script += (
                f"--algorithm {algorithm} --tailBlend {tail} --aaType {aa}\n"
            )

print("Script:")
print(script)

# Some CI systems have problems with negative exit codes, so we make sure to
# catch any errors:
try:
    subprocess.run(
        [
            "_install/vk_order_independent_transparency.exe",
            "--windowSize",
            "800",
            "512",
            "--sequencestring",
            script,
        ],
        check=True,
    )
except Exception as e:
    print(e)
    sys.exit(1)
