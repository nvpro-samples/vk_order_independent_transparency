cmake_minimum_required(VERSION 3.11...3.31)
Project(vk_order_independent_transparency VERSION 2.0 LANGUAGES CXX)

# Add the cmake folder to the module path
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/cmake")
find_package(NvproCore2 REQUIRED)

# Source files for this project
file(GLOB SOURCE_FILES *.cpp *.hpp *.inl *.h *.c)
file(GLOB GLSL_FILES shaders/*.*)
source_group("Source Files" FILES ${SOURCE_FILES})
source_group("GLSL Files" FILES ${GLSL_FILES})

add_executable(${PROJECT_NAME} ${SOURCE_FILES} ${GLSL_FILES})
set_property(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY VS_STARTUP_PROJECT ${PROJECT_NAME})

# Link with nvpro_core2 libraries we need
target_link_libraries(${PROJECT_NAME} PRIVATE
  nvpro2::nvapp
  nvpro2::nvgui
  nvpro2::nvutils
  nvpro2::nvvk
  nvpro2::nvvkglsl
)

add_project_definitions(${PROJECT_NAME})

# Copy runtime dependencies to build and install directories
copy_to_runtime_and_install(${PROJECT_NAME}
  LOCAL_DIRS "${CMAKE_CURRENT_LIST_DIR}/shaders"
  AUTO
)
