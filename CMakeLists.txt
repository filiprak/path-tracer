cmake_minimum_required(VERSION 3.1)

project(path-tracer)

set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake" ${CMAKE_MODULE_PATH})

# Set up include and lib paths
set(EXTERNAL "external")
set(QT_SRC "qt")
include_directories("${EXTERNAL}")
include_directories("${EXTERNAL}/include/")

if(${CMAKE_SYSTEM_NAME} MATCHES "Linux")
    set(EXTERNAL_LIB_PATH "${EXTERNAL}/lib/linux" "/usr/lib64")
elseif(WIN32 OR WIN64)
    set(EXTERNAL_LIB_PATH "${EXTERNAL}/lib/win")
endif()
link_directories(${EXTERNAL_LIB_PATH})
list(APPEND CMAKE_LIBRARY_PATH "${EXTERNAL_LIB_PATH}")


# Fing required packages

# QT5
set(CMAKE_AUTOMOC ON)
set(CMAKE_PREFIX_PATH "C:\\Qt\\5.9.3\\msvc2015\\")
include_directories("${QT_SRC}/include/")

find_package(Qt5Core REQUIRED)
find_package(Qt5Widgets REQUIRED)
find_package(Qt5Gui REQUIRED)

if (Qt5Core_FOUND AND Qt5Widgets_FOUND AND Qt5Gui_FOUND)
	message("-- Found ${Qt5Core_LIBRARIES}")
	message("-- Found ${Qt5Gui_LIBRARIES}")
	message("-- Found ${Qt5Widgets_LIBRARIES}")
else(Qt5Core_FOUND AND Qt5Widgets_FOUND AND Qt5Gui_FOUND)
	message("-- Qt5 required packages not found !")
endif()

# ASSIMP
set(ASSIMP_INCLUDE_DIRS "${EXTERNAL}/include")
set(ASSIMP_LIBRARY_DIRS "${CMAKE_LIBRARY_PATH}")
find_package(ASSIMP REQUIRED)

# OPENGL
find_package(OpenGL REQUIRED)


set(CORELIBS
	"${Qt5Core_LIBRARIES}"
	"${Qt5Widgets_LIBRARIES}"
	"${Qt5Gui_LIBRARIES}"
    "${OPENGL_LIBRARY}"
	"${ASSIMP_LIBRARY}"
    )
	
# Enable C++11 for host code
set(CMAKE_CXX_STANDARD 11)

# Enable CUDA debug info in debug mode builds
list(APPEND CUDA_NVCC_FLAGS_DEBUG -G -g)

# Specify CUDA nvcc flags
list(APPEND CUDA_NVCC_FLAGS -arch=sm_35)

# Linux-specific hacks/fixes
if(${CMAKE_SYSTEM_NAME} MATCHES "Linux")
    list(APPEND CMAKE_EXE_LINKER_FLAGS "-lX11 -lXxf86vm -lXrandr -lXi")
endif()

# CUDA linking
find_package(Threads REQUIRED)
find_package(CUDA REQUIRED)

set(CUDA_ATTACH_VS_BUILD_RULE_TO_CUDA_FILE ON)
set(CUDA_SEPARABLE_COMPILATION ON)


set(SRC_PATH "src")
include_directories("${SRC_PATH}")

# set up source files
file(GLOB_RECURSE SOURCES
	"${QT_SRC}/*.h"
	"${QT_SRC}/*.hpp"
    "${QT_SRC}/*.cpp"
    "${SRC_PATH}/*.h"
	"${SRC_PATH}/*.hpp"
    "${SRC_PATH}/*.cpp"
	"${SRC_PATH}/*.cu"
	"${SRC_PATH}/*.cuh"
)

cuda_add_executable(${CMAKE_PROJECT_NAME}
    ${SOURCES}
    )

target_link_libraries(${CMAKE_PROJECT_NAME}
    ${CORELIBS}
    )

add_custom_command(
    TARGET ${CMAKE_PROJECT_NAME}
    POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
        ${CMAKE_SOURCE_DIR}/shaders
        ${CMAKE_BINARY_DIR}/shaders
    )
