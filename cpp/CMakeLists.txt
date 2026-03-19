cmake_minimum_required(VERSION 3.0)
project(ZED_with_OpenCV)

option(LINK_SHARED_ZED "Link with the ZED SDK shared executable" ON)

if(NOT LINK_SHARED_ZED AND MSVC)
    message(FATAL_ERROR "LINK_SHARED_ZED OFF: ZED SDK static libraries not available on Windows")
endif()

if(COMMAND cmake_policy)
    cmake_policy(SET CMP0003 OLD)
    cmake_policy(SET CMP0015 OLD)
endif()

set(EXECUTABLE_OUTPUT_PATH ".")

# ---- Core dependencies (both targets) ----------------------------------------
find_package(ZED 3 REQUIRED)
find_package(OpenCV 4 REQUIRED)
find_package(CUDA ${ZED_CUDA_VERSION} EXACT REQUIRED)

# ---- GStreamer (zed-headless only) -------------------------------------------
find_package(PkgConfig REQUIRED)
pkg_check_modules(GST REQUIRED
    gstreamer-1.0
    gstreamer-app-1.0
    gstreamer-rtsp-server-1.0
)

# ---- iceoryx2 (zed-headless only) -------------------------------------------
# Installed inside the Docker image; see docker/Dockerfile for build steps.
# cmake package name differs from the GitHub repo name:
find_package(iceoryx2-cxx 0.8.1 REQUIRED)

# ---- Platform libs -----------------------------------------------------------
if(NOT WIN32)
    add_definitions(-Wno-format-extra-args)
    set(SPECIAL_OS_LIBS "pthread" "X11")
endif()

# ---- Common include / link directories ---------------------------------------
include_directories(
    ${CUDA_INCLUDE_DIRS}
    ${ZED_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS}
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

link_directories(
    ${ZED_LIBRARY_DIR}
    ${OpenCV_LIBRARY_DIRS}
    ${CUDA_LIBRARY_DIRS}
)

# ---- Shared ZED link set -----------------------------------------------------
if(LINK_SHARED_ZED)
    set(ZED_LIBS ${ZED_LIBRARIES} ${CUDA_CUDA_LIBRARY} ${CUDA_CUDART_LIBRARY})
else()
    set(ZED_LIBS ${ZED_STATIC_LIBRARIES} ${CUDA_CUDA_LIBRARY} ${CUDA_LIBRARY})
endif()

# ==============================================================================
# Target 1: ZED_with_OpenCV  (original display sample, C++14)
# Sources listed explicitly so zed_headless.cpp is not included.
# ==============================================================================
file(GLOB HDR_FILES include/*.hpp)

add_executable(${PROJECT_NAME}
    src/main.cpp
    src/SaveDepth.cpp
    ${HDR_FILES}
)

target_compile_options(${PROJECT_NAME} PRIVATE -std=c++14 -O3)

target_link_libraries(${PROJECT_NAME}
    ${ZED_LIBS}
    ${OpenCV_LIBRARIES}
    ${SPECIAL_OS_LIBS}
)

# ==============================================================================
# Target 2: zed-headless  (headless streaming binary, C++17)
#
# Features:
#   --shm      iceoryx2 publish-subscribe IPC (zero-copy, multi-subscriber)
#   --rtsp     RTSP streaming via GStreamer gst-rtsp-server
#   --record   File recording via OpenCV VideoWriter
#
# Install GStreamer dependencies on Manjaro/Arch:
#   sudo pacman -S gstreamer gst-plugins-base gst-plugins-good \
#                  gst-plugins-bad gst-rtsp-server \
#                  gst-plugin-libav          # for x264enc
#   # For NVIDIA hardware encoding (optional):
#   sudo pacman -S gst-plugins-bad           # contains nvh264enc
#
# Recommended receiver (any platform):
#   ffplay rtsp://<host>:8554/left
#   vlc    rtsp://<host>:8554/left
# ==============================================================================
add_executable(zed-headless
    src/zed_headless.cpp
)

# iceoryx2-cxx requires C++17.
target_compile_options(zed-headless PRIVATE -std=c++17 -O3)

target_include_directories(zed-headless PRIVATE
    ${CUDA_INCLUDE_DIRS}
    ${ZED_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS}
    ${GST_INCLUDE_DIRS}
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

link_directories(${GST_LIBRARY_DIRS})

target_link_libraries(zed-headless
    ${ZED_LIBS}
    ${OpenCV_LIBRARIES}
    ${GST_LIBRARIES}
    ${SPECIAL_OS_LIBS}
    iceoryx2-cxx::static-lib-cxx
)
