cmake_minimum_required(VERSION 3.16)
project(edge_ai_system LANGUAGES CXX)

# ---- Global Settings ----
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# ---- Build Type ----
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE RelWithDebInfo CACHE STRING "Build type" FORCE)
endif()

message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")

# ---- Compile Options ----
add_compile_options(
    -Wall
    -Wextra
    -Wpedantic
    -fno-omit-frame-pointer
)

# ---- Debug Symbols ----
set(CMAKE_CXX_FLAGS_DEBUG "-g -O0")
set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "-g -O2")

# ---- Sanitizers ----
option(ENABLE_ASAN "Enable AddressSanitizer" ON)
option(ENABLE_UBSAN "Enable UndefinedBehaviorSanitizer" ON)

if(ENABLE_ASAN)
    add_compile_options(-fsanitize=address)
    add_link_options(-fsanitize=address)
endif()

if(ENABLE_UBSAN)
    add_compile_options(-fsanitize=undefined)
    add_link_options(-fsanitize=undefined)
endif()

# ---- Dependencies ----
find_package(Protobuf REQUIRED)
find_package(gRPC REQUIRED)
find_package(Threads REQUIRED)

# prometheus-cpp: built + installed under dependency/ (like ONNX Runtime, it is
# NOT a system package). Provides the embedded HTTP server that answers
# `GET /metrics` -- a separate listener from each service's gRPC port.
# Re-install with: cmake --install <src>/_build --prefix <src>/_install
set(PROMETHEUS_CPP_ROOT "${CMAKE_SOURCE_DIR}/dependency/prometheus-cpp/_install"
    CACHE PATH "prometheus-cpp install prefix")
find_package(prometheus-cpp CONFIG REQUIRED
    PATHS ${PROMETHEUS_CPP_ROOT}/lib/cmake/prometheus-cpp NO_DEFAULT_PATH)

# ---- Proto Setup ----
set(PROTO_FILE ${CMAKE_SOURCE_DIR}/proto/services.proto)
get_filename_component(PROTO_PATH ${PROTO_FILE} PATH)

set(GENERATED_PROTO_SRC ${CMAKE_CURRENT_BINARY_DIR}/services.pb.cc)
set(GENERATED_PROTO_HDR ${CMAKE_CURRENT_BINARY_DIR}/services.pb.h)
set(GENERATED_GRPC_SRC ${CMAKE_CURRENT_BINARY_DIR}/services.grpc.pb.cc)
set(GENERATED_GRPC_HDR ${CMAKE_CURRENT_BINARY_DIR}/services.grpc.pb.h)

add_custom_command(
    OUTPUT
        ${GENERATED_PROTO_SRC}
        ${GENERATED_PROTO_HDR}
        ${GENERATED_GRPC_SRC}
        ${GENERATED_GRPC_HDR}
    COMMAND protobuf::protoc
    ARGS
        --grpc_out ${CMAKE_CURRENT_BINARY_DIR}
        --cpp_out ${CMAKE_CURRENT_BINARY_DIR}
        -I ${PROTO_PATH}
        --plugin=protoc-gen-grpc=$<TARGET_FILE:gRPC::grpc_cpp_plugin>
        ${PROTO_FILE}
    DEPENDS ${PROTO_FILE}
    COMMENT "Generating gRPC and protobuf sources"
)

add_library(inference_proto
    ${GENERATED_PROTO_SRC}
    ${GENERATED_GRPC_SRC}
)

target_link_libraries(inference_proto
    PUBLIC
    protobuf::libprotobuf
    gRPC::grpc++
    Threads::Threads
)

target_include_directories(inference_proto
    PUBLIC ${CMAKE_CURRENT_BINARY_DIR}
)

# ---- Tooling ----
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

include_directories(${PROJECT_SOURCE_DIR})

# ---- Subdirectories ----


add_subdirectory(common)
add_subdirectory(services/ingest_service)
add_subdirectory(services/inference_service)

# ---- Tests ----
# enable_testing() is what creates CTestTestfile.cmake in the build tree; without
# it `ctest` has nothing to run no matter how many test executables exist. Each
# add_test() below registers a binary whose EXIT CODE is the pass/fail verdict.
#   cmake --build build -j && ctest --test-dir build --output-on-failure
enable_testing()

add_executable(test_shm_writer ${CMAKE_SOURCE_DIR}/tests/test_shm_writer.cpp)
target_link_libraries(test_shm_writer PRIVATE rt)   # shm_open needs librt

# FrameQueue is header-only, so this needs no library -- just the include path.
add_executable(test_frame_queue ${CMAKE_SOURCE_DIR}/tests/test_frame_queue.cpp)
target_include_directories(test_frame_queue PRIVATE
    ${CMAKE_SOURCE_DIR}/services/inference_service/src
    ${CMAKE_SOURCE_DIR}/common
)
target_link_libraries(test_frame_queue PRIVATE Threads::Threads)
add_test(NAME frame_queue COMMAND test_frame_queue)

# RtspReader drags in gRPC, OpenCV and the shm segment, so it links ingest_core.
add_executable(test_rtsp_reader ${CMAKE_SOURCE_DIR}/tests/test_rtsp_reader.cpp)
target_include_directories(test_rtsp_reader PRIVATE
    ${CMAKE_SOURCE_DIR}/services/ingest_service/src
    ${CMAKE_SOURCE_DIR}/common
)
target_link_libraries(test_rtsp_reader PRIVATE ingest_core)
add_test(NAME rtsp_reader COMMAND test_rtsp_reader)
# Reconnect backoff on a dead URL makes this the slow one; fail rather than hang.
set_tests_properties(rtsp_reader PROPERTIES TIMEOUT 60)

# The Python suite runs under the same `ctest` invocation, so one command covers
# both languages. Skipped (not failed) when pytest isn't installed.
find_program(PYTEST_EXECUTABLE NAMES pytest)
if(PYTEST_EXECUTABLE)
    add_test(NAME python_unit
             COMMAND ${PYTEST_EXECUTABLE} -q ${CMAKE_SOURCE_DIR}/tests
             WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
    set_tests_properties(python_unit PROPERTIES TIMEOUT 120)
else()
    message(STATUS "pytest not found -- Python tests will not run under ctest")
endif()