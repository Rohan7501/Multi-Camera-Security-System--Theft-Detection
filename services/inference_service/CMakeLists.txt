find_package(OpenCV REQUIRED)
# ONNX Runtime is NOT a system package (like prometheus-cpp, see root CMakeLists.txt);
# it is vendored under dependency/. Override with -DONNXRUNTIME_ROOT=... if yours lives elsewhere.
set(ONNXRUNTIME_ROOT "${CMAKE_SOURCE_DIR}/dependency/onnxruntime-linux-x64-gpu-1.22.0"
    CACHE PATH "ONNX Runtime install root")

# target_include_directories(inference_core
#     PUBLIC
#     ${ONNXRUNTIME_ROOT}/include
# )

add_library(inference_core
    src/detector_onnx.cpp
    src/inference_server.cpp
)

target_include_directories(inference_core PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${ONNXRUNTIME_ROOT}/include
    ${CMAKE_BINARY_DIR}
)

target_link_libraries(inference_core
    common_lib
    inference_proto              # generates inference-ingest.grpc.pb.h; must build first
    metrics_lib                  # inference_server.cpp records ingest_module_latency
    gRPC::grpc++
    ${OpenCV_LIBS}
    ${ONNXRUNTIME_ROOT}/lib/libonnxruntime.so
)

# add_executable(inference_service
#     src/main.cpp
# )

add_executable(inference_service
    src/main.cpp
    src/inference_server.cpp
    src/detector_onnx.cpp
    src/preprocess.cpp
    src/postprocess.cpp
    src/nms.cpp
    src/inference_worker.cpp
    src/tracking_client.cpp
)

target_link_libraries(inference_service
    PRIVATE
    inference_core
    inference_proto
    shm
    metrics_lib              # Prometheus /metrics exposer (own HTTP port)
    gRPC::grpc++
    protobuf::libprotobuf
    ${OpenCV_LIBS}
)

set_target_properties(inference_service PROPERTIES
    # BUILD_RPATH "/home/rohan/RIT/Spring Sem/edge-ai-system/onnxruntime-linux-x64-gpu-1.17.3/lib"
    BUILD_RPATH "${ONNXRUNTIME_ROOT}/lib"
)
set_target_properties(inference_core PROPERTIES
    BUILD_RPATH "${ONNXRUNTIME_ROOT}/lib"
)

add_executable(onnx_test
    ${CMAKE_SOURCE_DIR}/tests/test_onnx_EP.cpp
    src/detector_onnx.cpp
    src/preprocess.cpp
    src/postprocess.cpp
    src/nms.cpp
)

target_include_directories(onnx_test PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${ONNXRUNTIME_ROOT}/include
)

target_link_libraries(onnx_test PRIVATE
    common_lib
    ${ONNXRUNTIME_ROOT}/lib/libonnxruntime.so
    ${OpenCV_LIBS}
)

set_target_properties(onnx_test PROPERTIES
    BUILD_RPATH "${ONNXRUNTIME_ROOT}/lib"
)

# target_link_libraries(inference_service
#     inference_core
# )