find_package(OpenCV REQUIRED)
add_library(common_lib INTERFACE)

target_include_directories(common_lib INTERFACE
    ${PROJECT_SOURCE_DIR}/common
)

target_link_libraries(common_lib
    INTERFACE
    ${OpenCV_LIBS}
)

# Shared-memory frame ring (compiled). Producer/consumer targets link this.
add_library(shm STATIC shm.cpp)
target_include_directories(shm PUBLIC ${PROJECT_SOURCE_DIR}/common)
target_link_libraries(shm PUBLIC ${OpenCV_LIBS} rt)   # shm_open needs librt

# Prometheus /metrics exposer (embedded HTTP server, separate from gRPC).
# prometheus-cpp is built + installed under dependency/ (see PROMETHEUS_CPP_ROOT
# in the top-level CMakeLists). Services link `metrics_lib` and get the headers.
add_library(metrics_lib STATIC metrics/metrics.cpp)
target_include_directories(metrics_lib PUBLIC ${PROJECT_SOURCE_DIR}/common/metrics)
target_link_libraries(metrics_lib PUBLIC
    prometheus-cpp::pull      # Exposer (civetweb) -- pulls in ::core
)