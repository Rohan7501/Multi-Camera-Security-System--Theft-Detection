option(ENABLE_VISUALIZATION "Enable RTSP visualization" OFF)

if(ENABLE_VISUALIZATION)
    add_compile_definitions(ENABLE_VISUALIZATION)
endif()

# ==============================
# Dependencies
# ==============================

find_package(yaml-cpp REQUIRED)
find_package(OpenCV REQUIRED)

# ==============================
# RTSP Reader Library
# ==============================

add_library(rtsp_reader STATIC
    src/rtsp_reader.cpp
)

target_include_directories(rtsp_reader
    PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)

target_link_libraries(rtsp_reader
    PUBLIC
    common_lib
    inference_proto          # generated services.grpc.pb.h lives in build/
    shm                      # createShm / writeFrame / getFrame
    gRPC::grpc++
    ${OpenCV_LIBS}
)

# ==============================
# Ingest Core Library
# ==============================

add_library(ingest_core STATIC
    src/ingest_server.cpp
    src/ingest_admin_service.cpp
    src/rtsp_reader.cpp
    src/inference_client.cpp
)

target_include_directories(ingest_core
    PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)


target_link_libraries(ingest_core
    PUBLIC
    inference_proto   
    gRPC::grpc++
    protobuf::libprotobuf
    opencv_core
    opencv_videoio
    shm
    rtsp_reader
    common_lib
    yaml-cpp
)

add_executable(ingest_service
    src/main.cpp
)

target_link_libraries(ingest_service
    PRIVATE
    ingest_core
    metrics_lib              # Prometheus /metrics exposer (own HTTP port)
)
