cmake_minimum_required(VERSION 3.16)
project(pongbot_mujoco_onnx CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(MUJOCO_INCLUDE_DIR "" CACHE PATH "Path to MuJoCo include directory")
set(MUJOCO_LIBRARY "" CACHE FILEPATH "Path to MuJoCo library file")
set(ONNXRUNTIME_INCLUDE_DIR "" CACHE PATH "Path to ONNX Runtime include directory")
set(ONNXRUNTIME_LIBRARY "" CACHE FILEPATH "Path to ONNX Runtime library file")

add_executable(pongbot_mujoco_onnx
    src/main.cpp
    src/onnx_policy.cpp
    src/mujoco_pongbot.cpp
)

target_include_directories(pongbot_mujoco_onnx PRIVATE
    include
    ${MUJOCO_INCLUDE_DIR}
    ${ONNXRUNTIME_INCLUDE_DIR}
)

if (MUJOCO_LIBRARY)
    target_link_libraries(pongbot_mujoco_onnx PRIVATE ${MUJOCO_LIBRARY})
endif()

if (ONNXRUNTIME_LIBRARY)
    target_link_libraries(pongbot_mujoco_onnx PRIVATE ${ONNXRUNTIME_LIBRARY})
endif()
