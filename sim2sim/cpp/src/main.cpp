#include "mujoco_pongbot.hpp"
#include "onnx_policy.hpp"

#include <iostream>
#include <string>

int main(int argc, char** argv) {
    std::string model_path;
    std::string onnx_path;

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--model" && i + 1 < argc) {
            model_path = argv[++i];
        } else if (arg == "--onnx" && i + 1 < argc) {
            onnx_path = argv[++i];
        }
    }

    std::cout << "model: " << model_path << "\n";
    std::cout << "onnx:  " << onnx_path << "\n";
    // TODO: Wire keyboard input, observation construction, ONNX inference, torque application, and CSV logging.
    return 0;
}
