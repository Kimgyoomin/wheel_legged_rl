#include "onnx_policy.hpp"

#include <stdexcept>
#include <utility>

namespace pongbot {

OnnxPolicy::OnnxPolicy(std::string onnx_path) : onnx_path_(std::move(onnx_path)) {}

std::array<float, 16> OnnxPolicy::infer(const std::array<float, 62>&) {
    // TODO: Create ONNX Runtime session, bind 62D input, and return 16D policy action.
    throw std::runtime_error("OnnxPolicy::infer is scaffold-only until Python sim2sim is validated.");
}

}  // namespace pongbot
