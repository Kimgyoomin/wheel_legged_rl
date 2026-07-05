#pragma once

#include <array>
#include <string>

namespace pongbot {

class OnnxPolicy {
public:
    explicit OnnxPolicy(std::string onnx_path);

    std::array<float, 16> infer(const std::array<float, 62>& obs);

private:
    std::string onnx_path_;
    // TODO: Replace with concrete Ort::Env / Ort::Session members.
};

}  // namespace pongbot
