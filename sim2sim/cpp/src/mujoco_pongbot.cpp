#include "mujoco_pongbot.hpp"

#include <stdexcept>
#include <utility>

namespace pongbot {

MujocoPongbot::MujocoPongbot(std::string model_path) : model_path_(std::move(model_path)) {
    // TODO: Load MuJoCo model/data and build name-based joint/body mappings.
}

void MujocoPongbot::reset() {
    // TODO: Reset free joint pose and actuated joint states to PongbotW contract defaults.
}

std::array<float, 62> MujocoPongbot::build_observation(const std::array<float, 16>&) const {
    // TODO: Build the 62D observation matching sim2sim/python/mujoco_onnx_policy_play.py.
    throw std::runtime_error("MujocoPongbot::build_observation is scaffold-only.");
}

void MujocoPongbot::apply_action(const std::array<float, 16>&) {
    // TODO: Decode 12 leg + 4 wheel actions, compute clipped torques, and apply to MuJoCo.
}

void MujocoPongbot::step() {
    // TODO: Advance with mj_step1 -> apply_action -> mj_step2, or fallback to mj_step.
}

}  // namespace pongbot
