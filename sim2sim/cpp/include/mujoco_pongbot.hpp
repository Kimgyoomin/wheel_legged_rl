#pragma once

#include <array>
#include <string>
#include <unordered_map>

namespace pongbot {

class MujocoPongbot {
public:
    explicit MujocoPongbot(std::string model_path);

    void reset();
    std::array<float, 62> build_observation(const std::array<float, 16>& previous_action) const;
    void apply_action(const std::array<float, 16>& action);
    void step();

private:
    std::string model_path_;
    std::unordered_map<std::string, int> joint_id_by_name_;
    std::unordered_map<std::string, int> body_id_by_name_;

    // TODO: Add mjModel* / mjData* ownership and qpos/dof address mappings.
};

}  // namespace pongbot
