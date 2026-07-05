#pragma once

#include <array>
#include <string_view>

namespace pongbot {

inline constexpr std::array<std::string_view, 16> kJointOrder = {
    "FL_HR_JOINT", "FL_HP_JOINT", "FL_KN_JOINT", "FL_WHEEL_JOINT",
    "FR_HR_JOINT", "FR_HP_JOINT", "FR_KN_JOINT", "FR_WHEEL_JOINT",
    "RL_HR_JOINT", "RL_HP_JOINT", "RL_KN_JOINT", "RL_WHEEL_JOINT",
    "RR_HR_JOINT", "RR_HP_JOINT", "RR_KN_JOINT", "RR_WHEEL_JOINT",
};

inline constexpr std::array<std::string_view, 12> kLegJointNames = {
    "FL_HR_JOINT", "FL_HP_JOINT", "FL_KN_JOINT",
    "FR_HR_JOINT", "FR_HP_JOINT", "FR_KN_JOINT",
    "RL_HR_JOINT", "RL_HP_JOINT", "RL_KN_JOINT",
    "RR_HR_JOINT", "RR_HP_JOINT", "RR_KN_JOINT",
};

inline constexpr std::array<std::string_view, 4> kWheelJointNames = {
    "FL_WHEEL_JOINT", "FR_WHEEL_JOINT", "RL_WHEEL_JOINT", "RR_WHEEL_JOINT",
};

inline constexpr std::array<std::string_view, 4> kWheelBodyNames = {
    "FL_WHEEL", "FR_WHEEL", "RL_WHEEL", "RR_WHEEL",
};

inline constexpr std::array<float, 16> kDefaultJointPos = {
    0.0f, 0.716f, -1.396f, 0.0f,
    0.0f, 0.716f, -1.396f, 0.0f,
    0.0f, 0.716f, -1.396f, 0.0f,
    0.0f, 0.716f, -1.396f, 0.0f,
};

inline constexpr std::array<float, 12> kLegKp = {
    200.0f, 200.0f, 200.0f,
    200.0f, 200.0f, 200.0f,
    200.0f, 200.0f, 200.0f,
    200.0f, 200.0f, 200.0f,
};

inline constexpr std::array<float, 12> kLegKd = {
    5.0f, 5.0f, 5.0f,
    5.0f, 5.0f, 5.0f,
    5.0f, 5.0f, 5.0f,
    5.0f, 5.0f, 5.0f,
};

inline constexpr std::array<float, 16> kTorqueLimits = {
    80.0f, 160.0f, 280.0f, 9.0f,
    80.0f, 160.0f, 280.0f, 9.0f,
    80.0f, 160.0f, 280.0f, 9.0f,
    80.0f, 160.0f, 280.0f, 9.0f,
};

inline constexpr float kLegActionScale = 0.5f;
inline constexpr float kWheelActionScale = 8.0f;
inline constexpr float kGaitPeriod = 0.72f;
inline constexpr float kWheelKv = 1.0f;
inline constexpr float kPolicyDt = 0.02f;

}  // namespace pongbot
