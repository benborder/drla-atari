#pragma once

#include "configuration.h"

#include <drla/configuration.h>
#include <drla/configuration/serialise_json.h>
#include <drla/types.h>
#include <nlohmann/json.hpp>
#include <spdlog/spdlog.h>

#include <iostream>

namespace atari
{

namespace Config
{

static inline void from_json(const nlohmann::json& json, Config::AtariEnv& env)
{
	env.rom_file << required_input{json, "rom_file"};
	env.end_episode_on_life_loss << optional_input{json, "end_episode_on_life_loss"};
	env.clip_reward << optional_input{json, "clip_reward"};
	env.frame_skip << optional_input{json, "frame_skip"};
	env.noop_reset_max_frames << optional_input{json, "noop_reset_max_frames"};
	env.frame_stack << optional_input{json, "frame_stack"};
	env.frame_stack = std::max(env.frame_stack, 1);
	env.grayscale << optional_input{json, "grayscale"};
	env.output_resolution << optional_input{json, "output_resolution"};
}
static inline void to_json(nlohmann::json& json, const Config::AtariEnv& env)
{
	json["rom_file"] = env.rom_file;
	json["end_episode_on_life_loss"] = env.end_episode_on_life_loss;
	json["clip_reward"] = env.clip_reward;
	json["frame_skip"] = env.frame_skip;
	json["noop_reset_max_frames"] = env.noop_reset_max_frames;
	json["frame_stack"] = env.frame_stack;
	json["grayscale"] = env.grayscale;
	json["output_resolution"] = env.output_resolution;
}

} // namespace Config

static inline void from_json(const nlohmann::json& json, ConfigData& config)
{
	config.env << required_input{json, "environment"};
	config.agent << required_input{json, "agent"};
	config.observation_save_period << optional_input{json, "observation_save_period"};
	config.observation_gif_save_period << optional_input{json, "observation_gif_save_period"};
}

static inline void to_json(nlohmann::json& json, const ConfigData& config)
{
	json["environment"] = config.env;
	json["agent"] = config.agent;
	json["observation_save_period"] = config.observation_save_period;
	json["observation_gif_save_period"] = config.observation_gif_save_period;
}

static inline void from_json(const nlohmann::json& json, EnvState& state)
{
}

static inline void to_json(nlohmann::json& json, const EnvState& state)
{
}

} // namespace atari

namespace drla
{

static inline void from_json(const nlohmann::json& json, State& state)
{
	state.episode_end << optional_input{json, "episode_end"};
	state.max_episode_steps << optional_input{json, "max_episode_steps"};
	state.step << optional_input{json, "step"};
	atari::EnvState env_state;
	env_state << optional_input{json, "env_state"};
	state.env_state = env_state;
}

static inline void to_json(nlohmann::json& json, const State& state)
{
	json["episode_end"] = state.episode_end;
	json["max_episode_steps"] = state.max_episode_steps;
	json["step"] = state.step;
	json["env_state"] = std::any_cast<atari::EnvState>(state.env_state);
}

} // namespace drla
