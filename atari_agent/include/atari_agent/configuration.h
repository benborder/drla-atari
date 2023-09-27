#pragma once

#include <drla/configuration.h>

#include <array>
#include <cstddef>
#include <string>

namespace atari
{

namespace Config
{

struct AtariEnv
{
	// The location of the ROM file to load
	std::string rom_file;
	// End the episode when a life is lost, but don't reset the environment until lives is 0
	bool end_episode_on_life_loss = false;
	// Bin reward to {+1, 0, -1} by its sign.
	bool clip_reward = false;
	// Return only every n frames. Values <= 1 will return every frame.
	int frame_skip = 1;
	// The number of frames to perform noops for on reset
	int noop_reset_max_frames = 0;
	// The number of frames to stack and output as an observation. (0 and 1 output a single frame)
	int frame_stack = 1;
	// Uses grayscale observations
	bool grayscale = false;
	// The output resolution. A number < 0 implies using the original resolution.
	std::array<int, 2> output_resolution = {0, 0};
};

} // namespace Config

struct ConfigData
{
	// The atari environment configuration
	Config::AtariEnv env;

	// Configuration specific to the agent
	drla::Config::Agent agent;

	// Every n episodes save the final frame
	int observation_save_period = 1000;

	// Every n episodes save the entire episode as a gif
	int observation_gif_save_period = 10000;

	// Every n train timesteps log any images from metrics
	int metric_image_log_period = 1000;
};

struct EnvState
{
	int lives = 0;
};

} // namespace atari
