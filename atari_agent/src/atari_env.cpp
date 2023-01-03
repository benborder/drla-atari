#include "atari_env.h"

#include <algorithm>
#include <filesystem>

using namespace atari;

Atari::Atari(const Config::AtariEnv& config) : config_(config)
{
	// Load the ROM file. (Also resets the system for new settings to take effect.)
	ale_.loadROM(std::filesystem::current_path() / config_.rom_file);

	// Get the vector of minimal actions
	action_set_ = ale_.getMinimalActionSet();

	observations_.resize(1);
}

int Atari::single_step(ale::Action action)
{
	auto reward = ale_.act(action);

	int lives = ale_.lives();
	if (config_.end_episode_on_life_loss)
	{
		episode_end_ |= lives < state_.lives;
	}
	state_.lives = lives;
	episode_end_ |= ale_.game_over();

	return reward;
}

drla::EnvStepData Atari::step(torch::Tensor action)
{
	ale::Action a = action_set_[action[0].item<int>()];
	torch::Tensor reward = torch::zeros(1);

	if (config_.frame_skip > 1)
	{
		reward[0] = 0.0F;
		std::vector<torch::Tensor> max_buffer;
		max_buffer.resize(2);
		for (int f = 0; f < config_.frame_skip; f++)
		{
			reward[0] += single_step(a);

			if (f == config_.frame_skip - 2)
			{
				max_buffer[0] = get_observation();
			}
			else if (f == config_.frame_skip - 1)
			{
				max_buffer[1] = get_observation();
			}
		}
		buffer_.push_back(max_buffer[0].max(max_buffer[1]));
	}
	else
	{
		reward[0] = single_step(a);
		buffer_.push_back(get_observation());
	}

	if (static_cast<int>(buffer_.size()) > config_.frame_stack)
	{
		buffer_.erase(buffer_.begin());
	}
	observations_[0] = torch::cat(buffer_);

	if (config_.clip_reward)
	{
		reward[0].sign_();
	}

	++step_;
	if (max_episode_steps_ > 0 && step_ > max_episode_steps_)
	{
		episode_end_ = true;
	}

	return {observations_, reward, {std::make_any<EnvState>(state_), step_, episode_end_, max_episode_steps_}};
}

// This is performed after a step but before the next step
drla::EnvStepData Atari::reset(const drla::State& initial_state)
{
	step_ = 0;
	episode_end_ = false;
	max_episode_steps_ = initial_state.max_episode_steps;
	if (config_.end_episode_on_life_loss && state_.lives > 0)
	{
		return {observations_, torch::zeros(1), {std::make_any<EnvState>(state_), step_, episode_end_, max_episode_steps_}};
	}

	ale_.reset_game();

	state_.lives = ale_.lives();

	buffer_.clear();
	for (int f = config_.noop_reset_max_frames; f > 0; f--)
	{
		if (f < config_.frame_stack)
		{
			buffer_.push_back(get_observation());
		}
		ale_.act(ale::PLAYER_A_NOOP);
	}

	while (static_cast<int>(buffer_.size()) < config_.frame_stack) { buffer_.push_back(get_observation()); }

	observations_[0] = torch::cat(buffer_);

	return {observations_, torch::zeros(1), {std::make_any<EnvState>(state_), step_, episode_end_}};
}

drla::Observations Atari::get_raw_observations() const
{
	std::vector<unsigned char> output_buffer;
	const auto& screen = ale_.getScreen();
	ale_.getScreenRGB(output_buffer);
	return {torch::from_blob(output_buffer.data(), {int(screen.height()), int(screen.width()), 3}, torch::kByte).clone()};
}

drla::EnvironmentConfiguration Atari::get_configuration() const
{
	drla::EnvironmentConfiguration config;
	config.name = config_.rom_file;
	const auto& screen = ale_.getScreen();
	int width = config_.output_resolution[0] > 0 ? config_.output_resolution[0] : screen.width();
	int height = config_.output_resolution[1] > 0 ? config_.output_resolution[1] : screen.height();
	config.observation_shapes.push_back({{config_.frame_stack, height, width}});
	config.observation_dtypes.push_back(torch::kFloat);
	config.action_space = {drla::ActionSpaceType::kDiscrete, {static_cast<int>(action_set_.size())}};
	config.reward_types = {"score"};
	return config;
}

torch::Tensor Atari::get_observation()
{
	std::vector<unsigned char> output_buffer;
	const auto& screen = ale_.getScreen();
	torch::Tensor raw_frame;
	int channels;
	if (config_.grayscale)
	{
		channels = 1;
		ale_.getScreenGrayscale(output_buffer);
	}
	else
	{
		channels = 3;
		ale_.getScreenRGB(output_buffer);
	}
	raw_frame =
		torch::from_blob(output_buffer.data(), {int(screen.height()), int(screen.width()), channels}, torch::kByte);

	torch::Tensor obs = raw_frame.permute({2, 0, 1}).to(torch::kFloat).div(255.0F);
	if (config_.output_resolution[0] > 0 || config_.output_resolution[1] > 0)
	{
		int width = config_.output_resolution[0] > 0 ? config_.output_resolution[0] : screen.width();
		int height = config_.output_resolution[1] > 0 ? config_.output_resolution[1] : screen.height();
		obs = torch::nn::functional::interpolate(
						obs.view({1, channels, int(screen.height()), int(screen.width())}),
						torch::nn::functional::InterpolateFuncOptions()
							.size(torch::make_optional<std::vector<int64_t>>({height, width}))
							.mode(torch::kArea))
						.view({channels, height, width});
	}
	return obs;
}
