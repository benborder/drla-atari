#include "runner.h"

#include "tensor_to_image.h"

#include <GifEncoder.h>
#include <fmt/chrono.h>
#include <fmt/color.h>
#include <fmt/core.h>
#include <spdlog/spdlog.h>

#include <filesystem>
#include <string>

using namespace atari;
using namespace drla;

AtariRunner::AtariRunner(atari::ConfigData config, const std::filesystem::path& path)
		: config_(config), data_path_(path), atari_agent_(std::move(config), this, path)
{
}

void AtariRunner::run(int env_count, int max_steps, bool save_gif)
{
	current_episodes_.resize(env_count);

	spdlog::info("Running {} environments\n", env_count);

	{
		drla::RunOptions options;
		options.capture_observations = save_gif;
		options.max_steps = max_steps;

		atari_agent_.run(env_count, options);
	}

	fmt::print("\n");
	spdlog::info("Complete!", env_count);

	TensorImage gif_dims;
	for (auto& episode_result : episode_results_)
	{
		if (config_.env.end_episode_on_life_loss)
		{
			for (size_t i = 0; i < episode_result.life_length.size(); i++)
			{
				spdlog::info("Life length: {}", episode_result.life_length[i]);
				spdlog::info("Life Reward: {}", episode_result.life_reward[i]);
			}
		}
		else
		{
			const float episode_reward = episode_result.reward[0].item<float>();
			spdlog::info("Episode length: {}", episode_result.length);
			spdlog::info("Episode Reward: {}", episode_reward);
		}

		spdlog::info("Score: {}", episode_result.score.item<float>());

		if (save_gif && episode_result.step_data.size() > 2)
		{
			GifEncoder gif_enc;
			auto sz = episode_result.step_data.front().raw_observation.front().sizes();
			int w = sz[1];
			int h = sz[0];
			int c = sz[2];
			auto gif_path = data_path_ / fmt::format(
																		 "capture_{:%Y-%m-%d}_score_{}_ep{}.gif",
																		 fmt::localtime(std::time(nullptr)),
																		 episode_result.score.item<float>(),
																		 episode_result.id);
			if (gif_enc.open(gif_path, w, h, 10, true, 0, w * h * c * c))
			{
				gif_dims.height = h;
				gif_dims.width = w;
				gif_dims.channels = c;
				for (auto& step_data : episode_result.step_data)
				{
					auto img = create_tensor_image(step_data.raw_observation.front());
					gif_enc.push(GifEncoder::PIXEL_FORMAT_RGB, img.data.data(), img.width, img.height, 2);
				}
				gif_enc.close();
			}
			else
			{
				spdlog::error("gif encoder error: could not create file '{}'\n", gif_path.string());
			}
		}
	}
}

void AtariRunner::train_init(const drla::InitData& data)
{
}

drla::AgentResetConfig AtariRunner::env_reset(const drla::StepData& data)
{
	std::lock_guard lock(m_step_);
	EpisodeResult& episode_result = current_episodes_[data.env];

	if (episode_result.length == 0 || std::any_cast<const EnvState&>(data.env_data.state.env_state).lives == 0)
	{
		// Clear the previous episode result for the env of this step data
		episode_result = {};
		episode_result.id = total_game_count_++;
		episode_result.reward = torch::zeros(data.reward.sizes());
		episode_result.score = torch::zeros(data.env_data.reward.sizes());
	}

	return {};
}

bool AtariRunner::env_step(const drla::StepData& data)
{
	std::lock_guard lock(m_step_);
	fmt::print("\rstep: ");
	for (auto& eps : current_episodes_) { fmt::print("{} ", eps.length); }

	EpisodeResult& episode_result = current_episodes_[data.env];

	episode_result.length++;
	episode_result.reward += data.reward;
	episode_result.score += data.env_data.reward;
	episode_result.step_data.push_back(data);

	if (data.env_data.state.episode_end)
	{
		bool game_over = true;
		if (config_.env.end_episode_on_life_loss)
		{
			game_over = std::any_cast<const EnvState&>(data.env_data.state.env_state).lives == 0;
			if (episode_result.life_length.empty())
			{
				episode_result.life_length.push_back(episode_result.length);
				episode_result.life_reward.push_back(episode_result.reward[0].item<float>());
			}
			else
			{
				episode_result.life_length.push_back(episode_result.length - episode_result.life_length.back());
				episode_result.life_reward.push_back(
					episode_result.reward[0].item<float>() - episode_result.life_reward.back());
			}
		}
		if (data.env_data.state.max_episode_steps > 0 && episode_result.length >= data.env_data.state.max_episode_steps)
		{
			game_over = true;
		}
		if (game_over)
		{
			episode_result.env = data.env;
			episode_results_.push_back(std::move(episode_result));
			return true;
		}
	}

	return false;
}

void AtariRunner::train_update(const drla::TrainUpdateData& timestep_data)
{
}

torch::Tensor AtariRunner::interactive_step()
{
	return {};
}

void AtariRunner::save(int steps, const std::filesystem::path& path)
{
}
