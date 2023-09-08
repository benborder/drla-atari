#include "logger.h"

#include "atari_agent/utility.h"

#include <spdlog/spdlog.h>

#include <filesystem>
#include <string>

using namespace atari;
using namespace drla;

AtariTrainingLogger::AtariTrainingLogger(atari::ConfigData config, const std::filesystem::path& path, bool resume)
		: config_(config), metrics_logger_(path, resume)
{
}

void AtariTrainingLogger::train_init(const drla::InitData& data)
{
	int env_count = 0;
	int start_timestep = 0;
	int total_timesteps = 0;

	std::visit(
		[&](auto& agent) {
			env_count = agent.env_count;
			std::visit(
				[&](auto& train_algorithm) {
					total_timesteps = train_algorithm.total_timesteps;
					start_timestep = train_algorithm.start_timestep;
				},
				agent.train_algorithm);
		},
		config_.agent);

	fmt::print("{:=<80}\n", "");
	fmt::print("Training Atari Agent\n");
	fmt::print("Train timesteps: {}\n", total_timesteps);
	fmt::print("Envs: {}\n", env_count);
	fmt::print("Start timestep: {}\n", start_timestep);
	fmt::print("{:=<80}\n", "");

	current_episodes_.resize(data.env_output.size());
	for (auto& ep : current_episodes_) { ep.id = total_episode_count_++; }

	metrics_logger_.init(total_timesteps);
}

drla::AgentResetConfig AtariTrainingLogger::env_reset(const drla::StepData& data)
{
	std::lock_guard lock(m_step_);
	EpisodeResult& episode_result = current_episodes_.at(data.env);
	episode_result.eval_episode = data.eval_mode;
	// in eval mode stop when reset as we only want a single episode
	auto stop = data.eval_mode && data.step > 0;
	return {stop, episode_result.render_gif};
}

bool AtariTrainingLogger::env_step(const drla::StepData& data)
{
	std::lock_guard lock(m_step_);
	EpisodeResult& episode_result = current_episodes_.at(data.env);

	if (episode_result.step_data.empty())
	{
		episode_result.reward = data.reward;
		episode_result.score = data.env_data.reward;
	}
	else
	{
		episode_result.reward += data.reward;
		episode_result.score += data.env_data.reward;
	}
	episode_result.step_data.push_back(data);

	if (data.env_data.state.episode_end)
	{
		bool game_over = true;
		if (config_.env.end_episode_on_life_loss)
		{
			game_over = std::any_cast<const EnvState&>(data.env_data.state.env_state).lives == 0 || data.eval_mode;
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
		if (game_over)
		{
			episode_result.env = data.env;
			episode_results_.push_back(std::move(episode_result));
			episode_result = {};
			episode_result.id = total_game_count_++;
			episode_result.render_final = episode_result.id % config_.observation_save_period == 0;
			episode_result.render_gif = episode_result.id % config_.observation_gif_save_period == 0;
		}
		if (!data.eval_mode)
		{
			++total_episode_count_;
		}
	}
	else
	{
		if (!episode_result.render_gif && !episode_result.eval_episode && episode_result.step_data.size() > 1)
		{
			episode_result.step_data.pop_front();
		}
		episode_result.length++;
	}

	return false;
}

void AtariTrainingLogger::train_update(const drla::TrainUpdateData& timestep_data)
{
	metrics_logger_.update(timestep_data);

	for (auto& episode_result : episode_results_)
	{
		if (episode_result.eval_episode)
		{
			metrics_logger_.add_scalar("environment", "reward_eval", episode_result.reward[0].item<float>());
			continue;
		}
		metrics_logger_.add_scalar("environment", "episode_length", double(episode_result.length));

		if (config_.env.end_episode_on_life_loss)
		{
			for (size_t i = 0; i < episode_result.life_length.size(); i++)
			{
				metrics_logger_.add_scalar("environment", "life_length", static_cast<float>(episode_result.life_length[i]));
				metrics_logger_.add_scalar("environment", "reward", episode_result.life_reward[i]);
			}
		}
		else
		{
			metrics_logger_.add_scalar("environment", "reward", episode_result.reward[0].item<float>());
		}

		metrics_logger_.add_scalar("environment", "score", episode_result.score.item<float>());

		if (episode_result.render_final)
		{
			metrics_logger_.add_image(
				"observations", "final_frame", episode_result.step_data.back().env_data.observation.front());
		}
		if (episode_result.render_gif)
		{
			std::vector<torch::Tensor> images;
			images.reserve(episode_result.step_data.size());
			for (auto& step_data : episode_result.step_data) { images.push_back(step_data.visualisation.front()); }
			metrics_logger_.add_animation("", "episode", images);
		}
	}

	metrics_logger_.print(timestep_data, total_episode_count_);

	episode_results_.clear();
}

torch::Tensor AtariTrainingLogger::interactive_step()
{
	return {};
}

void AtariTrainingLogger::save(int steps, const std::filesystem::path& path)
{
	auto config = config_;
	std::visit(
		[&](auto& agent) {
			std::visit([&](auto& train_algorithm) { train_algorithm.start_timestep = steps; }, agent.train_algorithm);
		},
		config_.agent);
	atari::utility::save_config(config, path);

	fmt::print("Configuration saved to: {}\n", path.string());
	fmt::print("{:-<80}\n", "");
}
