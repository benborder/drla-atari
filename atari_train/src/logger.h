#pragma once

#include "atari_agent/configuration.h"
#include "stats.h"

#include <drla/callback.h>
#include <tensorboard_logger.h>

#include <chrono>
#include <deque>
#include <filesystem>
#include <vector>

struct EpisodeResult
{
	int id = 0;
	int env = 0;
	int length = 0;
	std::vector<int> life_length;
	std::vector<std::vector<float>> life_reward;

	std::vector<torch::Tensor> reward;
	std::vector<torch::Tensor> score;
	std::deque<drla::StepData> step_data;

	// Indicates that this episode should be rendered
	bool render_final = false;
	bool render_gif = false;
	bool eval_episode = false;
};

class AtariTrainingLogger : public drla::AgentCallbackInterface
{
public:
	AtariTrainingLogger(atari::ConfigData config, const std::filesystem::path& path, bool resume);
	~AtariTrainingLogger();

private:
	void train_init(const drla::InitData& data) override;
	drla::AgentResetConfig env_reset(const drla::StepData& data) override;
	bool env_step(const drla::StepData& data) override;
	void train_update(const drla::TrainUpdateData& data) override;
	torch::Tensor interactive_step() override;

	void save(int steps, const std::filesystem::path& path) override;

	atari::ConfigData config_;

	TensorBoardLogger tb_logger_;

	std::mutex m_step_;

	std::vector<EpisodeResult> current_episodes_;
	std::vector<EpisodeResult> episode_results_;

	Stats<double> reward_stats_;
	Stats<double> score_stats_;
	Stats<double> eval_stats_;
	Stats<double> episode_length_stats_;
	Stats<double> life_length_stats_;
	Stats<double> fps_stats_;
	Stats<double> train_time_stats_;
	Stats<double> env_time_stats_;

	int timestep_ = 0;
	int total_episode_count_ = 0;
	int total_game_count_ = 0;
	int horizon_steps_ = 0;
	int total_timesteps_ = 0;
	int num_actors_ = 0;
	int actor_index_ = 0;
	int eval_period_ = 0;

	std::filesystem::path gif_path_;
	std::chrono::steady_clock::time_point start_time_;
};
