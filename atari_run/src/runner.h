#pragma once

#include "atari_agent.h"
#include "atari_agent/configuration.h"

#include <drla/callback.h>

#include <deque>
#include <filesystem>
#include <vector>

struct EpisodeResult
{
	int id = 0;
	int env = 0;
	int length = 0;
	std::vector<int> life_length;
	std::vector<float> life_reward;

	torch::Tensor reward;
	torch::Tensor score;
	std::deque<drla::StepData> step_data;
};

class AtariRunner : public drla::AgentCallbackInterface
{
public:
	AtariRunner(atari::ConfigData config, const std::filesystem::path& path);

	void run(int env_count, int max_steps, bool save_gif);

private:
	void train_init(const drla::InitData& data) override;
	drla::AgentResetConfig env_reset(const drla::StepData& data) override;
	bool env_step(const drla::StepData& data) override;
	void train_update(const drla::TrainUpdateData& data) override;
	torch::Tensor interactive_step() override;

	void save(int steps, const std::filesystem::path& path) override;

	atari::ConfigData config_;
	std::filesystem::path data_path_;

	atari::AtariAgent atari_agent_;

	std::mutex m_step_;
	std::vector<EpisodeResult> current_episodes_;
	std::vector<EpisodeResult> episode_results_;
	int total_game_count_ = 0;
};
