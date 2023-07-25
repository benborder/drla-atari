#pragma once

#include "configuration.h"

#include <ale_interface.hpp>
#include <drla/environment.h>

#include <vector>

namespace atari
{

class Atari final : public drla::Environment
{
public:
	Atari(const Config::AtariEnv& config);

	drla::EnvironmentConfiguration get_configuration() const override;

	drla::EnvStepData step(torch::Tensor action) override;
	drla::EnvStepData reset(const drla::State& initial_state) override;
	drla::Observations get_visualisations() const override;

	torch::Tensor expert_agent() override;

	std::unique_ptr<drla::Environment> clone() const override;

private:
	int single_step(ale::Action action);
	torch::Tensor get_observation();
	std::vector<int> get_legal_actions() const;

private:
	const Config::AtariEnv& config_;

	ale::ALEInterface ale_;
	ale::ActionVect action_set_;
	std::map<ale::Action, int> action_index_;

	EnvState state_;
	int step_ = 0;
	bool episode_end_;
	int max_episode_steps_ = 0;

	drla::Observations observations_;
	drla::Observations raw_observations_;
	std::vector<torch::Tensor> buffer_;
};

} // namespace atari
