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
	Atari(const Config::AtariEnv& config, const torch::Device& device);

	drla::EnvironmentConfiguration get_configuration() const override;

	drla::StepResult step(torch::Tensor action) override;
	drla::StepResult reset(const drla::State& initial_state) override;
	drla::Observations get_raw_observations() const override;
	void set_state(const drla::State& state) override;

private:
	int single_step(ale::Action action);
	torch::Tensor get_observation();

private:
	const Config::AtariEnv& config_;

	ale::ALEInterface ale_;
	ale::ActionVect action_set_;

	EnvState state_;
	int step_ = 0;
	bool episode_end_;
	int max_episode_steps_ = 0;

	drla::Observations observations_;
	drla::Observations raw_observations_;
	std::vector<torch::Tensor> buffer_;
};

} // namespace atari
