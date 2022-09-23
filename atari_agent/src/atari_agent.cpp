#include "atari_agent.h"

#include "atari_env.h"

#include <drla/interactive_agent.h>

#include <iostream>

using namespace atari;

AtariAgent::AtariAgent(
		ConfigData&& config, drla::AgentCallbackInterface* callback, const std::filesystem::path& data_path)
		: config_(std::move(config)), agent_(drla::make_agent(config_.agent, this, callback, data_path))
{
}

AtariAgent::~AtariAgent()
{
}

void AtariAgent::train()
{
	agent_->train();
}

void AtariAgent::stop_train()
{
	agent_->stop_train();
}

void AtariAgent::run(int env_count, drla::RunOptions options)
{
	std::vector<drla::State> initial_states;
	initial_states.resize(env_count);
	for (auto& state : initial_states)
	{
		state.max_episode_steps = options.max_steps;
	}
	return agent_->run(initial_states, std::move(options));
}

std::unique_ptr<drla::Environment> AtariAgent::make_environment(torch::Device device)
{
	return std::make_unique<Atari>(config_.env, device);
}

drla::State AtariAgent::get_initial_state()
{
	drla::State state;
	state.env_state = std::make_any<EnvState>();
	return state;
}
