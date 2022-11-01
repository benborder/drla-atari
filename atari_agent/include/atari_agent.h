#pragma once

#include "atari_agent/configuration.h"

#include <drla/agent.h>
#include <drla/callback.h>
#include <drla/environment.h>

#include <filesystem>
#include <memory>

namespace atari
{

class AtariAgent final : public drla::EnvironmentManager
{
public:
	AtariAgent(ConfigData&& config, drla::AgentCallbackInterface* callback, const std::filesystem::path& data_path = "");
	~AtariAgent();

	/// @brief Train the agent. Blocks until training finnished or stopped.
	void train();

	/// @brief Stop training the agent.
	void stop_train();

	/// @brief Run the agent, blocking until the max_steps reached or the environment terminates.
	/// @param env_count The number of environments to run
	/// @param options Options which change various behaviours of the agent. See RunOptions for more detail on available
	/// options.
	void run(int env_count, drla::RunOptions options = {});

protected:
	std::unique_ptr<drla::Environment> make_environment() override;
	drla::State get_initial_state() override;

private:
	const ConfigData config_;
	std::unique_ptr<drla::Agent> agent_;
};

} // namespace atari
