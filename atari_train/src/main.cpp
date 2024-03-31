#include "atari_agent.h"
#include "atari_agent/configuration.h"
#include "atari_agent/utility.h"
#include "logger.h"

#include <cxxopts.hpp>
#include <spdlog/spdlog.h>

#include <csignal>
#include <cstdio>
#include <filesystem>
#include <functional>

namespace
{
std::function<void(int)> shutdown_handler;

void signal_handler(int signum)
{
	shutdown_handler(signum);
}
} // namespace

// BUG: https://github.com/pytorch/pytorch/issues/49460
// This dummy function is a hack to fix an issue with loading pytorch models. It's unnecessary to invoke this function,
// just enforce library compiled
void dummy()
{
	std::regex regstr("Why");
	std::string s = "Why crashed";
	std::regex_search(s, regstr);
}

int main(int argc, char** argv)
{
	cxxopts::Options options(
		"Atari Train", "Trains an agent, periodically saving the model and a tensorboard event file.");
	options.add_options()(
		"c,config",
		"The config directory path or full file path. Relative paths use the data path as the base.",
		cxxopts::value<std::string>()->default_value(""))(
		"d,data", "The data path for saving/loading the model and training state", cxxopts::value<std::string>())(
		"h,help", "This printout", cxxopts::value<bool>()->default_value("false"));
	options.allow_unrecognised_options();
	auto result = options.parse(argc, argv);

	if (result["help"].as<bool>())
	{
		options.set_width(100);
		spdlog::fmt_lib::print("{}", options.help());
		return 0;
	}

	std::filesystem::path config_path = result["config"].as<std::string>();
	std::filesystem::path data_path = result["data"].as<std::string>();

	bool resume = true;
	if (config_path.empty())
	{
		config_path = data_path;
	}
	else if (config_path != data_path)
	{
		std::filesystem::create_directory(data_path);
		data_path = data_path / atari::utility::get_time();
		std::filesystem::create_directory(data_path);
		resume = false;
	}

	spdlog::set_level(spdlog::level::debug);
	spdlog::set_pattern("[%^%l%$] %v");

	auto config = atari::utility::load_config(config_path);

	AtariTrainingLogger logger(config, data_path, resume);
	atari::AtariAgent atari_agent(std::move(config), &logger, data_path);

	std::signal(SIGINT, ::signal_handler);
	shutdown_handler = [&]([[maybe_unused]] int signum) {
		spdlog::info("Stopping training...");
		atari_agent.stop_train();
		static int shutdown_attempt_count = 0;
		++shutdown_attempt_count;
		if (shutdown_attempt_count > 4)
		{
			std::abort();
		}
	};

	atari_agent.train();

	spdlog::info("Training finished!");

	return 0;
}
