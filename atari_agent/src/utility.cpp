#include "utility.h"

#include "serialise.h"

#include <nlohmann/json.hpp>
#include <spdlog/spdlog.h>

#include <cstdio>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <thread>

using namespace atari;

ConfigData utility::load_config(const std::filesystem::path& config_path)
{
	ConfigData config;

	std::filesystem::path filename = "config.json";
	if (config_path.extension() == ".json" || config_path.extension() == ".jsonc")
	{
		filename = config_path;
	}
	else if (!config_path.empty() && !config_path.has_extension())
	{
		filename = config_path / filename;
		if (!std::filesystem::exists(filename))
		{
			filename += "c";
		}
	}
	else
	{
		spdlog::error("No configuration found at: {}", config_path.string());
		throw std::runtime_error("No valid configuration found!");
	}

	if (std::filesystem::exists(filename))
	{
		std::ifstream config_file(filename);
		nlohmann::json json = nlohmann::json::parse(config_file, nullptr, true, true);
		config = json.get<ConfigData>();
		spdlog::info("Configuration loaded from: {}", filename.string());
	}
	else
	{
		spdlog::error("No configuration found at: {}", filename.string());
		throw std::runtime_error("No configuration found!");
	}

	return config;
}

void utility::save_config(const ConfigData& config, const std::filesystem::path& config_path)
{
	std::ofstream config_file(config_path / "config.json");
	config_file << save_config(config);
	config_file.close();
}

std::string utility::save_config(const ConfigData& config)
{
	nlohmann::json json;
	to_json(json, config);
	return json.dump(2);
}

std::string utility::get_time()
{
	time_t rawtime = 0;
	time(&rawtime);
	struct tm* timeinfo = localtime(&rawtime);

	std::string buffer;
	buffer.resize(16);
	strftime(buffer.data(), buffer.size(), "%Y%m%dT%H%M%S", timeinfo);
	return buffer.c_str();
}
