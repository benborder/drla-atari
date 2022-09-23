#pragma once

#include "atari_agent/configuration.h"

#include <drla/types.h>

#include <filesystem>
#include <string>

namespace atari
{

namespace utility
{

/// @brief Loads the configuration from the specified path and returns the config struct ConfigData
/// @param config_path The path to the directory which holds the configuration json or the filepath to the configuration
/// json
/// @return The configuration struct
ConfigData load_config(const std::filesystem::path& config_path = "");

/// @brief Saves the supplied configuration to the specified config_path in json format
/// @param config The configuration to save
/// @param config_path The directory path to save the configuration file to
void save_config(const ConfigData& config, const std::filesystem::path& config_path);

/// @brief Saves the supplied configuration to a string in json format
/// @param config The configuration to save
/// @return The serialised json string of the config
std::string save_config(const ConfigData& config);

/// @brief Returns a string identifying the current time, usable in file paths
/// @return A string representing the time in the format of YYYYMMDDTHHMMSS
std::string get_time();

} // namespace utility

} // namespace atari
