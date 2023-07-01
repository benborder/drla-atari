#include "logger.h"

#include "atari_agent/utility.h"
#include "tensor_to_image.h"

#include <GifEncoder.h>
#include <fmt/chrono.h>
#include <fmt/color.h>
#include <fmt/core.h>
#include <lodepng.h>

#include <filesystem>
#include <string>

using namespace atari;
using namespace drla;

AtariTrainingLogger::AtariTrainingLogger(atari::ConfigData config, const std::filesystem::path& path, bool resume)
		: config_(config), tb_logger_(path.c_str(), TensorBoardLoggerOptions{}.resume(resume)), gif_path_()
{
	auto tmp_dir = std::filesystem::temp_directory_path();
	std::filesystem::create_directory(tmp_dir);
	gif_path_ = tmp_dir / "episode.gif";
	if (std::filesystem::exists(gif_path_))
	{
		std::filesystem::remove(gif_path_);
	}
}

AtariTrainingLogger::~AtariTrainingLogger()
{
	google::protobuf::ShutdownProtobufLibrary();
}

void AtariTrainingLogger::train_init(const drla::InitData& data)
{
	int env_count = 0;
	int start_timestep = 0;

	std::visit(
		[&](auto& agent) {
			env_count = agent.env_count;
			eval_period_ = agent.eval_period;
			std::visit(
				[&](auto& train_algorithm) {
					total_timesteps_ = train_algorithm.total_timesteps;
					start_timestep = train_algorithm.start_timestep;
				},
				agent.train_algorithm);
		},
		config_.agent);

	num_actors_ = data.env_config.num_actors;

	fmt::print("{:=<80}\n", "");
	fmt::print("Training Atari Agent\n");
	fmt::print("Train timesteps: {}\n", total_timesteps_);
	fmt::print("Envs: {}\n", env_count);
	if (num_actors_ > 1)
	{
		fmt::print("Actors: {}\n", num_actors_);
	}
	fmt::print("Start timestep: {}\n", start_timestep);
	fmt::print("{:=<80}\n", "");

	current_episodes_.resize(data.env_output.size());
	for (auto& ep : current_episodes_) { ep.id = total_episode_count_++; }

	start_time_ = std::chrono::steady_clock::now();
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
		episode_result.reward.resize(num_actors_);
		episode_result.score.resize(num_actors_);
		episode_result.life_reward.resize(num_actors_);
		for (int i = 0; i < num_actors_; ++i)
		{
			episode_result.reward[i] = torch::zeros(data.reward.sizes());
			episode_result.score[i] = torch::zeros(data.env_data.reward.sizes());
		}

		episode_result.reward.at(actor_index_) += data.reward;
		episode_result.score.at(actor_index_) += data.env_data.reward;
	}
	else
	{
		auto turn_index = episode_result.step_data.back().env_data.turn_index;
		episode_result.reward.at(turn_index) += data.reward;
		episode_result.score.at(turn_index) += data.env_data.reward;
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
				for (size_t i = 0; i < episode_result.reward.size(); ++i)
				{
					episode_result.life_reward.at(i).push_back(episode_result.reward.at(i)[0].item<float>());
				}
			}
			else
			{
				episode_result.life_length.push_back(episode_result.length - episode_result.life_length.back());
				for (size_t i; i < episode_result.reward.size(); ++i)
				{
					episode_result.life_reward.at(i).push_back(
						episode_result.reward.at(i)[0].item<float>() - episode_result.life_reward.at(i).back());
				}
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
		total_episode_count_++;
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
	std::vector<TensorImage> observation_images;

	TensorImage gif_dims;
	for (auto& episode_result : episode_results_)
	{
		if (episode_result.eval_episode)
		{
			float episode_reward = episode_result.reward.at(0)[0].item<float>();
			for (size_t i = 1; i < episode_result.reward.size(); ++i)
			{
				episode_reward -= episode_result.reward[i][0].item<float>();
			}
			eval_stats_.update(episode_reward);
			tb_logger_.add_scalar("environment/reward_eval", timestep_data.timestep, episode_reward);
			continue;
		}
		episode_length_stats_.update(episode_result.length);
		tb_logger_.add_scalar("environment/episode_length", timestep_data.timestep, double(episode_result.length));

		if (config_.env.end_episode_on_life_loss)
		{
			for (size_t i = 0; i < episode_result.life_length.size(); i++)
			{
				const float life_length = static_cast<float>(episode_result.life_length[i]);
				life_length_stats_.update(life_length);
				tb_logger_.add_scalar("environment/life_length", timestep_data.timestep, life_length);

				if (num_actors_ > 1)
				{
					for (size_t actor; actor < episode_result.reward.size(); ++actor)
					{
						const float life_reward = episode_result.life_reward.at(actor)[i];
						if (static_cast<int>(actor) == actor_index_)
						{
							reward_stats_.update(life_reward);
						}
						tb_logger_.add_scalar(
							"environment/reward_actor" + std::to_string(actor), timestep_data.timestep, life_reward);
					}
				}
				else
				{
					const float life_reward = episode_result.life_reward.back()[i];
					reward_stats_.update(life_reward);
					tb_logger_.add_scalar("environment/reward", timestep_data.timestep, life_reward);
				}
			}
		}
		else
		{
			if (num_actors_ > 1)
			{
				for (size_t actor; actor < episode_result.reward.size(); ++actor)
				{
					const float episode_reward = episode_result.reward.at(actor)[0].item<float>();
					if (static_cast<int>(actor) == actor_index_)
					{
						reward_stats_.update(episode_reward);
					}
					tb_logger_.add_scalar(
						"environment/reward_actor" + std::to_string(actor), timestep_data.timestep, episode_reward);
				}
			}
			else
			{
				const float episode_reward = episode_result.reward.back()[0].item<float>();
				reward_stats_.update(episode_reward);
				tb_logger_.add_scalar("environment/reward", timestep_data.timestep, episode_reward);
			}
		}

		if (num_actors_ > 1)
		{
			for (size_t actor; actor < episode_result.reward.size(); ++actor)
			{
				const float score = episode_result.score.at(actor).item<float>();
				if (static_cast<int>(actor) == actor_index_)
				{
					score_stats_.update(score);
				}
				tb_logger_.add_scalar("environment/score_actor" + std::to_string(actor), timestep_data.timestep, score);
			}
		}
		else
		{
			const float score = episode_result.score.back().item<float>();
			score_stats_.update(score);
			tb_logger_.add_scalar("environment/score", timestep_data.timestep, score);
		}

		if (episode_result.render_final)
		{
			const auto& step = episode_result.step_data.back();
			auto channels = std::min<int>(step.env_data.observation.front().size(0), 3);
			observation_images.push_back(create_tensor_image(step.env_data.observation.front().narrow(0, 0, channels)));
		}
		if (episode_result.render_gif)
		{
			GifEncoder gif_enc;
			auto sz = episode_result.step_data.front().raw_observation.front().sizes();
			int w = sz[1];
			int h = sz[0];
			int c = sz[2];
			if (gif_enc.open(gif_path_, w, h, 10, true, 0, w * h * c * c))
			{
				gif_dims.height = h;
				gif_dims.width = w;
				gif_dims.channels = c;
				for (auto& step_data : episode_result.step_data)
				{
					auto img = create_tensor_image(step_data.raw_observation.front());
					gif_enc.push(GifEncoder::PIXEL_FORMAT_RGB, img.data.data(), img.width, img.height, 2);
				}
				gif_enc.close();
			}
			else
			{
				fmt::print("gif encoder error: could not create file '{}'\n", gif_path_.string());
			}
		}
	}

	if (!observation_images.empty())
	{
		for (const auto& obs_img : observation_images)
		{
			std::vector<unsigned char> png;
			unsigned error = lodepng::encode(png, obs_img.data, obs_img.width, obs_img.height, LCT_RGB);
			if (error != 0)
			{
				fmt::print("png encoder error {}: {}", error, lodepng_error_text(error));
				continue;
			}
			std::string img(std::make_move_iterator(png.begin()), std::make_move_iterator(png.end()));
			tb_logger_.add_image(
				"observations/final_frame", timestep_data.timestep, img, obs_img.height, obs_img.width, obs_img.channels);
		}
	}

	if (std::filesystem::exists(gif_path_))
	{
		std::ifstream fin(gif_path_, std::ios::binary);
		std::ostringstream ss;
		ss << fin.rdbuf();
		std::string img = ss.str();
		ss.str("");
		fin.close();
		tb_logger_.add_image("episode", timestep_data.timestep, img, gif_dims.height, gif_dims.width, gif_dims.channels);
		std::filesystem::remove(gif_path_);
	}

	size_t max_len = 12;
	for (const auto& data : timestep_data.update_data)
	{
		std::string name = drla::get_result_type_name(data.type);
		std::string tag = "train/" + name;
		tb_logger_.add_scalar(tag, timestep_data.timestep, data.value);
		max_len = std::max(max_len, name.size());
	}

	max_len += 1;

	fps_stats_.update(timestep_data.fps);
	// Only update the stats if its greater than 0
	const auto train_time = std::chrono::duration_cast<std::chrono::milliseconds>(timestep_data.update_duration).count();
	if (train_time > 0)
	{
		train_time_stats_.update(train_time);
	}
	env_time_stats_.update(std::chrono::duration_cast<std::chrono::milliseconds>(timestep_data.env_duration).count());

	double progress = 100 * static_cast<double>(timestep_data.timestep + 1) / static_cast<double>(total_timesteps_);

	fmt::print("{:<{}}| {:g} [{:g}]\n", "env_fps", max_len, fps_stats_.get_mean(), timestep_data.fps_env);
	fmt::print("{:<{}}| {:g} ms\n", "env_time", max_len, env_time_stats_.get_mean());
	fmt::print("{:<{}}| {:g} ms\n", "train_time", max_len, train_time_stats_.get_mean());
	fmt::print("{:<{}}| {:%H:%M:%S}\n", "elapsed_time", max_len, std::chrono::steady_clock::now() - start_time_);
	fmt::print(
		"{:<{}}| {} / {} [{:.2g}%]\n", "timesteps", max_len, timestep_data.timestep + 1, total_timesteps_, progress);
	fmt::print("{:<{}}| {}\n", "episodes", max_len, total_episode_count_);
	fmt::print("{:<{}}| {}\n", "global_steps", max_len, timestep_data.global_steps);
	for (const auto& data : timestep_data.update_data)
	{
		fmt::print("{:<{}}| {}\n", drla::get_result_type_name(data.type), max_len, data.value);
	}
	fmt::print(
		fmt::bg(fmt::detail::color_type(fmt::rgb(50, 50, 50))) | fmt::emphasis::bold,
		"{:<15}|{:^15}|{:^15}|{:^15}|{:^15}|",
		"",
		"mean",
		"stdev",
		"max",
		"min");
	fmt::print("\n");
	std::string stats_fmt = "{:<15}|{:>14g} |{:>14g} |{:>14g} |{:>14g} |\n";
	fmt::print(
		stats_fmt,
		"score",
		score_stats_.get_mean(),
		score_stats_.get_stdev(),
		score_stats_.get_max(),
		score_stats_.get_min());
	fmt::print(
		stats_fmt,
		"reward",
		reward_stats_.get_mean(),
		reward_stats_.get_stdev(),
		reward_stats_.get_max(),
		reward_stats_.get_min());
	if (eval_period_ > 0)
	{
		fmt::print(
			stats_fmt,
			"eval reward",
			eval_stats_.get_mean(),
			eval_stats_.get_stdev(),
			eval_stats_.get_max(),
			eval_stats_.get_min());
	}
	fmt::print(
		stats_fmt,
		"episode length",
		episode_length_stats_.get_mean(),
		episode_length_stats_.get_stdev(),
		episode_length_stats_.get_max(),
		episode_length_stats_.get_min());
	if (config_.env.end_episode_on_life_loss)
	{
		fmt::print(
			stats_fmt,
			"life length",
			life_length_stats_.get_mean(),
			life_length_stats_.get_stdev(),
			life_length_stats_.get_max(),
			life_length_stats_.get_min());
	}
	fmt::print("{:=<80}\n", "");

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
