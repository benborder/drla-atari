#pragma once

#include <math.h>

#include <limits>

template <typename T = double>
class Stats
{
public:
	T get_mean() const { return mean_; }

	T get_var() const { return var_; }

	T get_stdev() const { return std::sqrt(var_); }

	T get_max() const { return max_; }

	T get_min() const { return min_; }

	size_t get_count() const { return count_; }

	void set_ratio(double ratio) { ratio_ = ratio; }

	void update(const T& val)
	{
		if (count_ == 0)
		{
			mean_ = val;
			count_ = 1;
			max_ = val;
			min_ = val;
			return;
		}

		count_++;
		auto r = std::max(1.0 / (double)count_, ratio_);
		auto ir = (1.0 - r);
		auto new_mean = mean_ * ir + r * val;
		var_ = var_ * ir + r * (val - mean_) * (val - new_mean);
		mean_ = new_mean;
		max_ = std::max(max_, val);
		min_ = std::min(min_, val);
	}

private:
	T mean_ = 0;
	T var_ = 0;
	T max_ = -std::numeric_limits<T>::max();
	T min_ = std::numeric_limits<T>::max();
	double ratio_ = 0.01;
	size_t count_ = 0;
};
