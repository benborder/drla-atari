#pragma once

#include <torch/torch.h>

#include <string>

struct TensorImage
{
	std::vector<unsigned char> data;
	int height;
	int width;
	int channels;
};

inline TensorImage create_tensor_image(const torch::Tensor& tensor_image, bool invert = false)
{
	if (tensor_image.numel() == 0)
	{
		return {};
	}

	torch::Tensor rgb_tensor;
	if (tensor_image.dim() == 3)
	{
		if (tensor_image.size(0) == 1)
		{
			rgb_tensor =
					torch::cat({tensor_image[0].unsqueeze(2), tensor_image[0].unsqueeze(2), tensor_image[0].unsqueeze(2)}, -1)
							.detach();
		}
		else if (tensor_image.size(0) == 2)
		{
			rgb_tensor = torch::cat(
											 {tensor_image[0].unsqueeze(2),
												tensor_image[1].unsqueeze(2),
												torch::zeros(tensor_image[0].sizes()).unsqueeze(2)},
											 -1)
											 .detach();
		}
		else if (tensor_image.size(0) == 3)
		{
			rgb_tensor =
					torch::cat({tensor_image[0].unsqueeze(2), tensor_image[1].unsqueeze(2), tensor_image[2].unsqueeze(2)}, -1)
							.detach();
		}
		else
		{
			rgb_tensor = tensor_image;
		}
	}
	else if (tensor_image.dim() == 2)
	{
		rgb_tensor =
				torch::cat({tensor_image.unsqueeze(2), tensor_image.unsqueeze(2), tensor_image.unsqueeze(2)}, -1).detach();
	}

	assert(rgb_tensor.is_contiguous());

	if (rgb_tensor.is_floating_point())
	{
		rgb_tensor = rgb_tensor.clamp(0.0, 1.0).mul_(std::numeric_limits<unsigned char>::max()).to(torch::kUInt8);
	}

	// Invert to make it easier to see
	if (invert)
	{
		rgb_tensor = std::numeric_limits<unsigned char>::max() - rgb_tensor;
	}

	auto ptr = rgb_tensor.data_ptr<unsigned char>();
	return {
			std::vector<unsigned char>(ptr, ptr + (size_t)rgb_tensor.numel()),
			static_cast<int>(rgb_tensor.size(0)),
			static_cast<int>(rgb_tensor.size(1)),
			static_cast<int>(rgb_tensor.size(2))};
}
