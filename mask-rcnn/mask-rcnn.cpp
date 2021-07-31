// mask-rcnn.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include <ngraph/runtime/reference/roi_align.hpp>
#include <iostream>
#include <vector>

using namespace ngraph::runtime::reference;

int main()
{
	ngraph::Shape feature_maps_shape{ 1,1,50,50 };
	ngraph::Shape rois_shape{ 6,1 };
	ngraph::Shape batch_indices_shape{ 1,1,1,1 };
	ngraph::Shape out_shape{ 1,1,1,1 };

	int pooled_height = 7;
	int pooled_width = 7;
	int sampling_ratio = 1;
	float spatial_scale = 1.0f;
	const ROIPoolingMode pooling_mode{ ROIPoolingMode::MAX };

	std::vector<float> feature_maps(8*1024);
	std::vector<float> rois(8*1024);
	std::vector<int64_t> batch_indices(16*1024);
	std::vector<float> out(16*1024);
	
	roi_align<float>(feature_maps.data(),
		rois.data(),
		batch_indices.data(),
		out.data(),
		feature_maps_shape,
		rois_shape,
		batch_indices_shape,
		out_shape,
		pooled_height,
		pooled_width,
		sampling_ratio,
		spatial_scale,
		pooling_mode);
	
}

// Run program: Ctrl + F5 or Debug > Start Without Debugging menu
// Debug program: F5 or Debug > Start Debugging menu

// Tips for Getting Started: 
//   1. Use the Solution Explorer window to add/manage files
//   2. Use the Team Explorer window to connect to source control
//   3. Use the Output window to see build output and other messages
//   4. Use the Error List window to view errors
//   5. Go to Project > Add New Item to create new code files, or Project > Add Existing Item to add existing code files to the project
//   6. In the future, to open this project again, go to File > Open > Project and select the .sln file
