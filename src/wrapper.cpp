#include <string>
#include <cmath>
#include <vector>
#include <map>
#include "wrapper.hpp"
#include "core.hpp"
#include "support.hpp"
#include "streamAligner.hpp"


namespace optonaut {

	StreamAligner state;
	Image *prev = NULL;

	void Push(double extrinsics[], double intrinsics[], char *image, int width, int height, double newExtrinsics[], int id) {
		Mat inputExtrinsics = Mat(3, 3, CV_64F, extrinsics);

		Image *current = new Image();
		current->img = Mat(height, width, CV_8UC4, image);
		From3DoubleTo4Double(inputExtrinsics, current->extrinsics);
		current->intrinsics = Mat(3, 3, CV_64F, intrinsics);
		current->id = id;
		current->source = "dynamic";

		state.Push(current);

		Mat e = state.GetCurrentRotation();
		for(int i = 0; i < 4; i++)
			for(int j = 0; j < 4; j++)
				newExtrinsics[i * 4 + j] = e.at<double>(i, j);

		//Only safe because we know what goes on inside state. 
		if(prev != NULL) {
			delete prev;
		}

		prev = current;
	}

	void Free() {
		if(prev != NULL) {
			delete prev;
			prev = NULL;
		}
	}
}