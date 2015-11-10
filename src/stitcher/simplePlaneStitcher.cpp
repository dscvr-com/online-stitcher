
#include <vector>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/stitching.hpp>

#include "simplePlaneStitcher.hpp"

using namespace std;
using namespace cv;
using namespace cv::detail;

namespace optonaut {

StitchingResultP SimplePlaneStitcher::Stitch(const std::vector<ImageP> &in, const std::vector<cv::Point> &corners) const {
	size_t n = in.size();
    assert(n > 0);

    vector<Mat> images(n);
    vector<Size> sizes(n);
    for(size_t i = 0; i < n; i++) {
        images[i] = in[i]->data;
        sizes[i] = in[i]->size();
    }

	//Create masks and small images for fast stitching. 
    vector<Mat> masks(n);

    for(size_t i = 0; i < n; i++) {
        cout << "Creare mask " << i << " size: " << images[i].size() << endl;
        masks[i] = Mat(sizes[i], CV_8U);
        masks[i].setTo(Scalar::all(255));
    }

	Ptr<Blender> blender;
	blender = Blender::createDefault(Blender::FEATHER, true);
    blender->prepare(corners, sizes);

    Mat imageAsShort;

	for (size_t i = 0; i < n; i++)
	{
        cout << "Bleinging image " << i << endl;
        images[i].convertTo(imageAsShort, CV_16S);
		blender->feed(imageAsShort, masks[i], corners[i]);
	}

	StitchingResultP res(new StitchingResult());
    Mat image;
    Mat mask;
	blender->blend(image, mask);

    res->image = Image(image);
    res->mask = Image(mask);

    res->corner.x = corners[0].x;
    res->corner.y = corners[0].y;

    for(size_t i = 1; i < n; i++) {
        res->corner.x = min(res->corner.x, corners[i].x);
        res->corner.y = min(res->corner.y, corners[i].y);
    }

	return res;
}
}
