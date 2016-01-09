#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/stitching/detail/matchers.hpp>

using namespace cv;
using namespace cv::detail;
using namespace cv::xfeatures2d;
using namespace std; 

int main(int argc, char **argv) {

    // Does not work. 
    typedef FastFeatureDetector Detector;
    typedef BriefDescriptorExtractor Extractor;
    
    // Does work. 
    // typedef SURF Detector;
    // typedef SURF Extractor;

    assert(argc == 3);

    Mat a = imread(string(argv[1]));
    Mat b = imread(string(argv[2]));

    pyrDown(a, a);
    pyrDown(b, b);

    Ptr<Detector> detector = Detector::create();

    ImageFeatures aFeatures, bFeatures;

    detector->detect(a, aFeatures.keypoints);
    detector->detect(b, bFeatures.keypoints);

    Ptr<Extractor> extractor = Extractor::create();

    extractor->compute(a, aFeatures.keypoints, aFeatures.descriptors);
    extractor->compute(b, bFeatures.keypoints, bFeatures.descriptors);

    FlannBasedMatcher matcher(new flann::LshIndexParams(20, 10, 2));
    vector<vector<DMatch>> matches;

    matcher.knnMatch(aFeatures.descriptors, bFeatures.descriptors, matches, 2);

    if(matches.size() > 0)
        cout << "Woop woop it worked. :)" << endl;

    return 0;
}
