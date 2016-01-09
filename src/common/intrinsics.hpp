#include <opencv2/opencv.hpp>

namespace optonaut {
    double iPhone6IntrinsicsData[9] = {5.9066119, 0, 1.6875, 0, 5.9066119, 3, 0, 0, 1};
    double iPhone6LQIntrinsicsData[9] = {4.854369, 0, 2.4, 0, 4.854369, 3, 0, 0, 1};
    cv::Mat iPhone6Intrinsics(3, 3, CV_64F, iPhone6IntrinsicsData);

    double iPhone5IntrinsicsData[9] = {2.8091, 0, 1.2194, 0, 2.8091, 1.6364, 0, 0, 1};
    cv::Mat iPhone5Intrinsics(3, 3, CV_64F, iPhone5IntrinsicsData);
}
