#include <opencv2/opencv.hpp>

namespace optonaut {
    double iPhone6IntrinsicsData[9] = {6.9034, 0, 3, 0, 6.9034, 1.6875, 0, 0, 1};
    double iPhone6LQIntrinsicsData[9] = {4.854369, 0, 3, 0, 4.854369, 2.4, 0, 0, 1};
    cv::Mat iPhone6Intrinsics(3, 3, CV_64F, iPhone6IntrinsicsData);
}
