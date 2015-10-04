#include <opencv2/opencv.hpp>

namespace optonaut {
    double iPhone6IntrinsicsData[9] = {5.9266119, 0, 1.6875, 0, 5.9266119, 3, 0, 0, 1};
    double iPhone6LQIntrinsicsData[9] = {4.854369, 0, 2.4, 0, 4.854369, 3, 0, 0, 1};
    cv::Mat iPhone6Intrinsics(3, 3, CV_64F, iPhone6IntrinsicsData);
}
