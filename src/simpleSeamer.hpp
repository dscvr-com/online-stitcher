
#include <opencv2/core.hpp>
#include <opencv2/opencv_modules.hpp>
#include <opencv2/stitching.hpp>
#include <vector>

#ifndef OPTONAUT_SIMPLE_SEAM_HEADER
#define OPTONAUT_SIMPLE_SEAM_HEADER

using namespace std;
using namespace cv;
using namespace cv::detail;

class SimpleSeamer : public PairwiseSeamFinder
{
public:
    virtual void find(const std::vector<UMat> &src, const std::vector<Point> &corners,
                      std::vector<UMat> &masks);
    virtual void find(const std::vector<Size> &size, const std::vector<Point> &corners,
                      std::vector<UMat> &masks);
private:
    void findInPair(size_t first, size_t second, Rect roi);
};


void SimpleSeamer::find(const std::vector<UMat> &src, const std::vector<Point> &corners,
                             std::vector<UMat> &masks)
{
    PairwiseSeamFinder::find(src, corners, masks);
}

void SimpleSeamer::find(const std::vector<Size> &sizes, const std::vector<Point> &corners,
                             std::vector<UMat> &masks)
{
    if (sizes.size() == 0)
        return;

    sizes_ = sizes;
    corners_ = corners;
    masks_ = masks;
    run();
}


void SimpleSeamer::findInPair(size_t first, size_t second, Rect roi)
{
    Size imgA = sizes_[first];
    Size imgB = sizes_[second];
    Mat maskA = masks_[first].getMat(ACCESS_WRITE);
    Mat maskB = masks_[second].getMat(ACCESS_WRITE);
    Point tlA = corners_[first];
    Point tlB = corners_[second];

    assert(maskA.type() == CV_8U);
    assert(maskB.type() == CV_8U);

    assert(imgA.width == maskA.cols);
    assert(imgB.width == maskB.cols);
    assert(imgA.height == maskA.rows);
    assert(imgB.height == maskB.rows);


    //All coordinates in ROI center space

    //Center of first image in ROI space
    int aToRoiX = tlA.x - roi.x - roi.width / 2;
    int aToRoiY = tlA.y - roi.y - roi.height / 2;
    int ax = imgA.width / 2 + aToRoiX;
    int ay = imgA.height / 2 + aToRoiY;

    //Center of second image in ROI space
    int bToRoiX = tlB.x - roi.x - roi.width / 2;
    int bToRoiY = tlB.y - roi.y - roi.height / 2;
    int bx = imgB.width / 2 + bToRoiX;
    int by = imgB.height / 2 + bToRoiY;

    //Derivation of the line orthogonal to the line connecting a and b
    double dy = (ay - by);
    double dx = (ax - bx);
    double l = sqrt(dy * dy + ay * ay);
    dy = dx / l;
    dx = dy / l;

    bool greaterZero = (ax * dy) - (ay * dx) > 0;

    for(int x = 0; x < imgA.width; x++) 
    {
        for(int y = 0; y < imgA.height; y++) 
        {
            int xA = x + aToRoiX;
            int yA = y + aToRoiY;

            if(greaterZero) {
                if(xA * dy - yA * dx < 0 && maskB.at<uchar>(xA - bToRoiX, yA - bToRoiY) != 0) {
                    maskA.at<uchar>(y, x) = 0;
                }
            } else {
                if(xA * dy - yA * dx > 0 && maskB.at<uchar>(xA - bToRoiX, yA - bToRoiY) != 0) {
                    maskA.at<uchar>(y, x) = 0;
                }
            }
        }       
    }

    greaterZero = (bx * dy) - (by * dx) > 0;

    for(int x = 0; x < imgB.width; x++) 
    {
        for(int y = 0; y < imgB.height; y++) 
        {
            int xB = x + bToRoiX;
            int yB = y + bToRoiY;

            if(greaterZero) {
                if(xB * dy - yB * dx < 0 && maskA.at<uchar>(xB - aToRoiX, yB - aToRoiY) != 0) {
                    maskB.at<uchar>(y, x) = 0;
                }
            } else {
                if(xB * dy - yB * dx > 0 && maskA.at<uchar>(xB - aToRoiX, yB - aToRoiY) != 0) {
                    maskB.at<uchar>(y, x) = 0;
                }
            }
        }       
    }
}

#endif
