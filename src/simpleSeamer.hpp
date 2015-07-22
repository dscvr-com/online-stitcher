
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
    Size img1 = sizes_[first], img2 = sizes_[second];
    Mat mask1 = masks_[first].getMat(ACCESS_READ), mask2 = masks_[second].getMat(ACCESS_READ);
    Point tl1 = corners_[first], tl2 = corners_[second];

    //All coordinates in ROI center space

    int p0x = 0;
    int p0y = 0;

    //Center of first image in ROI space
    int aToRoiX = tl1.x - roi.x - roi.width / 2;
    int aToRoiY = tl1.y - roi.y - roi.height / 2;
    int ax = img1.width / 2 + aToRoiX;
    int ay = img1.height / 2 + aToRoiY;

    //Center of second image in ROI space
    int bToRoiX = tl2.x - roi.x - roi.width / 2;
    int bToRoiY = tl2.y - roi.y - roi.height / 2;
    int bx = img2.width / 2 + aToRoiX;
    int by = img2.height / 2 + bToRoiY;

    //Derivation of the line orthogonal to the line connecting a and b
    double dy = (ay - by);
    double dx = (ax - bx);
    double l = sqrt(dy * dy + ay * ay);
    dy = dx / l;
    dx = dy / l;

    bool greaterZero = (ax * dy) + (ay * dx) > 0;

    for(int x = 0; x < img1.width; x++) 
    {
        for(int y = 0; y < img1.width; y++) 
        {
            int x1 = x + aToRoiX;
            int y1 = y + aToRoiY;

            if(greaterZero) {
                if(x1 * dy + y1 * dx < 0) {
                    mask1.at<uchar>(x, y) = 0;
                }
            } else {
                if(x1 * dy + y1 * dx > 0) {
                    mask1.at<uchar>(x, y) = 0;
                }
            }
        }       
    }

    greaterZero = (bx * dy) + (by * dx) > 0;

    for(int x = 0; x < img2.width; x++) 
    {
        for(int y = 0; y < img2.width; y++) 
        {
            int x2 = x + bToRoiX;
            int y2 = y + bToRoiY;

            if(greaterZero) {
                if(x2 * dy + y2 * dx < 0) {
                    mask2.at<uchar>(x, y) = 0;
                }
            } else {
                if(x2 * dy + y2 * dx > 0) {
                    mask2.at<uchar>(x, y) = 0;
                }
            }
        }       
    }
}

#endif