
#include <opencv2/core.hpp>
#include <opencv2/opencv_modules.hpp>
#include <opencv2/stitching.hpp>
#include <vector>

#ifndef OPTONAUT_THRESHOLD_SEAM_HEADER
#define OPTONAUT_THRESHOLD_SEAM_HEADER

using namespace std;
using namespace cv;
using namespace cv::detail;

class ThresholdSeamer : public PairwiseSeamFinder
{
public:
    virtual void find(const std::vector<UMat> &src, const std::vector<Point> &corners,
                      std::vector<UMat> &masks);
    virtual void find(const std::vector<Size> &size, const std::vector<Point> &corners,
                      std::vector<UMat> &masks);
    static void createMask(Mat &mask);
    static void brightenMask(UMat &mask);
private:
    void findInPair(size_t first, size_t second, Rect roi);

};

void ThresholdSeamer::brightenMask(UMat &um) {
    Mat m = um.getMat(ACCESS_WRITE);
    for(int x = 0; x < m.cols; x++) 
    {
        for(int y = 0; y <  m.rows; y++) 
        {
            if(m.at<uchar>(y, x) > 0) {
                m.at<uchar>(y, x) = 255;
            }
        } 
    }
}

void ThresholdSeamer::createMask(Mat &mask) {

    //Draw a box gradient based on euclidean distance. 
    int cx = mask.cols / 2;
    int cy = mask.rows / 2;

    assert(mask.type() == CV_8U);

    for(int x = 0; x < mask.cols; x++) 
    {
        for(int y = 0; y < mask.rows; y++) 
        {
            mask.at<uchar>(y, x) = 255 - max(abs(cx - x) * 255 / cx, abs(cy - y) * 255 / cy);
        } 
    }

}

void ThresholdSeamer::find(const std::vector<UMat> &src, const std::vector<Point> &corners,
                             std::vector<UMat> &masks)
{
    PairwiseSeamFinder::find(src, corners, masks);
}

void ThresholdSeamer::find(const std::vector<Size> &sizes, const std::vector<Point> &corners,
                             std::vector<UMat> &masks)
{
    if (sizes.size() == 0)
        return;

    sizes_ = sizes;
    corners_ = corners;
    masks_ = masks;
    run();


}

void ThresholdSeamer::findInPair(size_t first, size_t second, Rect roi)
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

    //Center of first image in ROI space
    int roiXToA = -tlA.x + roi.x;
    int roiYToA = -tlA.y + roi.y;

    //Center of second image in ROI space
    int roiXToB = -tlB.x + roi.x;
    int roiYToB = -tlB.y + roi.y;

    
    for(int x = 0; x < roi.width; x++) 
    {
        for(int y = 0; y < roi.height; y++) 
        {
            int ax = x + roiXToA;
            int ay = y + roiYToA;
            int bx = x + roiXToB;
            int by = y + roiYToB;

            if(maskA.at<uchar>(ay, ax) > maskB.at<uchar>(by, bx)) {
                maskB.at<uchar>(by, bx) = 0;
            } else {
                maskA.at<uchar>(ay, ax) = 0;
            }
        }
    }
}

#endif