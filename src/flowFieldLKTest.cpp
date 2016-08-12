#include <iostream>
#include <algorithm>
#include <memory>
#include <chrono>
#include <thread>
#include <opencv2/stitching/detail/blenders.hpp>
#include <opencv2/core/ocl.hpp>

#include "minimal/imagePreperation.hpp"
#include "common/intrinsics.hpp"
#include "common/static_timer.hpp"
#include "imgproc/planarCorrelator.hpp"
#include "io/io.hpp"
#include "math/projection.hpp"
#include "recorder/recorder.hpp"
#include "stitcher/simpleSphereStitcher.hpp"
#include "stitcher/simplePlaneStitcher.hpp"

using namespace std;
using namespace cv;
using namespace optonaut;
using namespace std::chrono;

void RepackInliers(Mat &pA, Mat &pB, Mat &status, float scale = 1) {
    int k = 0;
    for(int i = 0; i < pA.rows; i++) {
        if(status.at<uchar>(i)) {
            pA.at<float>(k, 0) = pA.at<float>(i, 0) / scale;
            pA.at<float>(k, 1) = pA.at<float>(i, 1) / scale;
            pB.at<float>(k, 0) = pB.at<float>(i, 0) / scale;
            pB.at<float>(k, 1) = pB.at<float>(i, 1) / scale;
            k++;
        }
    }
    pA.resize(k);
    pB.resize(k);
    status.resize(k);
}

void FastPyrLK(const Mat &inA, const Mat &inB, Mat &pA, Mat &pB, const int verticalPointCount = 15, const cv::Size targetSize = cv::Size(160, 160)) {
    Assert(inA.type() == CV_8UC3 || inA.type() == CV_8UC1);
    Assert(inB.type() == CV_8UC3 || inB.type() == CV_8UC1);
    Assert(inA.size() == inB.size());

    const cv::Size inSize = inA.size();

    double scale = std::max(1.0, std::max((double)targetSize.width/inSize.width, (double)targetSize.height/inSize.height )); 

    const cv::Size workingSize(std::round(inSize.width * scale), std::round(inSize.height * scale));

    Mat a, b, ta, tb;

    if(inA.type() == CV_8UC3) {
        cvtColor(inA, ta, COLOR_BGR2GRAY);
    } else {
        ta = inA;
    }
    
    if(inB.type() == CV_8UC3) {
        cvtColor(inB, tb, COLOR_BGR2GRAY);
    } else {
        tb = inB;
    }

    if(inSize == workingSize) {
        a = ta;
        b = tb;
    } else {
        resize(ta, a, workingSize, 0.0, 0.0, INTER_AREA);
        resize(tb, b, workingSize, 0.0, 0.0, INTER_AREA);
    }

    const int countX = verticalPointCount;
    const int countY = std::ceil(countX * (double)workingSize.height / workingSize.width);
    const int pointCount = countX * countY;

    pA = Mat(pointCount, 2, CV_32F);
    pB = Mat(pointCount, 2, CV_32F);
    Mat status(pointCount, 1, CV_8UC1);

    for(int i = 0, k = 0; i < countX; i++) {
        for(int j = 0; j < countY; j++, k++) {
            pA.at<float>(k, 0) = (i + 0.5f) * workingSize.width / countX;
            pA.at<float>(k, 1) = (j + 0.5f) * workingSize.height / countY;
        }
    }


    calcOpticalFlowPyrLK(a, b, pA, pB, status, noArray(), Size(21, 21), 3,
                               TermCriteria(TermCriteria::MAX_ITER,40,0.1)); 

    RepackInliers(pA, pB, status, scale);
}

bool CompareByFilename (const string &a, const string &b) {
    return IdFromFileName(a) < IdFromFileName(b);
}

int main(int argc, char** argv) {
    cv::ocl::setUseOpenCL(false);

    auto images = minimal::ImagePreperation::LoadAndPrepareArgs(argc, argv, false);
    auto n = images.size();

    for(size_t i = 0; i < n - 1; i += 1) {
        Mat a = images[i]->image.data;
        Mat b = images[i + 1]->image.data;
        Mat pA, pB;

        FastPyrLK(a, b, pA, pB, 15, a.size());
        
        for(int j = 0; j < pA.rows; j++) {
            Point from(pA.at<float>(j, 0), pA.at<float>(j, 1));
            Point to(pB.at<float>(j, 0), pB.at<float>(j, 1));
            line(a, from, to, Scalar(0, 255, 0), 1);
        }

        Mat inliers(a.rows, 1, CV_8UC1);

        Mat trans = estimateRigidTransform(pA, pB, true);

        if(trans.cols == 0 || trans.rows == 0) {
            cout << "Skipping due to affine estimation error." << endl;
            continue;
        }

        cout << "Affine: " << trans << endl;

        Mat A, t;
        FromNMDoubleToUVFloat<2, 2, 2, 2>(trans(Rect(0, 0, 2, 2)), A);
        FromNMDoubleToUVFloat<1, 2, 1, 2>(trans(Rect(2, 0, 1, 2)), t);

        for(int j = 0; j < pA.rows; j++) {
            Mat diff = pB(Rect(0, j, 2, 1)).t() - A * pA(Rect(0, j, 2, 1)).t() - t;
            inliers.at<uchar>(j) = std::abs(diff.at<float>(0) * diff.at<float>(1)) < 4;
            //if(!inliers.at<uchar>(j)) {
            //    cout << "Reject: " << diff << endl;
            //} else {
            //    cout << "Pass: " << diff << endl;
            //}
        }
        
        RepackInliers(pA, pB, inliers);
        
        for(int j = 0; j < pA.rows; j++) {
            Point from(pA.at<float>(j, 0), pA.at<float>(j, 1));
            Point to(pB.at<float>(j, 0), pB.at<float>(j, 1));
            line(a, from, to, Scalar(255, 0, 0), 1);
        }

        Mat f = findFundamentalMat(pA, pB, CV_RANSAC, 1.0, 0.99, inliers);
        
        if(f.cols != 3 || f.rows != 3) {
            cout << "Skipping due to fundamental estimation error." << endl;
            continue;
        }

        RepackInliers(pA, pB, inliers);

        Mat H1, H2;

        if(pA.rows == 0 || pB.rows == 0) {
            cout << "Skipping, no more inliers." << endl;
            continue;
        }

        stereoRectifyUncalibrated(pA, pB, f, a.size(), H1, H2);
        
       // Mat oA(a.size(), CV_8UC3);
        Mat oB(b.size(), CV_8UC3);

        //cout << H1.size() << endl;

       // warpPerspective(a, oA, Mat::eye(3, 3, CV_32F), a.size()); 
        warpPerspective(b, oB, H1.inv() * H2, b.size()); 

       // imwrite("dbg/" + ToString(images[i]->id) + "rect_a.jpg", oA);
        imwrite("dbg/" + ToString(images[i]->id) + "rect_b.jpg", oB);
        
        for(int j = 0; j < pA.rows; j++) {
            Point from(pA.at<float>(j, 0), pA.at<float>(j, 1));
            Point to(pB.at<float>(j, 0), pB.at<float>(j, 1));
            line(a, from, to, Scalar(0, 0, 255), 1);
        }
        
        //cout << f << endl;
        imwrite("dbg/" + ToString(images[i]->id) + "_img.jpg", a);
    

    }
    return 0;
}
