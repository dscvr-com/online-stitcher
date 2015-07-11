#include <opencv2/opencv.hpp>

#ifndef QUAT_H
#define QUAT_H

namespace optonaut {

//Base quaternion support library. 
//Quaternion form: (w, x, y, z) 
//With w real, and x, y, z vector part. 
namespace quat {

    typedef double QT; 

	bool IsQuat(const cv::Mat &q);
	void ToMat(const cv::Mat &q, cv::Mat &a);
	void FromMat(const cv::Mat &a, cv::Mat &q);
    QT Dot(const cv::Mat& a, const cv::Mat &b);

    void Cross(const cv::Mat& a, const cv::Mat &b, cv::Mat &res);
    void Mult(const cv::Mat& a, const cv::Mat &b, cv::Mat &res);
    void Mult(const cv::Mat& a, const QT &b, cv::Mat &res);
    QT Norm(const cv::Mat& a);
    void Conjugate(const cv::Mat &a, cv::Mat &res);
}
}

#endif
