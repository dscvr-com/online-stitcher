/*
 * Quaternion module.  
 * Quaternion form: (w, x, y, z) 
 * With w real, and x, y, z vector part. 
 */

#include <opencv2/opencv.hpp>

#ifndef QUAT_H
#define QUAT_H

namespace optonaut {

namespace quat {

    // Set double as quaternion element type. 
    typedef double QT; 

    /*
     * Creates a new quaternion.
     */
    void MakeQuat(cv::Mat &q);

    /*
     * Checks if the given mat is a valid quaternion. 
     */
	bool IsQuat(const cv::Mat &q);

    /*
     * Converts a quaternion to its rotation matrix representation. 
     */
	void ToMat(const cv::Mat &q, cv::Mat &a);
    
    /*
     * Converts a rotation matrix to its quaternion representation. 
     */
	void FromMat(const cv::Mat &a, cv::Mat &q);

    /**
     * Calculates the dot product of two quaternions. 
     */
    QT Dot(const cv::Mat& a, const cv::Mat &b);

    /*
     * Calculates the cross product of two quaternions. 
     */
    void Cross(const cv::Mat& a, const cv::Mat &b, cv::Mat &res);

    /*
     * Multiplies two quaternions by each other. 
     */
    void Mult(const cv::Mat& a, const cv::Mat &b, cv::Mat &res);

    /*
     * Multiplies a quaternion by a scalar. 
     */
    void Mult(const cv::Mat& a, const QT &b, cv::Mat &res);

    /*
     * Calculates the norm of a quaternion. 
     */
    QT Norm(const cv::Mat& a);

    /*
     * Conjugates a quaterion. In case of a rotation quaternion, this inverts the rotation. 
     */
    void Conjugate(const cv::Mat &a, cv::Mat &res);
}
}

#endif
