/*
 * Math helper  module. 
 */
#include <algorithm>
#include <string>
#include <opencv2/opencv.hpp>
#include "../common/image.hpp"

#ifndef OPTONAUT_MATH_SUPPORT_HEADER
#define OPTONAUT_MATH_SUPPORT_HEADER

namespace optonaut {

    /*
     * Checks if a math is at least the given size and the given type. 
     */
	bool MatIs(const cv::Mat &in, int rows, int cols, int type);
   
    /*
     * Asserts that a given mat object equals another mat object. 
     */ 
    template <typename T>
    void AssertMatEQ(const cv::Mat &a, const cv::Mat &b) {
        AssertEQ(a.cols, b.cols);
        AssertEQ(a.rows, b.rows);

        for(int i = 0; i < a.cols; i++) {
            for(int j = 0; j < a.rows; j++) {
                AssertEQ(a.at<T>(j, i), b.at<T>(j, i));
            }
        }
    }

    /*
     * Scales a given set of intrinsic parameters to match the pixel dimension of the given image. 
     */
	void ScaleIntrinsicsToImage(const cv::Mat &intrinsics, const cv::Size &image, cv::Mat &scaled, double fupscaling = 1);

    /*
     * Scales a given set of intrinsic parameters to match the pixel dimension of the given size. 
     */
	void ScaleIntrinsicsToImage(const cv::Mat &intrinsics, const Image &image, cv::Mat &scaled, double fupscaling = 1);

    /*
     * Gets the horizontal field of view for given intrinsics. 
     */
	double GetHorizontalFov(const cv::Mat &intrinsics);
    
    /*
     * Gets the vertical field of view for given intrinsics. 
     */
    double GetVerticalFov(const cv::Mat &intrinsics);
    
    /*
     * Returns true if the given intrinsics correspond to a portrait camera. 
     */ 
    double IsPortrait(const cv::Mat &intrinsics);

    /*
     * Decomposes a rotation matrix into it's respective roations around x, y and z directions. 
     */
	void ExtractRotationVector(const cv::Mat &r, cv::Mat &vec);

    /*
     * Calculates the absolute angle of rotation of a rotation matrix.
     */
	double GetAngleOfRotation(const cv::Mat &r);

    /*
     * Calculates the absolute angle of rotation between to rotation matrices. 
     */ 
    double GetAngleOfRotation(const cv::Mat &a, const cv::Mat &b);
    
    /*
     * Creates a rotation matrix around the z axis. 
     */
	void CreateRotationZ(double radians, cv::Mat &out);
    
    /*
     * Creates a rotation matrix around the x axis. 
     */
	void CreateRotationX(double radians, cv::Mat &out);
    
    /*
     * Creates a rotation matrix around the y axis. 
     */
    void CreateRotationY(double radians, cv::Mat &out);
    
    /*
     * Calculates a linear interpolation between two rotation matrices. 
     */ 
    void Lerp(const cv::Mat &a, const cv::Mat &b, const double t, cv::Mat &out);

    /*
     * Calculates a spherical linear interpolation between two rotation matrices. 
     */ 
    void Slerp(const cv::Mat &a, const cv::Mat &b, const double t, cv::Mat &out);
    
    /*
     * Extracts the rotation angle relative to a given coordinate axis. 
     */ 
	double GetDistanceByDimension(const cv::Mat &a, const cv::Mat &b, int dim);
	double GetDistanceX(const cv::Mat &a, const cv::Mat &b);
	double GetDistanceY(const cv::Mat &a, const cv::Mat &b);
	double GetDistanceZ(const cv::Mat &a, const cv::Mat &b);
    
    /*
     * Conversion methods. Convert different matrix representations to each other.
     */
	void From4DoubleTo3Float(const cv::Mat &in, cv::Mat &out);
	void From3DoubleTo3Float(const cv::Mat &in, cv::Mat &out);
	void From3FloatTo4Double(const cv::Mat &in, cv::Mat &out);
	void From3FloatTo3Double(const cv::Mat &in, cv::Mat &out);
	void From3DoubleTo4Double(const cv::Mat &in, cv::Mat &out);
	void From4DoubleTo3Double(const cv::Mat &in, cv::Mat &out);

    /*
     * Converts NxM double matrix to UxV float matrix, where N and U are width (column count), M and V are height (row count)
     * of the matrices. If out matrix is bigger, the remaining entries are filled according to an identity matrix. 
     */
	template<int N, int M, int U, int V> void FromNMDoubleToUVFloat(const cv::Mat &in, cv::Mat &out) {
		assert(MatIs(in, M, N, CV_64F));

		out = cv::Mat::zeros(V, U, CV_32F);
		for(int i = 0; i < std::min(M, V); i++) {
			for(int j = 0; j < std::min(N, U); j++) {
				out.at<float>(i, j) = in.at<double>(i, j);
			}
		}
		for(int i = std::min(N, M); i < std::min(U, V); i++) {
            out.at<float>(i, i) = 1;
        }
	}
    
    /*
     * Checks if a matrix contains NaN entries. 
     */ 
	bool ContainsNaN(const cv::Mat &in);

    /*
     * Simple min/max functions. 
     */
	double min2(double a, double b);
	double max2(double a, double b);
	double min4(double a, double b, double c, double d);
	double max4(double a, double b, double c, double d);
    
    /*
     * Calculates the "average" of a angle. This is the angle which lies 
     * exactly between the two given angles. 
     * Handles wrapping around correctly. 
     */
	double angleAvg(double x, double y);
    
    /*
     * Interpolates y=f(x) on a line f given by x1, y1 and x2, y2
     */
	double interpolate(double x, double x1, double x2, double y1, double y2);
    
    /*
     * Returns y=g(x), for g is a gauss curve defined by (a, b, c)
     * a = height of peak
     * b = position of peak
     * c = width of peak
     */
    double gauss(double x, double a, double b, double c);

    /*
     * Calculates the gradient (first derivation) of a black and white image. 
     */
    void GetGradient(const cv::Mat &src_gray, cv::Mat &grad, double wx = 0.5, double wy = 0.5);
}

#endif
