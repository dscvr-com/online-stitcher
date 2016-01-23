#include <opencv2/opencv.hpp>
#include "../common/assert.hpp"

using namespace cv;
using namespace std;

#ifndef OPTONAUT_PANO_BLUR_HEADER
#define OPTONAUT_PANO_BLUR_HEADER

namespace optonaut {
    class PanoramaBlur {

        private: 
            Mat mirrorTransform;

            const cv::Size is; //Input Size
            const cv::Size ds; //Destination Size
            cv::Size ss; //Strip size;

            void VerticalBlend(cv::Mat &target, cv::Mat &source, int startRow, int endRow, bool quadratic = true) {
                
                bool inverse = false;
                
                if(endRow < startRow) {
                    std::swap(startRow, endRow);
                    startRow -= 1;
                    endRow -= 1;
                    inverse = true;
                }

                AssertEQ(target.size(), source.size());
                AssertGT(target.rows, startRow);
                AssertGE(target.rows, endRow);
                AssertGE(startRow, 0);

                int n = endRow - startRow;

                for(int i = startRow; i < endRow; i++) {
                    float alpha = (float)(i - startRow) / (float)n;
                    if(inverse) {
                        alpha = 1.0 - alpha;
                    }
                    if(quadratic) {
                        alpha = alpha * alpha;
                    }
                    addWeighted(target.row(i), alpha, 
                            source.row(i), 1.0 - alpha, 0.0, target.row(i)); 
                }
            }

            void FastBlur(Mat &in, Mat &out, int depth) {
                if(depth == 0)
                    return;

                cv::Size inSize = in.size();

                pyrDown(in, out);
                FastBlur(out, out, depth - 1);
                pyrUp(out, out);

                out = out(cv::Rect(cv::Point(0, 0), inSize));
            }

            void PyrDownRecu(const Mat &in, Mat &out, int depth) {
                if(depth <= 0) 
                    return;
                
                pyrDown(in, out);
                PyrDownRecu(out, out, depth - 1);
            }
            
            void PyrUpUntil(const Mat &in, Mat &out, cv::Size target) {
                pyrUp(in, out);
                
                if(out.rows >= target.height && out.cols >= target.width) { 
                    out = out(cv::Rect(cv::Point(0, 0), target));
                    return;
                }

                PyrUpUntil(out, out, target);
                
            }
        public:

            PanoramaBlur(const cv::Size inputSize, const cv::Size outputSize) :
                is(inputSize), ds(outputSize) { 

                AssertEQ(is.width, ds.width);
            
                vector<cv::Point2f> src = {
                    cv::Point2f(0, 0),
                    cv::Point2f(0, is.height - 1),
                    cv::Point2f(is.width, is.height - 1),
                    cv::Point2f(is.width, 0)
                };

                ss = cv::Size(ds.width, ds.height / 2 - is.height / 2);
                
                vector<cv::Point2f> dest = {
                    cv::Point2f(0, ss.height - 1),
                    cv::Point2f(0, 0),
                    cv::Point2f(ss.width, 0),
                    cv::Point2f(ss.width, ss.height - 1)
                };

                mirrorTransform = getPerspectiveTransform(src, dest);
            }

            void Blur(const Mat &input, Mat &output) {
                
                if(ss.height <= 0) {
                    output = input;
                    return;
                }

                AssertEQ(input.size(), is);
                output = Mat(ds, input.type());

                cv::Rect top(0, 0, ss.width, ss.height);
                const cv::Rect center(0, ss.height, is.width, is.height);
                cv::Rect bottom(0, ss.height + is.height, ss.width, ss.height);
                
                const cv::Rect halfTop(0, 0, ss.width, ss.height / 3 * 2);
                const cv::Rect halfBottom(0, ss.height + is.height + ss.height / 3,
                        ss.width, ss.height / 3 * 2);

                const cv::Rect blackTop(0, 0, ss.width, halfTop.height / 8); 
                const cv::Rect blackBottom(0, 
                        halfBottom.y + halfBottom.height / 8.0 * 7.0, 
                        ss.width, halfBottom.height / 8); 
                
                const int weakGradient = ds.height / 64;
                const int strongGradient = (top.height - halfTop.height) / 2;

                const int weakGradientOffset = weakGradient * 2.0 / 3.0;

                // Top
                warpPerspective(input, 
                   output(top), 
                   mirrorTransform, ss);     

                // Center
                input.copyTo(output(center)); 

                // Bottom
                warpPerspective(input, 
                   output(bottom), 
                   mirrorTransform, ss);

                top = cv::Rect(top.x, top.y, top.width, top.height - weakGradientOffset);
                bottom = cv::Rect(bottom.x, bottom.y + weakGradientOffset,
                        bottom.width, top.height);

                Mat blur6, blur8, blur;

                PyrDownRecu(output, blur6, 6);
                PyrUpUntil(blur6, blur, output.size());

                blur(top).copyTo(output(top));
                blur(bottom).copyTo(output(bottom));
                
                VerticalBlend(output, blur, top.height, top.height + weakGradient);
                VerticalBlend(output, blur, bottom.y + 1, bottom.y - weakGradient);
                
                PyrDownRecu(blur6, blur8, 1);
                PyrUpUntil(blur8, blur, output.size());
                
                blur(halfTop).copyTo(output(halfTop));
                blur(halfBottom).copyTo(output(halfBottom));
                
                VerticalBlend(output, blur, halfTop.height, 
                        halfTop.height + strongGradient);
                VerticalBlend(output, blur, halfBottom.y + 1, 
                        halfBottom.y - strongGradient);

                blur.setTo(Scalar::all(0));
                
                blur(blackTop).copyTo(output(blackTop));
                blur(blackBottom).copyTo(output(blackBottom));
                
                VerticalBlend(output, blur, blackTop.height, 
                        top.height, false);
                VerticalBlend(output, blur, blackBottom.y + 1, 
                        bottom.y, false);
            }

    };
}

#endif