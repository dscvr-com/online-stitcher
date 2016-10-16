#include <opencv2/opencv.hpp>
#include "../common/assert.hpp"
#include "../common/logger.hpp"

using namespace cv;
using namespace std;

#ifndef OPTONAUT_PANO_BLUR_HEADER
#define OPTONAUT_PANO_BLUR_HEADER

namespace optonaut {

    /*
     * Class capable of extending a single ring image by a blurry area. 
     */
    class PanoramaBlur {

        private: 
            Mat mirrorTransform;

            const cv::Size is; //Input Size
            const cv::Size ds; //Destination Size
            cv::Size ss; //Strip size;

            /*
             * Performs gradient blending on two images. 
             */
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

            /*
             * Does a fast blur by up- and downsampling. 
             */
            void FastBlur(Mat &in, Mat &out, int depth) {
                if(depth == 0)
                    return;

                cv::Size inSize = in.size();

                pyrDown(in, out);
                FastBlur(out, out, depth - 1);
                pyrUp(out, out);

                out = out(cv::Rect(cv::Point(0, 0), inSize));
            }

            /*
             * Recursiveley downsamples an image. 
             */
            void PyrDownRecu(const Mat &in, Mat &out, int depth) {
                if(depth <= 0) 
                    return;
                
                pyrDown(in, out);
                PyrDownRecu(out, out, depth - 1);
            }
            
            /*
             * Recursively upsamples an image. 
             */
            void PyrUpUntil(const Mat &in, Mat &out, cv::Size target) {
                pyrUp(in, out);
                
                if(out.rows >= target.height && out.cols >= target.width) { 
                    out = out(cv::Rect(cv::Point(0, 0), target));
                    return;
                }

                PyrUpUntil(out, out, target);
                
            }
        public:

            /*
             * Creates a new instance of this class. 
             *
             * @param inputSize The size of the input image. 
             * @param outputSize The size of the output image. 
             */
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

      
            void Black(const Mat &input, Mat &output) {
                
                if(ss.height <= 0) {
                    output = input;
                    return;
                }

                AssertEQ(input.size(), is);
                output = Mat(ds, input.type());
                output = cv::Scalar(0, 0, 0);
                
                const cv::Rect center(0, ss.height, is.width, is.height);
                cv::Rect bottom(0, ss.height + is.height, ss.width, ss.height);
                input.copyTo(output(center));
             
            }
        
            /*
             * Performs the panorama blur. Basically it mirrors the image once to the top,
             * once to the bottom, and then applies a gradient blur and a gradient to black.
             *
             * @param input The input image.
             * @param output The output image.
             */
            void Blur(const Mat &input, Mat &output) {
                
                Log << "In: " << input.size();
            
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
                
                const int blurBorder = ss.width / 4;
                
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
                
                Mat blurCanvas =
                Mat::zeros(output.rows, output.cols + blurBorder * 2, CV_8UC3);
                
                cv::Rect blurCanvasCenter =
                cv::Rect(blurBorder, 0, output.cols, output.rows);
                
                // Full output
                output.copyTo(blurCanvas(blurCanvasCenter));
                
                // Right border
                output(cv::Rect(0, 0, blurBorder, output.rows)).copyTo(
                                                                       blurCanvas(cv::Rect(blurBorder + output.cols, 0,
                                                                                           blurBorder, output.rows)));
                
                // Left border
                output(cv::Rect(output.cols - blurBorder, 0,
                                blurBorder, output.rows)).copyTo(
                                                                 blurCanvas(cv::Rect(0, 0,
                                                                                     blurBorder, output.rows)));
                
                PyrDownRecu(blurCanvas, blur6, 6);
                PyrUpUntil(blur6, blur, blurCanvas.size());
                
                blur = blur(blurCanvasCenter);
                
                blur(top).copyTo(output(top));
                blur(bottom).copyTo(output(bottom));
                
                VerticalBlend(output, blur, top.height, top.height + weakGradient);
                VerticalBlend(output, blur, bottom.y + 1, bottom.y - weakGradient);
                
                PyrDownRecu(blur6, blur8, 1);
                PyrUpUntil(blur8, blur, blurCanvas.size());
                
                blur = blur(blurCanvasCenter);
                
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
