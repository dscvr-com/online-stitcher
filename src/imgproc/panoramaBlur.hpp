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

            const Size is; //Input Size
            const Size ds; //Destination Size
            Size ss; //Strip size;

            void VerticalBlend(Mat &target, Mat &source, int startRow, int endRow) {
                
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
                    addWeighted(target.row(i), alpha, 
                            source.row(i), 1.0 - alpha, 0.0, target.row(i)); 
                }
            }

            void FastBlur(Mat &in, Mat &out, int depth) {
                if(depth == 0)
                    return;

                Size inSize = in.size();

                pyrDown(in, out);
                FastBlur(out, out, depth - 1);
                pyrUp(out, out);

                out = out(Rect(Point(0, 0), inSize));
            }

            void PyrDownRecu(const Mat &in, Mat &out, int depth) {
                if(depth <= 0) 
                    return;
                
                pyrDown(in, out);
                PyrDownRecu(out, out, depth - 1);
            }
            
            void PyrUpUntil(const Mat &in, Mat &out, Size target) {
                pyrUp(in, out);
                
                if(out.rows >= target.height && out.cols >= target.width) { 
                    out = out(Rect(Point(0, 0), target));
                    return;
                }

                PyrUpUntil(out, out, target);
                
            }
        public:

            PanoramaBlur(const Size inputSize, const Size outputSize) :
                is(inputSize), ds(outputSize) { 

                AssertEQ(is.width, ds.width);
            
                vector<Point2f> src = {
                    Point(0, 0),
                    Point(0, is.height),
                    Point(is.width, is.height),
                    Point(is.width, 0)
                };

                ss = Size(ds.width, ds.height / 2 - is.height / 2);
                
                vector<Point2f> dest = {
                    Point(0, ss.height),
                    Point(0, 0),
                    Point(ss.width, 0),
                    Point(ss.width, ss.height)
                };

                mirrorTransform = getPerspectiveTransform(src, dest);
            }

            void Blur(const Mat &input, Mat &output) {

                AssertEQ(input.size(), is);
                output = Mat(ds, input.type());

                const Rect top(0, 0, ss.width, ss.height);
                const Rect center(0, ss.height, is.width, is.height);
                const Rect bottom(0, ss.height + is.height, ss.width, ss.height);
                
                const Rect halfTop(0, 0, ss.width, ss.height / 3 * 2);                
                const Rect halfBottom(0, ss.height + is.height + ss.height / 3, 
                        ss.width, ss.height / 3 * 2);

                const Rect blackTop(0, 0, ss.width, halfTop.height / 2); 
                const Rect blackBottom(0, halfBottom.y + halfBottom.height / 2, 
                        ss.width, halfBottom.height / 2); 
                
                const int weakGradient = ds.height / 128;
                const int strongGradient = top.height - halfTop.height;

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
                VerticalBlend(output, blur, halfBottom.y, 
                        halfBottom.y - strongGradient);

                blur.setTo(Scalar::all(0));
                
                blur(blackTop).copyTo(output(blackTop));
                blur(blackBottom).copyTo(output(blackBottom));
                
                VerticalBlend(output, blur, blackTop.height, 
                        top.height);
                VerticalBlend(output, blur, blackBottom.y + 1, 
                        bottom.y);
            }

    };
}

#endif
