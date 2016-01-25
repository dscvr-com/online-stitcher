#include "../io/inputImage.hpp"
#include "../imgproc/planarCorrelator.hpp"
#include "../math/support.hpp"

#ifndef OPTONAUT_JUMP_FILTER_HEADER 
#define OPTONAUT_JUMP_FILTER_HEADER 

namespace optonaut {
    class JumpFilter {
        private:
            double threshold;
            
            Mat offs;
            Mat state;
            Mat lastDiff;
        
        static const bool enabled = false;
        public:
            // Jump thresh of 0.06 found via matlab. 
            JumpFilter(double threshold = 0.02) : threshold(threshold) { }

            const Mat &GetState() const {
                return state;
            }

            bool Push(Mat &in) {
                
                if(!enabled) {
                    in.copyTo(state);
                    return false;
                }
                
                if(state.cols == 0) {
                    //Init case. 
                    state = in.clone();
                    offs = Mat::eye(4, 4, CV_64F);
                    lastDiff = Mat::eye(4, 4, CV_64F);
                    return true;
                } else {
                    
                    AssertEQ(determinant(state), 1.0);
                    
                    Mat rvec;
                    Mat diff = state.inv() * offs.inv() * in;
                    ExtractRotationVector(diff, rvec);
                    
                    double hMovement = abs(rvec.at<double>(1));

                    if(hMovement > threshold) {
                        // Todo: Interpolate from last n. n >= 2. 
                        offs = offs * (lastDiff.inv() * diff);
                        state = state * lastDiff;
                        state.copyTo(in);
                        cout << "Avoiding Jump" << endl;
                        return false;
                    } else {
                        diff.copyTo(lastDiff);
                        state = state * diff;
                        state.copyTo(in);
                        return true;
                    }
                }
            }
    };
}

#endif
