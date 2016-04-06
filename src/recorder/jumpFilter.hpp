#include "../io/inputImage.hpp"
#include "../imgproc/planarCorrelator.hpp"
#include "../math/support.hpp"

#ifndef OPTONAUT_JUMP_FILTER_HEADER 
#define OPTONAUT_JUMP_FILTER_HEADER 

namespace optonaut {
    /*
     * Class to remove horizontal jumps in sensor input data. 
     */
    class JumpFilter {
        private:
            double threshold;
            
            Mat offs;
            Mat state;
            Mat lastDiff;
        
            Mat last;
        
        static const bool enabled = false;
        public:
            // Jump thresh of 0.06 found via matlab. 
            JumpFilter(double threshold = 0.015) : threshold(threshold) { }

            /*
             * Returns the current rotation estimated by the jump filter. 
             */
            const Mat &GetState() const {
                return state;
            }

            /*
             * Pushes extrinsics into the jump filter. 
             * Returns true if the extrinsics were valid, false if they needed to be corrected. 
             */
            bool Push(Mat &in) {
                
                // If globally disabled, just return input. 
                if(!enabled) {
                    in.copyTo(state);
                    return true;
                }
                
                if(state.cols == 0) {
                    //Init case. 
                    state = in.clone();
                    offs = Mat::eye(4, 4, CV_64F);
                    lastDiff = Mat::eye(4, 4, CV_64F);
                    last = in.clone();
                    return true;
                } else {
                    
                    //cout << "State: " << state << endl;
                   
                    // Calculate horizontal rotation difference.  
                    AssertEQ(determinant(state), 1.0);
                    
                    Mat rvec;
                    Mat diff = last.inv() * in;
                    in.copyTo(last);
                    
                    // Actually take diff of diffs (e.g. second derivation) 
                    ExtractRotationVector(diff * lastDiff.inv(), rvec);
                    
                    double hMovement = abs(rvec.at<double>(1));

                    if(hMovement > threshold) {
                        // Correction case. We update our state by the previous movement we had. 
                        // Todo: Interpolate from last n. n >= 2. 
                        offs = offs * (lastDiff.inv() * diff);
                        state = state * lastDiff;
                        state.copyTo(in);
                        
                        ExtractRotationVector(offs, rvec);
                        
                        cout << "Avoiding Jump, offset: " << rvec.t() << endl;
                        return false;
                    } else {
                        // Valid case. We update our state by the current movement. 
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
