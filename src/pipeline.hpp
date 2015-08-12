#include <algorithm>

#include "image.hpp"
#include "streamAligner.hpp"
#include "monoStitcher"
#include "imageSelector"

#ifndef OPTONAUT_PIPELINE_HEADER
#define OPTONAUT_PIPELINE_HEADER

namespace optonaut {
    class Pipeline {

    private: 

        Mat base;
        Mat baseInv;

        StreamAligner aligner;
        ImageSelector selector; 

        MonoStitcher stereoConverter;
        
        //Portrait to landscape (use with ios app)
        const double iosBaseData[] = {0, 1, 0, 0,
                                 1, 0, 0, 0, 
                                 0, 0, 1, 0,
                                 0, 0, 0, 1};

        //Landscape L to R (use with android app)
        const double androidBaseData[] = {-1, 0, 0, 0,
                                     0, -1, 0, 0, 
                                     0, 0, 1, 0,
                                     0, 0, 0, 1};
    public: 

        const Mat androidBase(4, 4, CV_64F, androidBaseData);
        const Mat iosBase(4, 4, CV_64F, iosBaseData);
        
        Pipeline(Mat base, Mat intrinsics) : base(base), selector(intrinsics) {
            baseInv = base.inv();
        }

        //In: Image with sensor sampled parameters attached. 
        void Push(ImageP image) {
            image->extrinsics = base * image->extrinsics * baseInv;

            aligner.push(image);
            image->extrinsics = aligner.GetCurrentRotation().clone();

            //Todo: Circle selection. Pair selection. 

      		
        }       
    };
}

#endif