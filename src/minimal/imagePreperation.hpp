#include <vector>
#include <string>

#include "../common/assert.hpp"
#include "../io/inputImage.hpp"
#include "../recorder/recorder.hpp"
#include "../common/assert.hpp"

#ifndef OPTONAUT_MINIMAL_IMAGE_PREPERATION_HEADER
#define OPTONAUT_MINIMAL_IMAGE_PREPERATION_HEADER

namespace optonaut {
namespace minimal {

    bool CompareByFilename (const std::string &a, const std::string &b) {
        return IdFromFileName(a) < IdFromFileName(b);
    }

    /*
     * Provides multiple methods for loading and selecting images in a test environment. 
     */
    class ImagePreperation {
        public:

        /*
         * For all images, creates a copy of the metadata and the downsampled image data. 
         */
        static std::vector<InputImageP> CreateMinifiedCopy(
                const std::vector<InputImageP> &in, int downsample = 2) {

            std::vector<InputImageP> copies; 

            AssertGT(downsample, 0);

            for(auto img : in) {
                InputImageP copy(new InputImage());
                bool loaded = false;

                if(!img->image.IsLoaded()) {
                    loaded = true;
                    img->image.Load();
                }

                cv::Mat small;

                pyrDown(img->image.data, small);

                for(int i = 1; i < downsample; i++) {
                    pyrDown(small, small);
                }

                if(loaded) {
                    img->image.Unload();
                }

                copy->image = Image(small);
                copy->dataRef = img->dataRef;
                copy->originalExtrinsics = img->originalExtrinsics.clone();
                copy->adjustedExtrinsics = img->adjustedExtrinsics.clone();
                copy->intrinsics = img->intrinsics.clone();
                copy->exposureInfo = img->exposureInfo;
                copy->id = img->id;

                copies.push_back(copy);
            }
            
            return copies;
        }

        /*
         * Copies extrinsics from all images in from to all images in to. 
         */
        static void CopyExtrinsics(
                const std::vector<InputImageP> &from, std::vector<InputImageP> &to) {
            auto byId = CreateImageMap(from);

            for(auto img : to) {
                byId.at(img->id)->adjustedExtrinsics.copyTo(img->adjustedExtrinsics);
                byId.at(img->id)->originalExtrinsics.copyTo(img->originalExtrinsics);
            }
        }
        
        /*
         * Copies intrinsics from all images in from to all images in to. 
         */
        static void CopyIntrinsics(
                const std::vector<InputImageP> &from, std::vector<InputImageP> &to) {
            auto byId = CreateImageMap(from);

            for(auto img : to) {
                byId.at(img->id)->intrinsics.copyTo(img->intrinsics);
            }
        }

        /*
         * Sorts the given image list by ID. 
         */
        static void SortById(std::vector<InputImageP> &images) {
            std::sort(images.begin(), images.end(), [] 
                    (const InputImageP &a, const InputImageP &b) {
                        return a->id < b->id;
                    });
        }

        static constexpr int ModeIOS = 0;
        static constexpr int ModeAndroid = 1;
        static constexpr int ModeNone = 2;
      
        /*
         * Loads an prepares a set of input images.
         * Intrinsics are converted as if they would come from an iPhone. 
         */ 
        static std::vector<InputImageP> LoadAndPrepare(
                std::vector<std::string> files, int mode = ModeIOS,
                bool shallow = true, int limit = -1, int step = 1) {
            
            std::sort(files.begin(), files.end(), CompareByFilename);

            std::vector<InputImageP> allImages;

            Mat base, zero, baseInv;

            if(mode == ModeIOS) {
                base = optonaut::Recorder::iosBase;
                zero = Recorder::iosZero;
            } else if(mode == ModeNone) {
                base = Mat::eye(4, 4, CV_64F);
                zero = Mat::eye(4, 4, CV_64F);
            } else {
                base = optonaut::Recorder::androidBase;
                zero = Recorder::androidZero;
            }

            baseInv = base.t();

            int n = files.size();
            limit = limit * step;

            if(limit > 0) {
                n = std::min(limit, n);
            }
            
            for(int i = 0; i < n; i += step) {
                auto img = InputImageFromFile(files[i], shallow); 
                if(mode != ModeNone) {
                    img->originalExtrinsics = base * zero * 
                        img->originalExtrinsics.inv() * baseInv;
                }
                img->adjustedExtrinsics = img->originalExtrinsics;

                allImages.push_back(img);
            }

            return allImages;
        } 

        /*
         * Loads all the image data of all images, if not already loaded. 
         */
        static void LoadAllImages(std::vector<InputImageP> &images) {
            for(auto img : images) {
                if(!img->image.IsLoaded()) {
                    img->image.Load();
                }
            }
        }

        /*
         * Loads a set of input images frmo args. Supports count, skip and phone arguments. 
         */
        static std::vector<InputImageP> LoadAndPrepareArgs(
                const int argc, char** argv, bool shallow = true, 
                int limit = -1, int step = 1) {
            int n = argc - 1;

            std::vector<std::string> files;

            int mode = ModeIOS;

            for(int i = 0; i < n; i++) {
                std::string imageName(argv[i + 1]);

                if(imageName == "-n") {
                    AssertGT(n, i + 1);
                    limit = ParseInt(std::string(argv[i + 2]));
                    i++;
                } else if (imageName == "-s") {
                    AssertGT(n, i + 1);
                    step = ParseInt(std::string(argv[i + 2]));
                    i++;
                } else if (imageName == "-m") {
                    if(argv[i + 2][0] == 'a' || argv[i + 2][0] == 'A') {
                        mode = ModeAndroid; 
                    } else if(argv[i + 2][0] == 'n' || argv[i + 2][0] == 'N') {
                        mode = ModeNone; 
                    }
                    i++;
                } else {
                    files.push_back(imageName);
                }
            }

            return LoadAndPrepare(files, mode, shallow, limit, step);
        }

        /*
         * Creates a lookup table that maps image Ids to their respective images. 
         */
        static std::map<size_t, InputImageP> CreateImageMap(
                const std::vector<InputImageP> &images) {
            
            std::map<size_t, InputImageP> imageById;

            for(auto img : images) {
                imageById[img->id] = img;
            }

            return imageById;
        }
    };
}
}
#endif
