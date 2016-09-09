#include <opencv2/opencv.hpp>
#include <mutex>

#include "../common/imageCorrespondenceGraph.hpp"
#include "../common/support.hpp"
#include "../math/projection.hpp"

using namespace cv;
using namespace std;

#ifndef OPTONAUT_EXPOSURE_COMPENSATOR_HEADER
#define OPTONAUT_EXPOSURE_COMPENSATOR_HEADER

namespace optonaut {
    /*
     * Represnets a difference in exposure between two images. 
     */
    struct ExposureDiff {
        size_t n; //Count of measured pixels
        double iFrom; //Overlapping area brightness mine. 
        double iTo; //Overlapping area birghtness other.

        ExposureDiff() : n(0), iFrom(0), iTo(0) { }
        
        friend ostream& operator<< (ostream& os, const ExposureDiff& e) {
            return os << e.iFrom << " -> " << e.iTo;
        }
    };
    

    /*
     * Holds a set of pairwise exposure differences for a pair of images. 
     *
     * This class is capable of finding optimal exposure gains for normalizing the
     * exposure of the image set. 
     */
    class ExposureCompensator : public ImageCorrespondenceGraph<ExposureDiff> {
        private: 
            static const bool debug = false;
            map<size_t, double> gains;
        public:
            /*
             * Creates a new instance of this class.
             */
            ExposureCompensator() {
                AssertFalseInProduction(debug);
            }

            /*
             * Copy constructor. 
             */ 
            ExposureCompensator(ExposureCompensator &ref) {
                AssertFalseInProduction(debug);
                SetGains(ref.GetGains());
            }
       
            /*
             * Sets gains for this exposure compensator. 
             */ 
            void SetGains(map<size_t, double> gains)
            {
                this->gains = gains;
            }

            /*
             * Gets the gains of this exposure compensator. 
             */
            const map<size_t, double>& GetGains() const {
                return gains;
            }
       
            /*
             * Convenience overload. 
             */
            ExposureDiff Register(InputImageP a, InputImageP b) {
                return ImageCorrespondenceGraph<ExposureDiff>::Register(a, b);
            }

            /*
             * Calculates the exposure difference of the given image pair and adds it to the exposure graph. 
             */
            virtual ExposureDiff GetCorrespondence(InputImageP imgA, InputImageP imgB, ExposureDiff &aToB, ExposureDiff &bToA) {
                AssertM(false, "This code is disabled. Exposure info is to be given from outside");

                // Extract the overlapping region of the image pair. 
                Mat a, b;
                GetOverlappingRegion(imgA, imgB, imgA->image, imgB->image, a, b);
                
                if(a.cols < 1 || a.rows < 1) {
                    // No overlap - no correspondence. 
                    ExposureDiff zero;
                    aToB.n = 0;
                    bToA.n = 0;
                    return zero;
                }

                assert(a.cols == b.cols && a.rows == b.rows);
                assert(a.type() == b.type());

                // Now just sum up all pixels and then substract the results. 
                double sumB = 0;
                double sumA = 0;
                unsigned char *aData = a.data;
                unsigned char *bData = b.data;
                size_t width = a.cols;
                size_t height = a.rows;

                static const int skip = 10;
                size_t size = width * height / skip;

                if(a.type() == CV_8UC1) {
                    // Grayscale images - faster!
                    for(size_t x = 0; x < width * height; x++) {
                        sumA += aData[x];
                        sumB += bData[x];
                    }
                } else if(a.type() == CV_8UC3) {
                    // BGR images. 
                    for(size_t x = 0; x < width * height; x += (3 * skip)) {
                        sumA += sqrt((int)aData[x] * (int)aData[x] + 
                                (int)aData[x + 1] * (int)aData[x + 1] + 
                                (int)aData[x + 2] * (int)aData[x + 2]);
                        
                        sumB += sqrt((int)bData[x] * (int)bData[x] + 
                                (int)bData[x + 1] * (int)bData[x + 1] + 
                                (int)bData[x + 2] * (int)bData[x + 2]);
                    }
                } else {
                    cout << "Unsupported mat type" << endl;
                    assert(false);
                } 
                
                // Add the correspondence to the graph. 
                bToA.n = aToB.n = size;

                bToA.iFrom = aToB.iTo = sumB / size;
                aToB.iFrom = bToA.iTo = sumA / size;
                
                if(debug) {
                    Mat target(a.rows, a.cols * 2, a.type());
                    a.copyTo(target(cv::Rect(0, 0, a.cols, a.rows)));
                    b.copyTo(target(cv::Rect(a.cols, 0, a.cols, a.rows)));

                    imwrite("dbg/corr" + ToString(imgA->id) + "_" + ToString(imgB->id) + "_sa_" + ToString(sumA / size) + "_sb_" + ToString(sumB / size) +  ".jpg", target);
                }
                
                return aToB;
            };

            /*
             * Finds optimal compensation gains for all exposure correspondences in this graph.
             */
            void FindGains() {

                // Find size of equation system
                size_t maxId = 0;

                for(auto &adj : relations.GetEdges()) {
                    maxId = max(maxId, adj.first);
                }
                
                vector<int> remap(maxId);
                vector<int> invmap;
                
                // Build lookup table
                for(auto &adj : relations.GetEdges()) {
                    remap[adj.first] = (int)invmap.size();
                    invmap.push_back((int)adj.first);
                }

                int n = (int)invmap.size();
               
                // Alpha damps the regression 
                double alpha = 0.1;
                double beta = 1 / alpha;
                
                // Build equation systen
                Mat I = Mat::zeros(n, n, CV_64F);
                Mat N = Mat::zeros(n, n, CV_64F);

                for(auto &adj : relations.GetEdges()) {
                    for(auto &edge : adj.second) {
                        I.at<double>(remap[edge.from], remap[edge.to]) = edge.value.iFrom;
                        N.at<double>(remap[edge.from], remap[edge.to]) = edge.value.n;
                    }
                }

                Mat A = Mat::zeros(n, n, CV_64F);
                Mat b = Mat::zeros(n, 1, CV_64F);
                Mat gains = Mat(n, 1, CV_64F);

                for(int i = 0; i < n; ++i) {
                    for(int j = 0; j < n; ++j) {
                        b.at<double>(i, 0) += beta * N.at<double>(i, j);
                        A.at<double>(i, i) += beta * N.at<double>(i, j);
                        if (j == i) continue;
                        A.at<double>(i, i) += 2 * alpha * I.at<double>(i, j) * I.at<double>(i, j) * N.at<double>(i, j);
                        A.at<double>(i, j) -= 2 * alpha * I.at<double>(i, j) * I.at<double>(j, i) * N.at<double>(i, j);
                    }
                }

                // Solve for optimum error. 
                solve(A, b, gains);
                
                assert((int)invmap.size() == n);

                for (int i = 0; i < n; ++i) {
                    this->gains[invmap[i]] = gains.at<double>(i, 0);
                    Log << invmap[i] << " gain: " << gains.at<double>(i, 0);
                }

            }

            /*
             * Applies the calculated exposure gain to the given image. 
             * The ev parameter allows for manual exposure adjustment. 
             */
            void Apply(Mat &image, size_t id, double ev = 1) const {
                if(gains.find(id) != gains.end()) {
                    multiply(image, gains.at(id) + ev, image);
                }
            }
    };

}

#endif
