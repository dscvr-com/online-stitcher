#include <opencv2/opencv.hpp>
#include <mutex>

#include "imageCorrespondenceGraph.hpp"
#include "projection.hpp"

using namespace cv;
using namespace std;

#ifndef OPTONAUT_EXPOSURE_COMPENSATOR_HEADER
#define OPTONAUT_EXPOSURE_COMPENSATOR_HEADER

namespace optonaut {
    struct ExposureDiff {
        size_t n; //Count of measured pixels
        double iFrom; //Overlapping area brightness mine. 
        double iTo; //Overlapping area birghtness other.

        ExposureDiff() : n(0), iFrom(0), iTo(0) { }
    };

    class ExposureCompensator : public ImageCorrespondenceGraph<ExposureDiff> {
        private: 
            static const bool debug = false;
            map<size_t, double> gains;
        public:
            ExposureCompensator() { }
            ExposureCompensator(ExposureCompensator &ref) {
                SetGains(ref.GetGains());
            }
        
            void SetGains(map<size_t, double> gains)
            {
                this->gains = gains;
            }

            const map<size_t, double>& GetGains() const {
                return gains;
            }
        
            ExposureDiff Register(InputImageP a, InputImageP b) {
                Mat ga, gb;
                GetOverlappingRegion(a, b, a->image, b->image, ga, gb);
                
                if(ga.cols < 1 || ga.rows < 1) {
                    ExposureDiff zero;
                    return zero;
                }

                return ImageCorrespondenceGraph<ExposureDiff>::Register(ga, a->id, gb, b->id);
            }

            virtual ExposureDiff GetCorrespondence(const Mat &a, size_t aId, const Mat &b, size_t bId, ExposureDiff &aToB, ExposureDiff &bToA) {
                assert(a.cols == b.cols && a.rows == b.rows);
                assert(a.type() == b.type());

                double sumB = 0;
                double sumA = 0;
                unsigned char *aData = a.data;
                unsigned char *bData = b.data;
                size_t width = a.cols;
                size_t height = a.rows;

                static const int skip = 10;
                size_t size = width * height / skip;

                if(a.type() == CV_8UC1) {
                    for(size_t x = 0; x < width * height; x++) {
                        sumA += aData[x];
                        sumB += bData[x];
                    }
                } else if(a.type() == CV_8UC3) {
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
                
                bToA.n = aToB.n = size;

                bToA.iFrom = aToB.iTo = sumB / size;
                aToB.iFrom = bToA.iTo = sumA / size;
                
                if(debug) {
                    Mat target(a.rows, a.cols * 2, a.type());
                    a.copyTo(target(cv::Rect(0, 0, a.cols, a.rows)));
                    b.copyTo(target(cv::Rect(a.cols, 0, a.cols, a.rows)));

                    imwrite("dbg/corr" + ToString(aId) + "_" + ToString(bId) + "_sa_" + ToString(sumA / size) + "_sb_" + ToString(sumB / size) +  ".jpg", target);
                }
                
                return aToB;
            };

            void PrintCorrespondence() {
                for(auto &adj : relations.GetEdges()) {
                    for(auto &edge : adj.second) {
                        cout << edge.from << " -> " << edge.to << ": " << edge.value.iFrom << " -> " << edge.value.iTo << endl;
                    }
                }
            }

            void FindGains() {

                size_t maxId = 0;

                for(auto &adj : relations.GetEdges()) {
                    maxId = max(maxId, adj.first);
                }
                
                vector<int> remap(maxId);
                vector<int> invmap;
                
                for(auto &adj : relations.GetEdges()) {
                    remap[adj.first] = (int)invmap.size();
                    invmap.push_back((int)adj.first);
                }

                int n = (int)invmap.size();
                
                double alpha = 0.1;
                double beta = 10;
                
                //Build equation systen
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

                solve(A, b, gains);
                
                assert((int)invmap.size() == n);

                for (int i = 0; i < n; ++i) {
                    this->gains[invmap[i]] = gains.at<double>(i, 0);
                    //cout << invmap[i] << " gain: " << gains.at<double>(i, 0) << endl;
                }

            }

            void Apply(Mat &image, size_t id, double ev = 0) const {
                if(gains.find(id) != gains.end()) {
                    multiply(image, gains.at(id) + ev, image);
                }
            }
    };

}

#endif
