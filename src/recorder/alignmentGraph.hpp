#include <opencv2/opencv.hpp>
#include <mutex>

#include "../common/imageCorrespondenceGraph.hpp"
#include "../common/support.hpp"
#include "../math/projection.hpp"
#include "../imgproc/pairwiseCorrelator.hpp"

using namespace cv;
using namespace std;

#ifndef OPTONAUT_ALIGNMENT_GRAPH_HEADER
#define OPTONAUT_ALIGNMENT_GRAPH_HEADER

namespace optonaut {
    struct AlignmentDiff {
        double dphi; //Difference for rotating around the vertical axis
        double valid; //Is it valid?
        int overlap;
        int error; 
        int rejectionReason;
        bool quartil;
        bool forced;

        AlignmentDiff() : 
            dphi(0), 
            valid(false), 
            overlap(0), 
            error(0),
            rejectionReason(0),
            quartil(false),
            forced(false) { }
        
        friend ostream& operator<< (ostream& os, const AlignmentDiff& e) {
            return os << e.dphi << " (" << e.overlap << ")";
        }
    };
    

    class AlignmentGraph: public ImageCorrespondenceGraph<AlignmentDiff> {
        private: 
            PairwiseCorrelator aligner;
            static const bool debug = false;
            map<size_t, double> alignmentCorrections;
        public:
            AlignmentGraph() { }
            AlignmentGraph(AlignmentGraph &ref) {
                SetAlignment(ref.GetAlignment());
            }
        
            void SetAlignment(map<size_t, double> alignment)
            {
                this->alignmentCorrections = alignment;
            }

            const map<size_t, double>& GetAlignment() const {
                return alignmentCorrections;
            }
        
            virtual AlignmentDiff GetCorrespondence(InputImageP imgA, InputImageP imgB, AlignmentDiff &aToB, AlignmentDiff &bToA) {

                int dist = min(abs(imgA->localId - imgB->localId),
                       imgA->ringSize - (int)abs(imgA->localId - imgB->localId));

                bool areNeighbors = 
                    dist % imgA->ringSize <= 3
                    && imgA->ringId == imgB->ringId;

                if(imgA->ringId != imgB->ringId 
                        || areNeighbors) {
                    int minSize = min(imgA->image.cols, imgA->image.rows) / 1.8;

                    auto res = aligner.Match(imgB, imgA, minSize, minSize);
                    
                    if(!res.valid && areNeighbors) {
                        res.valid = true;
                        res.angularOffset.x = 0;
                        res.overlap = imgA->image.cols * imgA->image.rows / 2;
                        aToB.forced = true;
                    }

                    aToB.dphi = res.angularOffset.x;
                    aToB.overlap = res.overlap;
                    aToB.valid = res.valid;
                    aToB.rejectionReason = res.rejectionReason;
                    aToB.error = res.inverseTestDifference.x;
                    bToA = aToB;
                    aToB.dphi *= -1;
                } else {
                    aToB.dphi = 0;
                    aToB.overlap = 0;
                    aToB.valid = false;

                    bToA = aToB;
                }
                
                return aToB;
            };

            Edges FindAlignment() {
                size_t maxId = 0;
                Edges allEdges;

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
                
                //Build equation systen
                //R = sum of relative offsets.
                //O = matrix of observation weights.
                //x = result, optimal relative offset. 
                Mat R = Mat::zeros(n, 1, CV_64F);
                Mat O = Mat::zeros(n, n, CV_64F);
                Mat x = Mat::zeros(n, 1, CV_64F);

                //Alpha - damping for our regression. 
                double alpha = 2;
                double beta = 1 / alpha;
                double error = 0;
                double weightSum = 0;
                double edgeCount = 0;

                const double quartil = 0.25;

                for(auto &adj : relations.GetEdges()) {

                    Edges correlatorEdges = fun::filter<Edge>(adj.second, 
                        [] (auto a) {
                            return !a.value.forced && a.value.overlap > 0; 
                        }); 

                    Edges forcedEdges = fun::filter<Edge>(adj.second, 
                        [] (auto a) {
                            return a.value.forced && a.value.overlap > 0; 
                        }); 

                    std::sort(correlatorEdges.begin(), correlatorEdges.end(), 
                        [] (auto a, auto b) {
                            return a.value.dphi < b.value.dphi;
                        });

                    size_t m = correlatorEdges.size();

                    for(size_t i = 0; i < m; i++) {
                        if(i < m * quartil || i >= m * (1.0f - quartil)) {
                            correlatorEdges[i].value.quartil = true;
                            allEdges.push_back(correlatorEdges[i]);
                        } else {
                            forcedEdges.push_back(correlatorEdges[i]);
                        }
                    }

                    for(auto &edge : forcedEdges) {
                        //Use non-linear weights - less penalty for less overlap. 
                        if(edge.value.overlap > 0) {
                            double weight = edge.value.overlap;

                            O.at<double>(remap[edge.from], remap[edge.to]) += 
                                beta * weight;
                            O.at<double>(remap[edge.from], remap[edge.from]) += 
                                alpha * weight;
                            R.at<double>(remap[edge.from]) += 
                                2 * weight * edge.value.dphi;

                            error += weight * abs(edge.value.dphi);
                            weightSum += weight;
                            edgeCount++;

                            allEdges.push_back(edge);
                        }
                    }
                }

                cout << "Error: " << (error / relations.GetEdges().size()) << endl;
                
                solve(O, R, x, DECOMP_SVD);
                
                assert((int)invmap.size() == n);

                for (int i = 0; i < n; ++i) {
                    this->alignmentCorrections[invmap[i]] = x.at<double>(i, 0);
                    //cout << invmap[i] << " alignment: " << x.at<double>(i, 0) << endl;
                }

                return allEdges;
            }

            void Apply(InputImageP in) const {
                Mat bias(4, 4, CV_64F);

                if(alignmentCorrections.find(in->id) != alignmentCorrections.end()) {
                    CreateRotationY(alignmentCorrections.at(in->id), bias);
                    in->adjustedExtrinsics = bias * in->adjustedExtrinsics; 
                }
            }
    };

}

#endif
