#include <opencv2/opencv.hpp>
#include <mutex>

#include "../common/imageCorrespondenceGraph.hpp"
#include "../common/support.hpp"
#include "../common/functional.hpp"
#include "../math/projection.hpp"
#include "../imgproc/pairwiseCorrelator.hpp"
#include "../recorder/recorderGraph.hpp"

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
            const RecorderGraph &graph; 
            const BiMap<size_t, uint32_t> &imagesToTargets;
            static const bool debug = false;
            map<size_t, double> alignmentCorrections;
        public:
            AlignmentGraph(const RecorderGraph &graph, 
                    const BiMap<size_t, uint32_t> &imagesToTargets) : 
                graph(graph), 
                imagesToTargets(imagesToTargets)
            { }
            AlignmentGraph(AlignmentGraph &ref) : 
                graph(ref.graph), 
                imagesToTargets(ref.imagesToTargets)
            {
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

                const bool dampUncorrelatedNeighbors = true;
                const bool dampAllNeighbors = false;

                uint32_t pidA, pidB;
                SelectionPoint tA, tB;

                Assert(imagesToTargets.GetValue(imgA->id, pidA));
                Assert(imagesToTargets.GetValue(imgB->id, pidB));

                Assert(graph.GetPointById(pidA, tA));
                Assert(graph.GetPointById(pidB, tB));

                int dist = min(abs((int)tA.localId - (int)tB.localId),
                       (int)tA.ringSize - abs((int)tA.localId - (int)tB.localId));

                dist = dist % tA.ringSize;

                bool areNeighbors = 
                    dist <= 1
                    && tA.ringId == tB.ringId;
                        
                Mat rvec;
                ExtractRotationVector(imgA->adjustedExtrinsics.inv() * 
                        imgB->adjustedExtrinsics, rvec);

                double angularDist = rvec.at<double>(1);
                double maxDist = dist * (M_PI * 2 / tA.ringSize);
                double angularDiff = maxDist - angularDist; 

                if(tA.ringId != tB.ringId 
                        || areNeighbors) {
                    int minSize = min(imgA->image.cols, imgA->image.rows) / 1.8;

                    auto res = aligner.Match(imgB, imgA, minSize, minSize);

                    if(!res.valid && areNeighbors && dampUncorrelatedNeighbors) {
                        res.valid = true;
                        res.angularOffset.x = 0;
                        res.overlap = imgA->image.cols * imgA->image.rows * 100 *
                            (1 + abs(angularDiff)) * (1 + abs(angularDiff));
                        aToB.forced = true;
                    }

                    aToB.dphi = res.angularOffset.x;
                    aToB.overlap = res.correlationCoefficient;
                    aToB.valid = res.valid;
                    aToB.rejectionReason = res.rejectionReason;
                    aToB.error = res.inverseTestDifference.x;
                } else {
                    aToB.dphi = 0;
                    aToB.overlap = 0;
                    aToB.valid = false;
                }
                    
                bToA = aToB;
                aToB.dphi *= -1;

                if(dampAllNeighbors && areNeighbors) {

                    AssertM(false, "All neighbor damping implementation possibly incorrect - dphi is relative, also we should only damp successively recorded images");

                    AlignmentDiff aToBDamp, bToADamp; 
                    aToBDamp.dphi = -angularDiff / 8; 
                    aToBDamp.overlap = imgA->image.cols * imgA->image.rows * 0.01 *
                            (1 + abs(angularDiff)) * (1 + abs(angularDiff));
                    aToBDamp.forced = true;
                    aToBDamp.valid = true;
                    bToADamp = aToBDamp;
                    bToADamp.dphi *= -1;

                    InsertCorrespondence(imgA->id, imgB->id, aToBDamp, bToADamp);
                }
                
                return aToB;
            };

            Edges FindAlignment(double &outError) {
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

                Mat res = Mat::zeros(n, 1, CV_64F);
                
                for(int i = 0; i < 1; i++) { 
                    //Build equation systen
                    //R = sum of relative offsets.
                    //O = matrix of observation weights.
                    //x = result, optimal relative offset. 
                    Mat R, O, x;
                    R = Mat::zeros(n, 1, CV_64F);
                    O = Mat::zeros(n, n, CV_64F);
                    x = Mat::zeros(n, 1, CV_64F);

                    //Alpha - damping for our regression. 
                    double alpha = 2;
                    double beta = 1 / alpha;
                    double error = 0;
                    double weightSum = 0;
                    double edgeCount = 0;

                    const double quartil = 0.1;

                    for(auto &adj : relations.GetEdges()) {

                        Edges correlatorEdges = fun::filter<Edge>(adj.second, 
                            [] (const Edge &a) {
                                return !a.value.forced && a.value.overlap > 0
                                && a.value.valid; 
                            }); 

                        Edges forcedEdges = fun::filter<Edge>(adj.second, 
                            [] (const Edge & a) {
                                return a.value.forced && a.value.overlap > 0 
                                && a.value.valid; 
                            }); 

                        //Reject upper quartil only and sort by absolute delta. 
                        std::sort(correlatorEdges.begin(), correlatorEdges.end(), 
                            [] (const Edge &a, const Edge &b) {
                                return abs(a.value.dphi) < abs(b.value.dphi);
                            });

                        size_t m = correlatorEdges.size();

                        for(size_t i = 0; i < m; i++) {
                            if(i >= (double)m * (1.0 - quartil)) {
                                correlatorEdges[i].value.quartil = true;
                                allEdges.push_back(correlatorEdges[i]);
                            } else {
                                forcedEdges.push_back(correlatorEdges[i]);
                            }
                        }
                        
                        for(auto &edge : forcedEdges) {
                            //Use non-linear weights - less penalty for less overlap. 
                            assert(!edge.value.quartil);
                            if(edge.value.overlap > 0) {
                                double weight = edge.value.overlap;

                                O.at<double>(remap[edge.from], remap[edge.to]) += 
                                    beta * weight;
                                O.at<double>(remap[edge.from], remap[edge.from]) += 
                                    alpha * weight;
                                R.at<double>(remap[edge.from]) += 
                                    2 * weight * edge.value.dphi;

                                if(!edge.value.forced) {
                                    error += weight * abs(edge.value.dphi);
                                    weightSum += weight;
                                    edgeCount++;
                                }

                                allEdges.push_back(edge);
                            }
                        }
                    }

                    outError = error / relations.GetEdges().size();
               
                    solve(O, R, x, DECOMP_SVD);

                    res = res + x;

                    for(auto &adj : relations.GetEdges()) {
                       for(auto &edge : adj.second) {
                            edge.value.dphi -= x.at<double>(remap[edge.from]);
                            edge.value.dphi += x.at<double>(remap[edge.to]);
                       } 
                    }
                }
                
                assert((int)invmap.size() == n);

                for (int i = 0; i < n; ++i) {
                    this->alignmentCorrections[invmap[i]] = res.at<double>(i, 0);
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
