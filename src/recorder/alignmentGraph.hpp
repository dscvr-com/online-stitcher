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

    /*
     * Represents a horizontal alignment difference between two images. 
     */
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
    

    /*
     * Calculates and holds a set of alignment differences for a set of images.
     * 
     * The alignment differences are hold in a big graph. The graph can be converted to a linear
     * equation system, which, when solved for the minimal error, provides an optimal solution for 
     * minimizing the alignment errors. 
     *
     * It is also possible to insert artificial adges into the graph, for example for damping. 
     */
    class AlignmentGraph: public ImageCorrespondenceGraph<AlignmentDiff> {
        private: 
            PairwiseCorrelator aligner;
            const RecorderGraph &graph; 
            const BiMap<size_t, uint32_t> &imagesToTargets;
            static const bool debug = true;
            map<size_t, double> alignmentCorrections;
        public:

            /*
             * Creates a new alignment graph based on the given recorder graph. 
             */
            AlignmentGraph(const RecorderGraph &graph, 
                    const BiMap<size_t, uint32_t> &imagesToTargets) : 
                graph(graph), 
                imagesToTargets(imagesToTargets)
            { }

            /*
             * Copy constructor. 
             */
            AlignmentGraph(AlignmentGraph &ref) : 
                graph(ref.graph), 
                imagesToTargets(ref.imagesToTargets)
            {
                SetAlignment(ref.GetAlignment());
            }
       
            /*
             * Sets given alignment offsets.
             */ 
            void SetAlignment(map<size_t, double> alignment)
            {
                this->alignmentCorrections = alignment;
            }

            /*
             * Gets the alignment offsets associated with this graph. 
             */
            const map<size_t, double>& GetAlignment() const {
                return alignmentCorrections;
            }
       
            /*
             * Calculates the alignment difference for two given images. 
             */ 
            virtual AlignmentDiff GetCorrespondence(InputImageP imgA, InputImageP imgB, AlignmentDiff &aToB, AlignmentDiff &bToA) {

                STimer tFindCorrespondence;

                const bool dampUncorrelatedNeighbors = true;
                const bool dampAllNeighbors = false;
                const bool dampSuccessors = true;

                uint32_t pidA = 0, pidB = 0;
                SelectionPoint tA, tB;

                Assert(imagesToTargets.GetValue(imgA->id, pidA));
                Assert(imagesToTargets.GetValue(imgB->id, pidB));

                Assert(graph.GetPointById(pidA, tA));
                Assert(graph.GetPointById(pidB, tB));

                // Dist: Distance, in steps, between the selection points. 
                int dist = min(abs((int)tA.localId - (int)tB.localId),
                       (int)tA.ringSize - abs((int)tA.localId - (int)tB.localId));

                dist = dist % tA.ringSize;

                bool areNeighbors = 
                    dist <= 1
                    && tA.ringId == tB.ringId;

                bool areSuccessors = areNeighbors && 
                    imgA->id < imgB->id;
                        
                Mat rvec;
                ExtractRotationVector(imgA->adjustedExtrinsics.inv() * 
                        imgB->adjustedExtrinsics, rvec);

                double angularDist = rvec.at<double>(1);
                double maxDist = dist * (M_PI * 2 / tA.ringSize);
                double angularDiff = maxDist - angularDist; 

                // If these are neighbors or we are on different rings, do try to match the images. 
                if(tA.ringId != tB.ringId 
                        || areNeighbors) {
                    int minSize = min(imgA->image.cols, imgA->image.rows) / 1.8;

                    STimer tMatch;
                    auto res = aligner.Match(imgB, imgA, minSize, minSize);
                    tMatch.Tick("Matching");

                    // If our match was invalid, just do damping it. 
                    if(!res.valid && areNeighbors && dampUncorrelatedNeighbors) {
                        res.valid = true;
                        res.angularOffset.y = 0;
                        res.overlap = imgA->image.cols * imgA->image.rows * 100 *
                            (1 + abs(angularDiff)) * (1 + abs(angularDiff));
                        aToB.forced = true;
                    }

                    // Copy matching information to the correspondence info
                    aToB.dphi = res.angularOffset.y;
                    aToB.overlap = res.correlationCoefficient * 2;
                    aToB.valid = res.valid;
                    aToB.rejectionReason = res.rejectionReason;
                } else {
                    aToB.dphi = 0;
                    aToB.overlap = 0;
                    aToB.valid = false;
                }
                   
                // Copy correspondence info for b<>a from a<>b since it is guaranteed to be symmetrical.  
                bToA = aToB;
                aToB.dphi *= -1;

                // If configured so, add damping to neighors.  
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
                       
                    // Insert this extra correspondence.  
                    InsertCorrespondence(imgA->id, imgB->id, aToBDamp, bToADamp);
                }

                if(dampSuccessors && areSuccessors) {
                    AlignmentDiff aToBDamp, bToADamp; 
                    aToBDamp.dphi = 0; 
                    aToBDamp.overlap = imgA->image.cols * imgA->image.rows * 0.5;
                    aToBDamp.forced = true;
                    aToBDamp.valid = true;
                    bToADamp = aToBDamp;
                    bToADamp.dphi *= -1;
                       
                    // Insert this extra correspondence.  
                    InsertCorrespondence(imgA->id, imgB->id, aToBDamp, bToADamp);
                } 

                tFindCorrespondence.Tick("Find Correspondence");
               
                return aToB;
            };

            /*
             * From the alignment graph, finds a solution for re-aligning all images with minimal error. 
             */
            Edges FindAlignment(double &outError) {

                STimer tFindAlignment;
                // Pre-calculate the size of our equation system. 
                size_t maxId = 0;
                Edges allEdges;

                for(auto &adj : relations.GetEdges()) {
                    maxId = max(maxId, adj.first);
                }
               
                // Build forward/backward lookup tables.  
                vector<int> remap(maxId);
                vector<int> invmap;
                
                for(auto &adj : relations.GetEdges()) {
                    remap[adj.first] = (int)invmap.size();
                    invmap.push_back((int)adj.first);
                }

                int n = (int)invmap.size();

                Mat res = Mat::zeros(n, 1, CV_64F);
               
                // Build equation systen
                // R = sum of relative offsets.
                // O = matrix of observation weights.
                // x = result, optimal relative offset. 
                Mat R, O, x;
                R = Mat::zeros(n, 1, CV_64F);
                O = Mat::zeros(n, n, CV_64F);
                x = Mat::zeros(n, 1, CV_64F);

                // Alpha - damping for our regression. 
                double alpha = 2;
                double beta = 1 / alpha;
                double error = 0;
                double weightSum = 0;
                double edgeCount = 0;

                // Quartil for selecting edges that are used when optimizing. 
                const double quartil = 0.1;

                for(auto &adj : relations.GetEdges()) { // For adjacency list each node in our graph...
                    // Find all edges created by the aligner. 
                    Edges correlatorEdges = fun::filter<Edge>(adj.second, 
                        [] (const Edge &a) {
                            return !a.value.forced && a.value.overlap > 0
                            && a.value.valid; 
                        }); 

                    // Find all adges manually created.  
                    Edges forcedEdges = fun::filter<Edge>(adj.second, 
                        [] (const Edge & a) {
                            return a.value.forced && a.value.overlap > 0 
                            && a.value.valid; 
                        }); 

                    //Reject upper quartil of correlator edges only and sort by absolute delta. 
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
                            // If the edge is not in the quartil, add it to the forced edges.
                            forcedEdges.push_back(correlatorEdges[i]);
                        }
                    }
                    
                    // For all edges we decide to work with...
                    for(auto &edge : forcedEdges) {

                        Assert(!edge.value.quartil);

                        if(edge.value.overlap > 0) {
                            double weight = edge.value.overlap;

                            // Add values to equation system. 
                            O.at<double>(remap[edge.from], remap[edge.to]) += 
                                beta * weight;
                            O.at<double>(remap[edge.from], remap[edge.from]) += 
                                alpha * weight;
                            R.at<double>(remap[edge.from]) += 
                                2 * weight * edge.value.dphi / 2;

                            //Use non-linear weights - less penalty for less overlap. 
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
          
                // Solve the equation system for minimal error.  
                solve(O, R, x, DECOMP_SVD);

                res = res + x;

                // Extract solution from equation system. 
                for(auto &adj : relations.GetEdges()) {
                   for(auto &edge : adj.second) {
                        edge.value.dphi -= x.at<double>(remap[edge.from]);
                        edge.value.dphi += x.at<double>(remap[edge.to]);
                   } 
                }
                
                assert((int)invmap.size() == n);

                // Remember the alignment corrections. 
                for (int i = 0; i < n; ++i) {
                    this->alignmentCorrections[invmap[i]] = res.at<double>(i, 0);
                    //cout << invmap[i] << " alignment: " << x.at<double>(i, 0) << endl;
                }

                tFindAlignment.Tick("Find Alignment");

                return allEdges;
            }

            /*
             * Applies the calculated alignment correction to the submitted image. 
             */
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
