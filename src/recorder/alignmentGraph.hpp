#include <opencv2/opencv.hpp>
#include <mutex>

#include "../common/imageCorrespondenceGraph.hpp"
#include "../common/support.hpp"
#include "../common/logger.hpp"
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
        int dx;
        double dtheta; //Difference for rotating around the horizontal axis
        int dy;
        double valid; //Is it valid?
        int overlap;
        int error; 
        int rejectionReason;
        bool quartil;
        bool forced;

        AlignmentDiff() : 
            dphi(0), 
            dx(0),
            dtheta(0),
            dy(0),
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
            static const bool debug = false;
            map<size_t, Point2d> alignmentCorrections;
        public:

            /*
             * Creates a new alignment graph based on the given recorder graph. 
             */
            AlignmentGraph()
            {
                AssertFalseInProduction(debug);
            }

            /*
             * Copy constructor. 
             */
            AlignmentGraph(AlignmentGraph &ref) {
                AssertFalseInProduction(debug);
                SetAlignment(ref.GetAlignment());
            }
       
            /*
             * Sets given alignment offsets.
             */ 
            void SetAlignment(map<size_t, Point2d> alignment)
            {
                this->alignmentCorrections = alignment;
            }

            /*
             * Gets the alignment offsets associated with this graph. 
             */
            const map<size_t, Point2d>& GetAlignment() const {
                return alignmentCorrections;
            }
       
            /*
             * Calculates the alignment difference for two given images. 
             */ 
            virtual AlignmentDiff GetCorrespondence(InputImageP, InputImageP, AlignmentDiff &aToB, AlignmentDiff &) {

                AssertM(false, "Alignment correspondences are to be added from outside.");
                /*
                STimer tFindCorrespondence(false);

                uint32_t pidA = 0, pidB = 0;
                SelectionPoint tA, tB;

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

                    STimer tMatch(false);
                    auto res = aligner.Match(imgB, imgA, minSize, 
                            minSize, false, 0.5);

                    tMatch.Tick("Matching");

                    // If our match was invalid, just do damping it. 
                    if(!res.valid && areNeighbors && dampUncorrelatedNeighbors) {
                        res.valid = true;
                        res.angularOffset.y = 0;
                        res.angularOffset.x = 0;
                        res.overlap = imgA->image.cols * imgA->image.rows * 100 *
                            (1 + abs(angularDiff)) * (1 + abs(angularDiff));
                        aToB.forced = true;
                    }

                    // Copy matching information to the correspondence info
                    aToB.dphi = -res.angularOffset.y;
                    if(areNeighbors) {
                        aToB.dtheta = res.angularOffset.x;
                    } else {
                        aToB.dtheta = NAN; 
                        // Only work with vertical offsets for neighbors. 
                    }
                    aToB.dx = res.offset.x;
                    aToB.dy = res.offset.y;
                    aToB.overlap = res.correlationCoefficient * 2;
                    aToB.valid = res.valid;
                    aToB.rejectionReason = res.rejectionReason;
                } else {
                    aToB.dx = 0;
                    aToB.dy = 0;
                    aToB.dphi = 0;
                    aToB.dtheta = 0;
                    aToB.overlap = 0;
                    aToB.valid = false;
                }
                   
                // Copy correspondence info for b<>a from a<>b since it is guaranteed to be symmetrical.  
                bToA = aToB;
                bToA.dphi *= -1;
                bToA.dtheta *= -1;
                bToA.dx *= -1;
                bToA.dy *= -1;
*/
                return aToB;
            };

            /*
             * Optimizes for vertical alignment. 
             */
            Edges FindAlignmentHorizontal(double &outError) {
                Mat res;
                vector<int> invmap;

                Edges edges = FindAlignmentInternal(outError,
                        [] (const AlignmentDiff &x) {
                            return x.dphi;
                        }, 
                        [] (AlignmentDiff &x, const double v) {
                            x.dphi = v;
                        }, res, invmap);
                
                // Remember the alignment corrections. 
                for (size_t i = 0; i < invmap.size(); ++i) {
                    this->alignmentCorrections[invmap[i]].y = res.at<double>(i, 0);
                    //Log << invmap[i] << " alignment y: " << res.at<double>(i, 0);
                }
                
                return edges;
            }
            
            /*
             * Optimizes for horizontal alignment. 
             */
            Edges FindAlignmentVertical(double &outError) {
                Mat res;
                vector<int> invmap;

                Edges edges = FindAlignmentInternal(outError,
                        [] (const AlignmentDiff &x) {
                            return x.dtheta;
                        }, 
                        [] (AlignmentDiff &x, const double v) {
                            x.dtheta = v;
                        }, res, invmap);
                
                // Remember the alignment corrections. 
                for (size_t i = 0; i < invmap.size(); ++i) {
                    this->alignmentCorrections[invmap[i]].x = res.at<double>(i, 0);
                    //Log << invmap[i] << " alignment x: " << res.at<double>(i, 0);
                }
                
                return edges;
            }

            /*
             * From the alignment graph, finds a solution for re-aligning all 
             * images with minimal error. 
             */
            Edges FindAlignment(double &outError) {
                double vError = 0, hError = 0;

                FindAlignmentVertical(vError);
                Edges edges = FindAlignmentHorizontal(hError);

                outError = vError + hError;

                return edges;
            }

            /*  
             * Performs alignment for a given property of the graph. 
             * While solving, the property is treated as an offset. 
             *
             * That means that the result is applied by adding it to the value. 
             *
             * @param outError The overall error of the found solution. 
             * @param extractor Property extractor function.
             * @param applier Property apply function. 
             */
            Edges FindAlignmentInternal(double &outError, 
                    std::function<double(const AlignmentDiff&)> extractor, 
                    std::function<void(AlignmentDiff&, const double)> applier, 
                    Mat &res, 
                    vector<int> &invmap) {

                STimer tFindAlignment(false);
                // Pre-calculate the size of our equation system. 
                size_t maxId = 0;
                Edges allEdges;

                for(auto &adj : relations.GetEdges()) {
                    maxId = max(maxId, adj.first);
                }
               
                // Build forward/backward lookup tables.  
                vector<int> remap(maxId + 1);
                
                for(auto &adj : relations.GetEdges()) {
                    remap[adj.first] = (int)invmap.size();
                    invmap.push_back((int)adj.first);
                }

                int n = (int)invmap.size();

                res = Mat::zeros(n, 1, CV_64F);
               
                // Build equation systen
                // R = sum of relative offsets.
                // O = matrix of observation weights.
                // x = result, optimal relative offset. 
                Mat R, O, x;
                R = Mat::zeros(n, 1, CV_64F);
                O = Mat::zeros(n, n, CV_64F);
                x = Mat::zeros(n, 1, CV_64F);

                // Alpha - damping for our regression. 
                double alpha = 1.2;
                double beta = 1 / alpha;
                double error = 0;
                double weightSum = 0;
                double edgeCount = 0;

                // Quartil for selecting edges that are used when optimizing. 
                const double quartil = 0.0;

                for(auto &adj : relations.GetEdges()) { // For adjacency list each node in our graph...
                    // Find all edges created by the aligner. 
                    Edges correlatorEdges = fun::filter<Edge>(adj.second, 
                        [] (const Edge &a) {
                            return !a.value.forced && a.value.overlap > 0
                            && a.value.valid; 
                        }); 
                    //Log << "Adding " << correlatorEdges.size() << " of " << adj.second.size();

                    // Find all adges manually created.  
                    Edges forcedEdges = fun::filter<Edge>(adj.second, 
                        [] (const Edge & a) {
                            return a.value.forced && a.value.overlap > 0 
                            && a.value.valid; 
                        }); 

                    //Reject upper quartil of correlator edges only and sort by absolute delta. 
                    std::sort(correlatorEdges.begin(), correlatorEdges.end(), 
                        [&] (const Edge &a, const Edge &b) {
                            return abs(extractor(a.value)) < abs(extractor(b.value));
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

                        //Log << "Adding edge: " << edge.to << " <> " << edge.from << ": " << extractor(edge.value);

                        Assert(!edge.value.quartil);

                        if(edge.value.overlap > 0) {
                            double weight = edge.value.overlap;

                            double value = extractor(edge.value);

                            if(value == value) {
                                // Add values to equation system. 
                                O.at<double>(remap[edge.from], remap[edge.to]) += 
                                    beta * weight;
                                O.at<double>(remap[edge.from], remap[edge.from]) += 
                                    alpha * weight;
                                R.at<double>(remap[edge.from]) += 
                                    2 * weight * value / 2;

                                if(!edge.value.forced) {
                                    error += weight * abs(extractor(edge.value));
                                    weightSum += weight;
                                    edgeCount++;
                                }

                                allEdges.push_back(edge);
                            }
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
                        applier(edge.value, 
                                extractor(edge.value) - 
                                x.at<double>(remap[edge.from]));

                        applier(edge.value, 
                                extractor(edge.value) + 
                                x.at<double>(remap[edge.to]));
                   } 
                }
                
                AssertEQ((int)invmap.size(), n);

                tFindAlignment.Tick("Find Alignment");

                return allEdges;
            }

            /*
             * Applies the calculated alignment correction to the submitted image. 
             */
            void Apply(InputImageP in) const {
                Mat ybias(4, 4, CV_64F);
               // Mat xbias(4, 4, CV_64F);

                if(alignmentCorrections.find(in->id) != alignmentCorrections.end()) {
                    CreateRotationY(alignmentCorrections.at(in->id).y, ybias);
                //    CreateRotationX(alignmentCorrections.at(in->id).x, xbias);
                    Log << "Adjusting " << in->id << " by y: " << alignmentCorrections.at(in->id).y << ", x: " << alignmentCorrections.at(in->id).x;
                    in->adjustedExtrinsics = in->adjustedExtrinsics * ybias;// * xbias; 
                }
            }
    };

}

#endif
