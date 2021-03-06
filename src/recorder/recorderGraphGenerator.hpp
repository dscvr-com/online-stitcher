#include <opencv2/opencv.hpp>
#include <math.h>

#include "../io/inputImage.hpp"
#include "../math/support.hpp"
#include "../math/projection.hpp"
#include "../common/ringProcessor.hpp"
#include "../common/bimap.hpp"
#include "../common/logger.hpp"
#include "recorderGraph.hpp"

using namespace cv;
using namespace std;

#define _USE_MATH_DEFINES

#ifndef OPTONAUT_RECORDER_GRAPH_GENERATOR_HEADER
#define OPTONAUT_RECORDER_GRAPH_GENERATOR_HEADER

namespace optonaut {

/*
 * Class to generate and modify recorder graphs.
 */
class RecorderGraphGenerator {

private:
    static const bool debug = false;

    /*
     * Inserts a new edge into the given recorder graph.
     */
    static void CreateEdge(RecorderGraph &res,
            const SelectionPoint &a, const SelectionPoint &b) {
        SelectionEdge edge;
        edge.from = a.globalId;
        edge.to = b.globalId;
        edge.recorded = false;

        res.AddEdge(edge);
    };

    /*
     * Adds a new node into the given recorder graph.
     */
    static void AddNode(RecorderGraph &res, const SelectionPoint &a) {
        res.AddNewNode(a);
    };

    /*
     * Moves all images in the given chain according to the difference between the aligned and the unaligneg image.
     */
    static void PushBy(InputImageP aligned, InputImageP unaligned, const vector<InputImageP> &chain, vector<InputImageP> &res) {
        Mat diff = aligned->adjustedExtrinsics *
            unaligned->originalExtrinsics.inv();

        for(auto img : chain) {
            img->adjustedExtrinsics = diff * img->originalExtrinsics;
            res.push_back(img);
        }
    }
public:

    /*
     * Adjusts a dense graph from the changes in a sparse graph.
     */
    static vector<InputImageP> AdjustFromSparse(
            vector<InputImageP> sparseImages,
            const RecorderGraph &sparse,
            const BiMap<size_t, uint32_t> &sparseImagesToTargets,
            vector<InputImageP> denseImages,
            const RecorderGraph &dense,
            const BiMap<size_t, uint32_t> &denseImagesToTargets,
            const BiMap<uint32_t, uint32_t> &) {

        vector<InputImageP> res;


        auto sortById = [](const InputImageP &a, const InputImageP &b) {
            return a->id < b->id;
        };

        std::sort(sparseImages.begin(), sparseImages.end(), sortById);
        std::sort(denseImages.begin(), denseImages.end(), sortById);

        size_t dj = 0;
        vector<InputImageP> prePreChain;

        while(dj < denseImages.size() &&
                denseImages[dj]->id != sparseImages[0]->id) {
            prePreChain.push_back(denseImages[dj]);
            dj++;
        }

        if(prePreChain.size() != 0) {
            PushBy(sparseImages.front(), prePreChain.back(), prePreChain, res);
        }

        for(size_t sj = 0; sj < sparseImages.size() - 1; sj++) {
            //note:mca neighboring image
            auto start = sparseImages[sj];
            auto end = sparseImages[sj + 1];

            auto tStart = TargetFromImage(sparse, sparseImagesToTargets, start->id);
            auto tEnd = TargetFromImage(sparse, sparseImagesToTargets, end->id);

            if(tStart.ringId == tEnd.ringId) {
                // Case 1: start/end are on the same ring. Just use slerp.
                // spherical liner interpolation
                // any point on the curve must be a linear combination of the ends
                vector<InputImageP> denseChain;

                while(dj < denseImages.size() &&
                        denseImages[dj]->id != end->id) {
                    denseChain.push_back(denseImages[dj]);
                    AssertGTM(end->id, denseImages[dj]->id,
                            "Did not find sparse image in dense");
                    dj++;
                }
                denseChain.push_back(denseImages[dj]);

                size_t n = denseChain.size();

                for(size_t k = 0; k < n; k++) {
                    Slerp(start->adjustedExtrinsics,
                            end->adjustedExtrinsics,
                            (double)k / (double)(n - 1),
                            denseChain[k]->adjustedExtrinsics);

                    res.push_back(denseChain[k]);
                }
            } else {
                // Case 2: start/end are not on the same ring. Can't slerp.
                // Split chain into pre and post, then move images according to diff.
                vector<InputImageP> preChain;
                vector<InputImageP> postChain;

                while(dj < denseImages.size() &&
                        denseImages[dj]->id != end->id) {

                    AssertGTM(end->id, denseImages[dj]->id,
                            "Did not find sparse image in dense");

                    auto tDense = TargetFromImage(dense, denseImagesToTargets,
                            denseImages[dj]->id);

                    if(tDense.ringId == tStart.ringId) {
                        preChain.push_back(denseImages[dj]);
                    } else {
                        AssertEQM(tDense.ringId, tEnd.ringId,
                                "Image belongs either to start or end part.");
                        postChain.push_back(denseImages[dj]);
                    }
                    dj++;
                }

                if(preChain.size() > 0) {
                    PushBy(start, preChain.back(), preChain, res);
                }

                if(postChain.size() > 0) {
                    PushBy(end, postChain.front(), postChain, res);
                }
            }
        }

        vector<InputImageP> postPostChain;

        while(dj < denseImages.size()) {
            postPostChain.push_back(denseImages[dj]);
            dj++;
        }

        if(postPostChain.size() != 0) {
            PushBy(sparseImages.back(), postPostChain.front(), postPostChain, res);
        }

        return res;

        //Attempt #2 - use graph structure to align images.
        /*
        for(auto ring : sparse.GetRings()) {
            for(auto cur : ring) {
                SelectionPoint next;
                Assert(sparse.GetNext(cur, next));

                auto imgA = ImageFromTarget(sparseImages,
                        sparseImagesToTargets, cur.globalId);
                auto imgB = ImageFromTarget(sparseImages,
                        sparseImagesToTargets, next.globalId);

                SelectionPoint tdA = TargetFromImage(dense,
                        denseImagesToTargets, imgA->id);
                SelectionPoint tdB = TargetFromImage(dense,
                        denseImagesToTargets, imgB->id);

                vector<SelectionPoint> chain;
                chain.push_back(tdA);

                while(chain.back().globalId != tdB.globalId) {
                    SelectionPoint next;
                    Assert(dense.GetNext(chain.back(), next));
                    chain.push_back(next);
                }

                chain.push_back(tdB);

                // For each pair (cur, next) in sparse
                // 1) Get corresponding targets in dense. Done.
                // 2) Find targets in between. Done.
                // 3) Interpolate images in between.
            }
        }
        */
    }

    /*
     * Convenience. Gets the selection point associated with a given image id for the given recorder graph.
     */
    static SelectionPoint TargetFromImage(const RecorderGraph &in,
            const BiMap<size_t, uint32_t> &imagesToTargets,
            size_t imageId) {
        uint32_t pid;
        SelectionPoint p;

        Assert(imagesToTargets.GetValue(imageId, pid));
        Assert(in.GetPointById(pid, p));

        return p;
    }

    /*
     * Convenience. Gets the image associated with a given selection point.
     */
    static InputImageP ImageFromTarget(const vector<InputImageP> &in,
            const BiMap<size_t, uint32_t> &imagesToTargets,
            uint32_t pid) {
        size_t id;

        Assert(imagesToTargets.GetKey(pid, id));
        auto it = std::find_if(in.begin(), in.end(), [id](const InputImageP &img) {
                        return (int)id == img->id;
                    });

        Assert(it != in.end());

        return *it;
    }

    /*
     * Creates a sparse recorder graph from a dense recorder graph.
     *
     * Skip gives the amount of selection points to skip from the dense graph.
     */
    static RecorderGraph Sparse(const RecorderGraph &in, int skip, int ring = -1) {
        BiMap<size_t, uint32_t> x;
        BiMap<size_t, uint32_t> y;
        BiMap<uint32_t, uint32_t> z;

        return Sparse(in, skip, x, y, z, ring);
    }

    /*
     * Creates a sparse recorder graph from a dense recorder graph.
     *
     * Skip gives the amount of selection points to skip from the dense graph.
     */
    static RecorderGraph Sparse(const RecorderGraph &in, int skip,
            const BiMap<size_t, uint32_t> &denseImages,
            BiMap<size_t, uint32_t> &sparseImagesToTargets,
            BiMap<uint32_t, uint32_t> &denseToSparse,
            int ring = -1) {

        int ringCount = ring < 0 ? in.ringCount : 1;
        size_t startRing = ring < 0 ? 0 : ring;
        size_t endRing = ring < 0 ? in.ringCount : ring + 1;

        RecorderGraph sparse(ringCount, in.intrinsics);

        size_t globalId = 0;
        size_t localId = 0;

        const auto &rings = in.GetRings();

        AssertGT(rings.size(), startRing);
        AssertGE(startRing, (size_t)0);
        AssertGE(rings.size(), endRing);

        for(size_t i = startRing; i < endRing; i++) {

            int newRingSize = (rings[i].size()) / skip;

            AssertEQM(newRingSize * skip, (int)rings[i].size(),
                    "Ring size divisible by skip factor");

            AssertEQ((uint32_t)(i - startRing), sparse.AddNewRing(newRingSize));

            RingProcessor<SelectionPoint> hqueue(1,
                    bind(CreateEdge, std::ref(sparse),
                        placeholders::_1, placeholders::_2),
                    bind(AddNode, std::ref(sparse), placeholders::_1));

            localId = 0;
            int c = 0;

            for(size_t j = 0; j < rings[i].size(); j += skip) {
                SelectionPoint copy = rings[i][j];
                size_t imageId = 0;

                if(denseImages.Size() > 0) {
                    Assert(denseImages.GetKey(copy.globalId, imageId));

                    denseToSparse.Insert(copy.globalId, globalId);
                    sparseImagesToTargets.Insert(imageId, globalId);
                }

                copy.ringId = i - startRing;
                copy.localId = localId;
                copy.globalId = globalId;
                copy.ringSize = newRingSize;
                copy.hFov *= skip;
                hqueue.Push(copy);

                localId++;
                globalId++;
                c++;
            }

            AssertEQM(c, newRingSize, "Size calculation was correct");

            hqueue.Flush();
        }

        return sparse;
    }
    
    /*
     * Generates a new recorder graph, based on the given parameters.
     *
     * @param intrinsics The intrinsics to base the recorder graph on.
     * @param mode The recorder graph generation mode. Please see the constants defined in RecorderGraph.
     * @param density The density of the recorder graph. 1 is normal. 2 leads to twice as many selection points.
     * @param lastRingOverdrive Additional edges to add to the last ring, so we record extra.
     * @param divider The ring size is guaranteed to be divisible by this divider. Useful if a sparse graph should be generated from this graph.
     * @param hOverlap The horizontal extra space for each image, left and right. Take note that this parameter is not reliable, since it happens *before* focal length estimation.
     * @param vOverlap The vertical extra space for each image, bottom and top.
     */
    static RecorderGraph Generate(const Mat &intrinsics, const int mode = RecorderGraph::ModeAll, const float density = RecorderGraph::DensityNormal, const int lastRingOverdrive = 0, const int divider = 1, double hOverlap_ = 0.7, double vOverlap_ = 0.25) {
        AssertWGEM(density, 1.f,
                "Reducing recorder graph density below one is potentially unsafe.");

        double hOverlap = 1.0 - (1.0 - hOverlap_) / density;
        double vOverlap = vOverlap_;

        double maxHFov = GetHorizontalFov(intrinsics);
        double maxVFov = GetVerticalFov(intrinsics);
				double hFov = maxHFov * (1.0 - hOverlap);
				double vFov = maxVFov * (1.0 - vOverlap);

        Log << "H-FOV: " << (maxHFov * 180 / M_PI);
        Log << "V-FOV: " << (maxVFov * 180 / M_PI);
        Log << "Ratio: " << (sin(maxVFov) / sin(maxHFov));

        uint32_t vCount = ceil(M_PI / vFov);
        double vStart = 0;
        uint32_t id = 0;
        uint32_t hCenterCount = ceil(2 * M_PI / hFov);

        //TODO: Wrong assumptions.
        //This actually builds the recorder graph from the bottom.

        if(mode == RecorderGraph::ModeTruncated || mode == RecorderGraph::ModeTinyDebug) {
           //Configuration for ModeTruncated
           //Optimize for 3 rings.

           vCount = 3;
           //vFov stays the same.
           vStart = (M_PI - (vFov * 3)) * (1.f / 2.f);

        }
        else if(mode == RecorderGraph::ModeNoBot) {
            //Configuration for ModeNoBot
            //vCount = vCount - 1;
            vCount = vCount - 1;
            vStart = M_PI - (vFov * vCount);

        } else if(mode == RecorderGraph::ModeCenter) {
            vCount = 1;
            vStart = M_PI / 2 - (vFov / 2);
        } else {
            //Configuration for ModeAll
            vStart = maxVFov * vOverlap;
            vFov = (M_PI - 2 * vStart) / vCount;
        }

        if(vCount % 2 == 0 && mode == RecorderGraph::ModeCenter) {
            AssertM(false, "Center mode not possible with even number of rings." );
        }

        RecorderGraph res(vCount, intrinsics);

		for(uint32_t i = 0; i < vCount; i++) {

            //Vertical center, bottom and top of ring
			double vCenter = i * vFov + vFov / 2.0 - M_PI / 2.0 + vStart;
			res.vCenterList.push_back(vCenter);

			uint32_t hCount = hCenterCount * cos(vCenter);

            if(hCount % divider != 0) {
                hCount += divider - ((hCount) % divider);
            }
			hFov = M_PI * 2 / hCount;

            if(mode ==  RecorderGraph::ModeTinyDebug) {
                hCount = 4 * divider;
            }

            double hLeft = 0;
            SelectionEdge edge;

            int ringOverdrive = 0;

            if(i == vCount -1) {
                ringOverdrive = lastRingOverdrive;
            }

            hCount = hCount + ringOverdrive;
            AssertEQ(i, res.AddNewRing(hCount));

            RingProcessor<SelectionPoint> hqueue(1,
                    bind(CreateEdge, std::ref(res), placeholders::_1, placeholders::_2),
                    bind(AddNode, std::ref(res), placeholders::_1));

            for(uint32_t j = 0; j < hCount; j++) {
                if(debug) {
                    cout << "Recorder Graph Pushing " << hLeft << ", " << vCenter << endl;
                }

                hLeft = j * hFov;

                SelectionPoint p;
                p.globalId = id;
                p.hPos = hLeft;
                p.vPos = vCenter;
                p.ringId = i;
                p.localId = j;
                p.vFov = maxVFov;
                p.hFov = hFov;
                p.ringSize = hCount;

                GeoToRot(hLeft, vCenter, p.extrinsics);

                hqueue.Push(p);

                id++;
            }

            hqueue.Flush();

		}

        return res;
	}
};
}

#endif
