#include <iostream>
#include <algorithm>
#include <memory>
#include <chrono>
#include <thread>
#include <opencv2/stitching/detail/blenders.hpp>
#include <opencv2/core/ocl.hpp>
#include <srba.h>
#include <mrpt/graphslam.h> 

#include "common/intrinsics.hpp"
#include "common/static_timer.hpp"
#include "io/io.hpp"
#include "recorder/recorder.hpp"
#include "math/projection.hpp"
#include "stitcher/simpleSphereStitcher.hpp"
#include "stitcher/simplePlaneStitcher.hpp"
#include "imgproc/pairwiseVisualAligner.hpp"
#include "debug/visualDebugHook.hpp"

using namespace std;
using namespace cv;
using namespace optonaut;
using namespace std::chrono;
using namespace srba;

struct RBA_OPTIONS : public RBA_OPTIONS_DEFAULT {
    typedef options::observation_noise_identity      obs_noise_matrix_t;
    typedef options::sensor_pose_on_robot_none       sensor_pose_on_robot_t;
};

typedef RbaEngine<
    kf2kf_poses::SE3,              // Parameterization  of KF-to-KF poses
    landmarks::Euclidean3D,        // Parameterization of landmark positions
    observations::MonocularCamera, // Type of observations
    RBA_OPTIONS
>  OptonautSRBA;

bool CompareByFilename (const string &a, const string &b) {
    return IdFromFileName(a) < IdFromFileName(b);
}

PairwiseVisualAligner aligner;

void MatchImages(const InputImageP &a, const InputImageP &b) {
   aligner.FindCorrespondence(a, b); 
}

void FinishImage(const InputImageP) { }

int main(int argc, char** argv) {
    cv::ocl::setUseOpenCL(false);
    VisualDebugHook debugger;

    int n = argc - 1;
    vector<string> files;

    for(int i = 0; i < n; i++) {
        string imageName(argv[i + 1]);
        files.push_back(imageName);
    }

    std::sort(files.begin(), files.end(), CompareByFilename);
    
    if(n > 1) {
        n = 500;
    }
        
    auto base = Recorder::iosBase;
    auto zero = Recorder::iosZero;
    auto baseInv = base.t();

    RingProcessor<InputImageP> combiner(1, &MatchImages, &FinishImage); 

    Mat scaledK;

    for(int i = 200; i < n - 1; i += 5) {
        auto img = InputImageFromFile(files[i], false);
        pyrDown(img->image.data, img->image.data);
        img->originalExtrinsics = base * zero * img->originalExtrinsics.inv() * baseInv;
        combiner.Push(img);

        if(scaledK.cols == 0) {
	        ScaleIntrinsicsToImage(img->intrinsics, img->image.size(), scaledK);
        }
    }

    OptonautSRBA srba;
    //Set to 2 for verbose
    srba.setVerbosityLevel(2);
    
    srba.parameters.srba.use_robust_kernel = true;
    srba.parameters.obs_noise.std_noise_observations = 0.5; 
    
    srba.parameters.srba.max_tree_depth = 3;
    srba.parameters.srba.max_optimize_depth = 3;

    

    mrpt::utils::TCamera & c = srba.parameters.sensor.camera_calib;
    c.ncols = scaledK.at<double>(0, 2) * 2;
    c.nrows = scaledK.at<double>(1, 2) * 2;
    c.cx(scaledK.at<double>(0, 2));
    c.cy(scaledK.at<double>(1, 2));
    c.fx(scaledK.at<double>(0, 0));
    c.fy(scaledK.at<double>(1, 1));
    c.dist.setZero();

    const vector<ImageFeatures> &features = aligner.GetFeatures();
    const vector<vector<FeatureChainInfo>> &chains = aligner.GetFeatureChains();
    const vector<vector<size_t>> &chainRefs = aligner.GetChainRefs();
        
    for(size_t i = 0; i < features.size(); i++) {
        InputImageP img = aligner.GetImageById(i); 
        vector<KeyPoint> keypoints = features[i].keypoints;

        OptonautSRBA::new_kf_observations_t  observations;
        OptonautSRBA::new_kf_observation_t   observation;

        observation.is_fixed = false; 
        observation.is_unknown_with_init_val = false;

        for(size_t j = 0; j < keypoints.size(); j++) {
            size_t globalFeatureId = chainRefs[i][j];
            if(globalFeatureId != aligner.NO_CHAIN && chains[globalFeatureId].size() > 1) {
                //Add observation. 
                observation.obs.feat_id = globalFeatureId;
                observation.obs.obs_data.px.x = keypoints[j].pt.x;
                observation.obs.obs_data.px.y = keypoints[j].pt.y;
                observations.push_back(observation);
            }
        }
        OptonautSRBA::TNewKeyFrameInfo newKeyframe;
        srba.define_new_keyframe(
        observations,      // Input observations for the new KF
        newKeyframe,   // Output info
        true           // Also run local optimization?
        );

        observations.clear();
/*
        cout << "Created KF #" << newKeyframe.kf_id
        << " | # kf-to-kf edges created:" <<  newKeyframe.created_edge_ids.size()  << endl
        << "Optimization error: " << newKeyframe.optimize_results.total_sqr_error_init << " -> " << newKeyframe.optimize_results.total_sqr_error_final << endl
        << "-------------------------------------------------------" << endl;
        */
    }
    /*int i = 0;
    for(auto chain : aligner.GetFeatureChains()) {
        i++;
        if(chain.size() > 2) {
            cout << "Chain " << i << ": ";

            for(auto ref : chain) {
                cout << ref.imageId << " -> ";
            }
            cout << endl;
        }
    }*/

    OptonautSRBA::TOpenGLRepresentationOptions  opengl_options;
    mrpt::opengl::CSetOfObjectsPtr rba_3d = mrpt::opengl::CSetOfObjects::Create();

    srba.build_opengl_representation(
    0,  // Root KF,
    opengl_options, // Rendering options
    rba_3d  // Output scene 
    );

    mrpt::graphs::CNetworkOfPoses3D poseGraph;
    srba.get_global_graphslam_problem(poseGraph);

    // Run optimization:
    mrpt::graphslam::TResultInfoSpaLevMarq out_info;
    mrpt::utils::TParametersDouble extra_params;
    
    mrpt::graphslam::optimize_graph_spa_levmarq(
    poseGraph, 
    out_info,
    NULL,
    extra_params
    );

    double m = 0;

    for(auto node : poseGraph.nodes) {
        for(int i = 0; i < 3; i++) {
            if(m < node.second.m_coords[i]) {
                m = node.second.m_coords[i];
            }
        }
    }

    cout << m << endl;

    for(auto node : poseGraph.nodes) {
       double yaw, pitch, roll;
       double x = node.second.m_coords[0] / m * 3;
       double y = node.second.m_coords[1] / m * 3;
       double z = node.second.m_coords[2] / m * 3;
       node.second.getYawPitchRoll(yaw, pitch, roll);
       cout << "Rotation yaw: " << yaw << " pitch: " << pitch << " roll: " << roll << endl;
       cout << "Position x: " << x << " y: " << y << " z: " << z  << endl;
       cout << "As String: " << node.second.asString() << endl;
       debugger.PlaceFeature(x, y, z);
    }

    debugger.Draw();
    debugger.WaitForExit();

    poseGraph.saveToTextFile("test.graph");

    return 0;
}
