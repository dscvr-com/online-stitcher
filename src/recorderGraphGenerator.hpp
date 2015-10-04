#include <opencv2/opencv.hpp>
#include <math.h>

#include "image.hpp"
#include "support.hpp"
#include "recorderGraph.hpp"

using namespace cv;
using namespace std;

#define _USE_MATH_DEFINES

#ifndef OPTONAUT_IMAGE_SELECTOR_HEADER
#define OPTONAUT_IMAGE_SELECTOR_HEADER

namespace optonaut {

class RecorderGraphGenerator {

private:
	//adj[n] contains m if m is right of n
	//Horizontal and Vertical overlap in percent. 
	const double hOverlap = 0.9;
	const double vOverlap = 0.25;
    const double hBufferRatio = 1;
    const double vBufferRatio = 0.05;
    
    Mat intrinsics;

    void GeoToRot(double hAngle, double vAngle, Mat &res) {
        Mat hRot;
        Mat vRot;
        
        //cout << hAngle << ", " << vAngle << endl;
        
        CreateRotationY(hAngle, hRot);
        CreateRotationX(vAngle, vRot);
        
        res = hRot * vRot;
    }
    
    ImageP CreateDebugImage(Mat pos, double scale, Scalar color) const {
        ImageP d(new Image());
        
        d->originalExtrinsics = pos;
        d->adjustedExtrinsics = pos;
        d->intrinsics = intrinsics.clone();
        d->intrinsics.at<double>(0, 2) *= scale;
        d->intrinsics.at<double>(1, 2) *= scale;
        d->img = Mat::zeros(240, 320, CV_8UC3);
        d->id = 0;
        
        line(d->img, Point2f(0, 0), Point2f(320, 240), color, 4);
        line(d->img, Point2f(320, 0), Point2f(0, 240), color, 4);
        
        return d;
    }
    

public:
    
    RecorderGraph Generate(const Mat &intrinsics, const int mode = RecorderGraph::ModeAll) {
        
        RecorderGraph res;

		this->intrinsics = intrinsics;
        double maxHFov = GetHorizontalFov(intrinsics);
        double maxVFov = GetVerticalFov(intrinsics); 
		double hFov = maxHFov * (1.0 - hOverlap);
		double vFov = maxVFov * (1.0 - vOverlap);

        cout << "H-FOV: " << (maxHFov * 180 / M_PI) << endl;
        cout << "V-FOV: " << (maxVFov * 180 / M_PI) << endl;
        cout << "Ratio: " << (sin(maxVFov) / sin(maxHFov)) << endl;

        uint32_t vCount = ceil(M_PI / vFov);
        double vStart, vBuffer;
        uint32_t id = 0;
        uint32_t hCenterCount = ceil(2 * M_PI / hFov);
        
        //TODO: Wrong assumptions.
        //This actually builds the recorder graph from the bottom. 

        if(mode == RecorderGraph::ModeTruncated) {
           //Configuration for ModeTruncated
           //Optimize for 3 rings.
           
           vCount = vCount - 2; //Always skip out two rings.
           //vFov stays the same.
           vStart = (M_PI - (vFov * 3)) / 2;
           vBuffer = vFov * vBufferRatio;
            
        }
        else if(mode == RecorderGraph::ModeNoBot) {
            //Configuration for ModeNoBot
            vCount = vCount - 1;
            vStart = (M_PI - (vFov * 4)) / 2;
            vBuffer = vFov * vBufferRatio;
            
        } else if(mode == RecorderGraph::ModeCenter) {
            assert(true); //Not implemented and probably not needed any more.
        } else {
            //Configuration for ModeAll
            vStart = maxVFov * vOverlap;
            
            vFov = (M_PI - 2 * vStart) / vCount;
            vBuffer = vFov * vBufferRatio;
        }

        if(vCount % 2 == 0 && mode == RecorderGraph::ModeCenter) {
            cout << "Center mode not possible with even number of rings." << endl;
            assert(false);
        }

		for(uint32_t i = 0; i < vCount; i++) {

            //Vertical center, bottom and top of ring
			double vCenter = i * vFov + vFov / 2.0 - M_PI / 2.0 + vStart;
            double vTop = vCenter - vFov / 2.0;
            double vBot = vCenter + vFov / 2.0;

			uint32_t hCount = hCenterCount * cos(vCenter);
			hFov = M_PI * 2 / hCount;
            double hBuffer = hFov * hBufferRatio;

			uint32_t firstId = 0;
            double hLeft = 0;
            double hCenter = 0;
            double hRight = 0;
            SelectionEdge edge;

			res.targets.push_back(vector<SelectionPoint>());
                    
            for(uint32_t j = 0; j < hCount; j++) {
                
                hLeft = j * hFov;
                hCenter = hLeft + hFov / 2;
                hRight = hLeft + hFov;
                
                SelectionPoint p;
                p.globalId = id;
                GeoToRot(hLeft, vCenter, p.extrinsics);
                p.enabled = true;
                p.ringId = i;
                p.localId = j;
                
                res.adj.push_back(vector<SelectionEdge>());
                
                SelectionEdge edge;
                edge.from = id;
                edge.to = id + 1;
                edge.roiCenter = p.extrinsics;
                GeoToRot(hCenter, vCenter, edge.roiCenter);
                GeoToRot(hLeft - hBuffer, vTop - vBuffer, edge.roiCorners[0]);
                GeoToRot(hRight + hBuffer, vTop - vBuffer, edge.roiCorners[1]);
                GeoToRot(hRight + hBuffer, vBot + vBuffer, edge.roiCorners[2]);
                GeoToRot(hLeft - hBuffer, vBot + vBuffer, edge.roiCorners[3]);
                
                if(j == 0) {
                    //Remember Id of first one.
                    firstId = id;
                }
                
                if(j != hCount - 1) {
                    res.adj[id].push_back(edge);
                } else {
                    //Loop last one back to first one.
                    edge.to = firstId;
                    res.adj[id].push_back(edge);
                }
                res.targets[i].push_back(p);
                
                id++;
            }
            
		}

        /*for(size_t i = 0; i < adj.size(); i++) {
            for(size_t j = 0; j < adj[i].size(); j++) {
                cout << i << " -> "<< adj[i][j] << endl;
            }
        }*/
        
        return res;
	}

    vector<ImageP> GenerateDebugImages(RecorderGraph &graph) const {
		vector<ImageP> images;

		for(auto ring : graph.targets) {
			for(auto t : ring) {

				images.push_back(CreateDebugImage(t.extrinsics, 1.0, Scalar(0, 255, 0)));

                for(auto edge : graph.adj[t.globalId]) {
                    for(auto corner : edge.roiCorners) {
				        images.push_back(CreateDebugImage(corner, 0.1, Scalar(0, 0, 255)));
                    }
                }
			}
		}

		return images;
	}


};
}

#endif
