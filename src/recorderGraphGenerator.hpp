#include <opencv2/opencv.hpp>
#include <math.h>

#include "inputImage.hpp"
#include "support.hpp"
#include "recorderGraph.hpp"
#include "ringProcessor.hpp"

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
    const double hBufferRatio = 3;
    const double vBufferRatio = 0.05;
    
    Mat intrinsics;

    static void GeoToRot(double hAngle, double vAngle, Mat &res) {
        Mat hRot;
        Mat vRot;
        
        //cout << hAngle << ", " << vAngle << endl;
        
        CreateRotationY(hAngle, hRot);
        CreateRotationX(vAngle, vRot);
        
        res = hRot * vRot;
    }
    
    InputImageP CreateDebugImage(Mat pos, double scale, Scalar color) const {
        InputImageP d(new InputImage());
        
        d->originalExtrinsics = pos;
        d->adjustedExtrinsics = pos;
        d->intrinsics = intrinsics.clone();
        d->intrinsics.at<double>(0, 2) *= scale;
        d->intrinsics.at<double>(1, 2) *= scale;
        d->image = Image(Mat::zeros(240, 320, CV_8UC3));
        d->id = 0;
        
        line(d->image.data, Point2f(0, 0), Point2f(320, 240), color, 4);
        line(d->image.data, Point2f(320, 0), Point2f(0, 240), color, 4);
        
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
        double vStart = 0, vBuffer = 0;
        uint32_t id = 0;
        uint32_t hCenterCount = ceil(2 * M_PI / hFov);
        
        //TODO: Wrong assumptions.
        //This actually builds the recorder graph from the bottom. 

        if(mode == RecorderGraph::ModeTruncated || mode == RecorderGraph::ModeTinyDebug) {
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
            vStart = (M_PI - (vFov * 3)) / 2;
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

			uint32_t hCount = hCenterCount * cos(vCenter);
			hFov = M_PI * 2 / hCount;
            double hBuffer = hFov * hBufferRatio;

            double hLeft = 0;
            SelectionEdge edge;
            
            if(mode ==  RecorderGraph::ModeTinyDebug) {
                hCount = 6;
            }

			res.targets.push_back(vector<SelectionPoint>(hCount));

            auto createEdge = [hBuffer, vBuffer, &res] (SelectionPoint &a, SelectionPoint &b) {
                SelectionEdge edge;
                edge.from = a.globalId;
                edge.to = b.globalId;

                double hLeft = a.hPos;
                double hRight = b.hPos;
                if(hLeft > hRight) {
                    hRight += 2 * M_PI;
                }
                double hCenter = (hLeft + hRight) / 2.0;
                double vCenter = a.vPos;
                double vTop = vCenter - a.vFov / 2.0;
                double vBot = vCenter + a.vFov / 2.0;

                GeoToRot(hCenter, vCenter, edge.roiCenter);
                GeoToRot(hLeft - hBuffer, vTop - vBuffer, edge.roiCorners[0]);
                GeoToRot(hRight + hBuffer, vTop - vBuffer, edge.roiCorners[1]);
                GeoToRot(hRight + hBuffer, vBot + vBuffer, edge.roiCorners[2]);
                GeoToRot(hLeft - hBuffer, vBot + vBuffer, edge.roiCorners[3]);

                assert(hLeft - hBuffer < hRight + hBuffer); 
                    
                res.adj[edge.from].push_back(edge);
            };

            auto finish = [&res] (SelectionPoint &a) {
                res.targets[a.ringId][a.localId] = a;
            };

            RingProcessor<SelectionPoint> hqueue(1, createEdge, finish);
                    
            for(uint32_t j = 0; j < hCount; j++) {
                res.adj.push_back(vector<SelectionEdge>());
                
                hLeft = j * hFov;
               // hCenter = hLeft + hFov / 2;
               // hRight = hLeft + hFov;
                
                SelectionPoint p;
                p.globalId = id;
                GeoToRot(hLeft, vCenter, p.extrinsics);
                p.hPos = hLeft;
                p.vPos = vCenter;
                p.ringId = i;
                p.localId = j;
                p.vFov = vFov;
                p.hFov = hFov;
                
                hqueue.Push(p);
                
                id++;
            }

            hqueue.Flush();
            
		}

        /*for(size_t i = 0; i < adj.size(); i++) {
            for(size_t j = 0; j < adj[i].size(); j++) {
                cout << i << " -> "<< adj[i][j] << endl;
            }
        }*/
        
        return res;
	}

    vector<InputImageP> GenerateDebugImages(RecorderGraph &graph) const {
		vector<InputImageP> images;

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
