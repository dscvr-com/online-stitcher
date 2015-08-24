#include <opencv2/features2d.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/opencv.hpp>
#include <math.h>

#include "image.hpp"
#include "support.hpp"

using namespace cv;
using namespace std;

#define _USE_MATH_DEFINES

#ifndef OPTONAUT_IMAGE_SELECTOR_HEADER
#define OPTONAUT_IMAGE_SELECTOR_HEADER

namespace optonaut {

struct SelectionPoint {
	int id;
	int ringId;
	int localId;
	bool enabled;
	Mat extrinsics;
};

struct SelectionInfo {
	SelectionPoint closestPoint;
	ImageP image;
	double dist;
	bool isValid;

	SelectionInfo() {
		isValid = false;
	}
};

struct SelectionEdge {
    int from;
    int to;

    Mat roiCorners[4] ;
    Mat roiCenter;
};
    
class ImageSelector {

private:
	//adj[n] contains m if m is right of n
	vector<vector<SelectionEdge>> adj;
	vector<vector<SelectionPoint>> targets;
	Mat intrinsics;
	//Horizontal and Vertical overlap in percent. 
	const double hOverlap = 0.6;
	const double vOverlap = 0.3;

	//Tolerance, measured on sphere, for hits. 
	const double tolerance = M_PI / 32;

    Mat GeoToRot(double hAngle, double vAngle) {
        Mat hRot;
        Mat vRot;
        
        CreateRotationY(hAngle, hRot);
        CreateRotationX(vAngle, vRot);
        
        return ((Mat)(hRot * vRot)).clone();
    }
public:

    static const int ModeAll = 0;
    static const int ModeCenter = 1;
    
	ImageSelector(const Mat &intrinsics, const int mode = ModeAll) {

		this->intrinsics = intrinsics;

		double hFov = GetHorizontalFov(intrinsics) * (1.0 - hOverlap);
		double vFov = GetVerticalFov(intrinsics) * (1.0 - vOverlap);

		int vCount = ceil(M_PI / vFov);
		int hCenterCount = ceil(2 * M_PI / hFov);
	
		vFov = M_PI / vCount;

		int id = 0;

		for(int i = 0; i < vCount; i++) {

            //Vertical center, bottom and top of ring
			double vCenter = i * vFov + vFov / 2 - M_PI / 2;
            double vTop = vCenter - vFov / 2;
            double vBot = vCenter + vFov / 2;

			int hCount = hCenterCount * cos(vCenter);
			hFov = M_PI * 2 / hCount;

			int firstId = -1;
            double hLeft = 0;
            double hCenter = 0;
            double hRight = 0;
            SelectionEdge edge;

			targets.push_back(vector<SelectionPoint>());

			for(int j = 0; j < hCount; j++) {
                
                if(mode == ModeAll || (mode == ModeCenter && i == vCount / 2)) {
                    
                    hLeft = j * hFov;
                    hCenter = hLeft + hFov / 2;
                    hRight = hLeft + hFov;

                    SelectionPoint p;
                    p.id = id;
                    p.extrinsics = GeoToRot(hLeft, vCenter);
                    p.enabled = true;
                    p.ringId = i;
                    p.localId = j;
                
                    adj.push_back(vector<SelectionEdge>());

                    SelectionEdge edge; 
                    edge.from = id;
                    edge.to = id + 1;
                    edge.roiCenter = p.extrinsics;
                    edge.roiCorners[0] = GeoToRot(hLeft, vTop);
                    edge.roiCorners[1] = GeoToRot(hRight, vTop);
                    edge.roiCorners[2] = GeoToRot(hRight, vBot);
                    edge.roiCorners[3] = GeoToRot(hLeft, vBot);

                    if(j == 0) {
                        //Remember Id of first one. 
                        firstId = id;
                    }

                    if(j != hCount) {
                        adj[id].push_back(edge);
                    } else {
                        //Loop last one back to first one.
                        edge.to = firstId;
                        adj[id].push_back(edge);    
                    } 
                    targets[i].push_back(p);
                    
                    id++;
                }
			}
		}

        /*for(size_t i = 0; i < adj.size(); i++) {
            for(size_t j = 0; j < adj[i].size(); j++) {
                cout << i << " -> "<< adj[i][j] << endl;
            }
        }*/
	}

	const vector<vector<SelectionPoint>> &GetRings() const {
		return targets;
	}

    ImageP CreateDebugImage(Mat pos, double scale, Scalar color) const {
        ImageP d(new Image());

        d->extrinsics = pos;
        d->intrinsics = intrinsics.clone();
        d->intrinsics.at<double>(0, 2) *= scale;
        d->intrinsics.at<double>(1, 2) *= scale;
        d->img = Mat::zeros(240, 320, CV_8UC3);
        d->id = 0;

        line(d->img, Point2f(0, 0), Point2f(320, 240), color, 4);
        line(d->img, Point2f(320, 0), Point2f(0, 240), color, 4);

        return d;
    }

	vector<ImageP> GenerateDebugImages() const {
		vector<ImageP> images;

		for(auto ring : targets) {
			for(auto t : ring) {

				images.push_back(CreateDebugImage(t.extrinsics, 1.0, Scalar(0, 255, 0)));

                for(auto edge : adj[t.id]) {
                    for(auto corner : edge.roiCorners) {
				        images.push_back(CreateDebugImage(corner, 0.1, Scalar(0, 0, 255)));
                    }
                }
			}
		}

		return images;
	}

	const SelectionInfo FindClosestSelectionPoint(ImageP img) const {
		SelectionInfo info;
		info.dist = -1;
		info.isValid = false;
		Mat eInv = img->extrinsics.inv();

		for(auto ring : targets) {
			for(auto target : ring) {
				if(!target.enabled)
					continue;
			    double dist = GetAngleOfRotation(eInv * target.extrinsics);
			    if (dist < info.dist || info.dist < 0) {
					info.image = img;
					info.closestPoint = target;
					info.dist = dist;
			    }
			}
		}
					
        info.isValid = tolerance >= info.dist;

		return info;
	}

	void DisableAdjacency(const SelectionPoint& left, const SelectionPoint& right) {
        for(auto it = adj[left.id].begin(); it != adj[left.id].end(); it++) {
            if(it->to == right.id) {
                adj[left.id].erase(it);
                break;
            }
        }
	}

	bool AreAdjacent(const SelectionPoint& left, const SelectionPoint& right) const {
        for(auto it = adj[left.id].begin(); it != adj[left.id].end(); it++) {
            if(it->to == right.id) {
                return true;
            }
        }
        return false;
	}	

};
}

#endif
