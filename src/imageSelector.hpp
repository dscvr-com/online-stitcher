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

class ImageSelector {

private:
	//adj[n] contains m if m is right of n
	vector<vector<int>> adj;
	vector<vector<SelectionPoint>> targets;
	Mat intrinsics;
	//Horizontal and Vertical overlap in procent. 
	const double hOverlap = 0.8;
	const double vOverlap = 0.3;

	//Tolerance, measured on sphere, for hits. 
	const double tolerance = M_PI / 16;
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

			double vAngle = i * vFov + vFov / 2 - M_PI / 2;

			int hCount = hCenterCount * cos(vAngle);
			hFov = M_PI * 2 / hCount;

			int initId = -1;

			targets.push_back(vector<SelectionPoint>());

			for(int j = 0; j < hCount; j++) {
                
                if(mode == ModeAll || (mode == ModeCenter && i == 2)) {
                    Mat hRot;
                    Mat vRot;
                    
                    CreateRotationY(j * hFov + hFov / 2, hRot);
                    CreateRotationX(vAngle, vRot);
                    
                    SelectionPoint p;
                    p.id = id;
                    p.extrinsics = hRot * vRot;
                    p.enabled = true;
                    p.ringId = i;
                    p.localId = j;
                    
                    
                    adj.push_back(vector<int>());
                    if(j != 0)
                        adj[id - 1].push_back(id);
                    else
                        initId = id;
                    
                    targets[i].push_back(p);
                    
                    id++;
                }
			}

            if(initId != -1) {
                adj[id - 1].push_back(initId);
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

	vector<ImageP> GenerateDebugImages() const {
		vector<ImageP> images;

		for(auto ring : targets) {
			for(auto t : ring) {
				ImageP d(new Image());

				d->extrinsics = t.extrinsics;
				d->intrinsics = intrinsics;
				d->img = Mat::zeros(240, 320, CV_8UC3);
				d->id = 0;

				line(d->img, Point2f(0, 0), Point2f(320, 240), Scalar(0, 255, 0), 4);
				line(d->img, Point2f(320, 0), Point2f(0, 240), Scalar(0, 255, 0), 4);

				images.push_back(d);
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

	void DisableSelectionPoint(const SelectionPoint& p) {
		targets[p.ringId][p.localId].enabled = false;
	}

	bool AreAdjacent(const SelectionPoint& left, const SelectionPoint& right) const {
		return find(adj[left.id].begin(), adj[left.id].end(), right.id) != adj[left.id].end();
	}	

};
}

#endif
