/*
 * Quaternion module.  
 */

#include <opencv2/opencv.hpp>
#include "quat.hpp"
#include "support.hpp"

using namespace std;
using namespace cv;

namespace optonaut {
namespace quat {

    bool IsQuat(const Mat &q) {
        return MatIs(q, 4, 1, CV_64F);
    }
    
    void MakeQuat(Mat &q) {
        q = Mat(4, 1, CV_64F);
    }

    void FromMat(const Mat &a, Mat &q) {

        assert(IsQuat(q));
        assert(MatIs(a, 3, 3, CV_64F));

        QT w, x, y, z;
        QT trace = a.at<QT>(0, 0) + a.at<QT>(1, 1) + a.at<QT>(2, 2);
        if( trace > 0 ) {
            QT s = 0.5f / sqrtf(trace+ 1.0f);
            w = 0.25f / s;
            x = ( a.at<QT>(2, 1) - a.at<QT>(1, 2) ) * s;
            y = ( a.at<QT>(0, 2) - a.at<QT>(2, 0) ) * s;
            z = ( a.at<QT>(1, 0) - a.at<QT>(0, 1) ) * s;
        } else {
            if ( a.at<QT>(0, 0) > a.at<QT>(1, 1) && a.at<QT>(0, 0) > a.at<QT>(2, 2) ) {
                QT s = 2.0f * sqrtf( 1.0f + a.at<QT>(0, 0) - a.at<QT>(1, 1) - a.at<QT>(2, 2));
                w = (a.at<QT>(2, 1) - a.at<QT>(1, 2) ) / s;
                x = 0.25f * s;
                y = (a.at<QT>(0, 1) + a.at<QT>(1, 0) ) / s;
                z = (a.at<QT>(0, 2) + a.at<QT>(2, 0) ) / s;
            } else if (a.at<QT>(1, 1) > a.at<QT>(2, 2)) {
                QT s = 2.0f * sqrtf( 1.0f + a.at<QT>(1, 1) - a.at<QT>(0, 0) - a.at<QT>(2, 2));
                w = (a.at<QT>(0, 2) - a.at<QT>(2, 0) ) / s;
                x = (a.at<QT>(0, 1) + a.at<QT>(1, 0) ) / s;
                y = 0.25f * s;
                z = (a.at<QT>(1, 2) + a.at<QT>(2, 1) ) / s;
            } else {
                QT s = 2.0f * sqrtf( 1.0f + a.at<QT>(2, 2) - a.at<QT>(0, 0) - a.at<QT>(1, 1) );
                w = (a.at<QT>(1, 0) - a.at<QT>(0, 1) ) / s;
                x = (a.at<QT>(0, 2) + a.at<QT>(2, 0) ) / s;
                y = (a.at<QT>(1, 2) + a.at<QT>(2, 1) ) / s;
                z = 0.25f * s;
            }
        }

        q.at<QT>(0, 0) = w;
        q.at<QT>(1, 0) = x;
        q.at<QT>(2, 0) = y;
        q.at<QT>(3, 0) = z;
    }

    void ToMat(const Mat &q, Mat &a) {

        assert(IsQuat(q));
        assert(MatIs(a, 3, 3, CV_64F));

        QT w = q.at<QT>(0, 0);
        QT x = q.at<QT>(1, 0);
        QT y = q.at<QT>(2, 0);
        QT z = q.at<QT>(3, 0);

        QT sqw = w*w;
        QT sqx = x*x;
        QT sqy = y*y;
        QT sqz = z*z;

        QT invs = 1 / (sqx + sqy + sqz + sqw);
        QT m00 = ( sqx - sqy - sqz + sqw)*invs ; 
        QT m11 = (-sqx + sqy - sqz + sqw)*invs ;
        QT m22 = (-sqx - sqy + sqz + sqw)*invs ;
        
        QT tmp1 = x*y;
        QT tmp2 = z*w;
        QT m10 = 2.0 * (tmp1 + tmp2)*invs;
        QT m01 = 2.0 * (tmp1 - tmp2)*invs;
        
        tmp1 = x*z;
        tmp2 = y*w;
        QT m20 = 2.0 * (tmp1 - tmp2)*invs;
        QT m02 = 2.0 * (tmp1 + tmp2)*invs;
        tmp1 = y*z;
        tmp2 = x*w;
        QT m21 = 2.0 * (tmp1 + tmp2)*invs;
        QT m12 = 2.0 * (tmp1 - tmp2)*invs;

        a.at<QT>(0, 0) = m00;
        a.at<QT>(1, 1) = m11;
        a.at<QT>(2, 2) = m22;
        a.at<QT>(1, 0) = m10;
        a.at<QT>(0, 1) = m01;
        a.at<QT>(2, 0) = m20;
        a.at<QT>(0, 2) = m02;
        a.at<QT>(2, 1) = m21;
        a.at<QT>(1, 2) = m12;
    }

    QT Dot(const Mat& a, const Mat &b) {
        assert(IsQuat(a) && IsQuat(b));
        
        return a.at<QT>(0, 0) * b.at<QT>(0, 0) +
        a.at<QT>(1, 0) * b.at<QT>(1, 0) +
        a.at<QT>(2, 0) * b.at<QT>(2, 0) +
        a.at<QT>(3, 0) * b.at<QT>(3, 0);
    }

    void Cross(const Mat& a, const Mat &b, Mat &res) {
        assert(IsQuat(a) && IsQuat(b) && IsQuat(res));

        res.at<QT>(0, 0) = 0;

        res.at<QT>(1, 0) = 
            a.at<QT>(2, 0) * b.at<QT>(3, 0) -
            a.at<QT>(3, 0) * b.at<QT>(2, 0);

        res.at<QT>(2, 0) = 
            a.at<QT>(3, 0) * b.at<QT>(1, 0) -
            a.at<QT>(1, 0) * b.at<QT>(3, 0);

        res.at<QT>(3, 0) = 
            a.at<QT>(1, 0) * b.at<QT>(2, 0) -
            a.at<QT>(2, 0) * b.at<QT>(1, 0);
    }

    void Mult(const Mat& a, const Mat &b, Mat &res) {

        assert(IsQuat(a));
        assert(IsQuat(res));

        Cross(a, b, res);

        res.at<QT>(0, 0) = 
            a.at<QT>(0, 0) * b.at<QT>(0, 0) -
            a.at<QT>(1, 0) * b.at<QT>(1, 0) +
            a.at<QT>(2, 0) * b.at<QT>(2, 0) +
            a.at<QT>(3, 0) * b.at<QT>(3, 0);

        res.at<QT>(1, 0) = res.at<QT>(1, 0) + 
            a.at<QT>(0, 0) * b.at<QT>(1, 0) +
            a.at<QT>(1, 0) * b.at<QT>(0, 0);

        res.at<QT>(2, 0) = res.at<QT>(2, 0) + 
            a.at<QT>(0, 0) * b.at<QT>(2, 0) +
            a.at<QT>(2, 0) * b.at<QT>(0, 0);

        res.at<QT>(3, 0) = res.at<QT>(3, 0) + 
            a.at<QT>(0, 0) * b.at<QT>(3, 0) +
            a.at<QT>(3, 0) * b.at<QT>(0, 0);
    }

    void Mult(const Mat& a, const QT &b, Mat &res) {

        assert(IsQuat(a));
        assert(IsQuat(res));

        res.at<QT>(0, 0) = a.at<QT>(0, 0) * b;
        res.at<QT>(1, 0) = a.at<QT>(1, 0) * b;
        res.at<QT>(2, 0) = a.at<QT>(2, 0) * b;
        res.at<QT>(3, 0) = a.at<QT>(3, 0) * b;
    }

    QT Norm(const Mat& a) {
        assert(IsQuat(a));
        return a.at<QT>(0, 0) * a.at<QT>(0, 0) +
            a.at<QT>(1, 0) * a.at<QT>(1, 0) +
            a.at<QT>(2, 0) * a.at<QT>(2, 0) +
            a.at<QT>(3, 0) * a.at<QT>(3, 0);
    }

    //Inverse, if its a unit quat. 
    void Conjugate(const Mat &a, Mat &res) {
        assert(IsQuat(a));
        assert(IsQuat(res));
        res.at<QT>(0, 0) = a.at<QT>(0, 0);
        res.at<QT>(1, 0) = -a.at<QT>(1, 0);
        res.at<QT>(2, 0) = -a.at<QT>(2, 0);
        res.at<QT>(3, 0) = -a.at<QT>(3, 0);
    }
}
}
