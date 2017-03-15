#include <vector>
#include "../math/support.hpp"
#include "../math/quat.hpp"
#include "../common/assert.hpp"

using namespace std;
using namespace cv;
using namespace optonaut;

void TestQuatConversion(const Mat &in) {
    Mat a = in(Rect(0, 0, 3, 3));
    Mat q(4, 1, CV_64F);
    Mat r(3, 3, CV_64F);

    quat::FromMat(a, q);
    quat::ToMat(q, r);

    AssertMatEQ<double>(a, r); 
}

void TestQuatProduct(const Mat &inA, const Mat &inB) {
    Mat a = inA(Rect(0, 0, 3, 3));
    Mat b = inB(Rect(0, 0, 3, 3));
    Mat rm(3, 3, CV_64F);
    Mat rq(3, 3, CV_64F);

    Mat q(4, 1, CV_64F);
    Mat k(4, 1, CV_64F);
    Mat x(4, 1, CV_64F);

    quat::FromMat(a, q);
    quat::FromMat(b, k);

    rm = a * b;
    quat::Mult(q, k, x);
    quat::ToMat(x, rq);

    AssertMatEQ<double>(rm, rq);
}

int main(int, char**) {
    Mat a = Mat::eye(4, 4, CV_64F);
    Mat b = Mat::eye(4, 4, CV_64F); 
    Mat test = Mat::eye(4, 4, CV_64F);
    TestQuatConversion(test);

    double angles[] = {0, M_PI, M_PI / 2, M_PI / 8, 0.12, 0.99, 0.1};
    size_t n = sizeof(angles) / sizeof(*angles);

    for(size_t i = 0; i < n; i++) {
        CreateRotationY(angles[i], a);
        for(size_t j = 0; j < n; j++) {
            CreateRotationZ(angles[j], b);
            for(size_t k = 0; k < n; k++) {
                CreateRotationX(angles[k], test);

                test = test * a * b;
                TestQuatConversion(test);
            }
        }
    }

    cout << "[\u2713] Quat conversion module." << endl;
    
    for(size_t i = 0; i < n; i++) {
        CreateRotationY(angles[i], a);
        for(size_t j = 0; j < n; j++) {
            CreateRotationZ(angles[j], b);

            TestQuatProduct(a, b);
        }
    }

    cout << "[\u2713] Quat arithmetic module." << endl;
}
