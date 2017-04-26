#include <iostream>
#include <opencv2/video/tracking.hpp>


using namespace std;
using namespace cv;

void
FarnebackPrepareGaussian(int n, double sigma, float *g, float *xg, float *xxg,
                         double &ig11, double &ig03, double &ig33, double &ig55)
{
    if( sigma < FLT_EPSILON )
        sigma = n*0.3;

    double s = 0.;
    for (int x = -n; x <= n; x++)
    {
        g[x] = (float)std::exp(-x*x/(2*sigma*sigma));
        s += g[x];
    }

    s = 1./s;
    for (int x = -n; x <= n; x++)
    {
        g[x] = (float)(g[x]*s);
        xg[x] = (float)(x*g[x]);
        xxg[x] = (float)(x*x*g[x]);
    }

    Mat_<double> G(6, 6);
    G.setTo(0);

    for (int y = -n; y <= n; y++)
    {
        for (int x = -n; x <= n; x++)
        {
            G(0,0) += g[y]*g[x];
            G(1,1) += g[y]*g[x]*x*x;
            G(3,3) += g[y]*g[x]*x*x*x*x;
            G(5,5) += g[y]*g[x]*x*x*y*y;
        }
    }

    //G[0][0] = 1.;
    G(2,2) = G(0,3) = G(0,4) = G(3,0) = G(4,0) = G(1,1);
    G(4,4) = G(3,3);
    G(3,4) = G(4,3) = G(5,5);

    // invG:
    // [ x        e  e    ]
    // [    y             ]
    // [       y          ]
    // [ e        z       ]
    // [ e           z    ]
    // [                u ]
    Mat_<double> invG = G.inv(DECOMP_CHOLESKY);

    ig11 = invG(1,1);
    ig03 = invG(0,3);
    ig33 = invG(3,3);
    ig55 = invG(5,5);
}

void printArray(float* g, int n, const char* label) {
    cout << label << " = {"; 

    for(int i = 0; i <= n * 2; i++) {
        cout << g[i] << " ";
    }

    cout << "}" << std::endl;
}


void fbCalcAndPrint(int n, float sigma) {
    AutoBuffer<float> kbuf(n*6 + 3);
    float* g = kbuf + n;
    float* xg = g + n*2 + 1;
    float* xxg = xg + n*2 + 1;
    double ig11, ig03, ig33, ig55;

    FarnebackPrepareGaussian(n, sigma, g, xg, xxg, ig11, ig03, ig33, ig55);

    cout << "n = " << n << "; sigma = " << sigma << ";" << std::endl;
    cout << "ig11 = " << ig11 << "; ig03 = " << ig03 << "; ig33 = " << ig33 << " = ig55; " << std::endl;

    printArray(g, n, "g");
    printArray(xg, n, "xg");
    printArray(xxg, n, "xxg");
}

int main() {

    fbCalcAndPrint(7, 1.5);
}

