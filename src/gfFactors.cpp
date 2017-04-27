#include <iostream>
#include <opencv2/video/tracking.hpp>
#include "io/inputImage.hpp"


using namespace std;
using namespace cv;

void saveDest(const Mat &dest, const char* name) {

    Mat planes[5];
    split(dest, planes);  

    for(int i = 0; i < 5; i++) {
        Mat out;
        planes[i].convertTo(out, CV_8UC1, 255*30, 0);
        imwrite("dbg/" + std::string(name) + optonaut::ToString(i) +  ".jpg", out);
    }
}

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

static void FarnebackPolyExp( const Mat& src, Mat& dst, int n, double sigma )
{
    int k, x, y;

    CV_Assert( src.type() == CV_32FC1 );
    int width = src.cols;
    int height = src.rows;
    AutoBuffer<float> kbuf(n*6 + 3);
    std::vector<AutoBuffer<float>> allBuffers;
    float* g = kbuf + n;
    float* xg = g + n*2 + 1;
    float* xxg = xg + n*2 + 1;
    double ig11, ig03, ig33, ig55;

    FarnebackPrepareGaussian(n, sigma, g, xg, xxg, ig11, ig03, ig33, ig55);

    dst.create( height, width, CV_32FC(5));

    for( y = 0; y < height; y++ )
    {
        AutoBuffer<float> _row((width + n*2)*3);
        float *row = (float*)_row + n*3;
        allBuffers.push_back(_row);
        float g0 = g[0], g1, g2;
        const float *srow0 = src.ptr<float>(y), *srow1 = 0;
        float *drow = dst.ptr<float>(y);

        // vertical part of convolution
        for( x = 0; x < width; x++ )
        {
            row[x*3] = srow0[x]*g0;
            row[x*3+1] = row[x*3+2] = 0.f;
        }

        for( k = 1; k <= n; k++ )
        {
            g0 = g[k]; g1 = xg[k]; g2 = xxg[k];
            srow0 = src.ptr<float>(std::max(y-k,0));
            srow1 = src.ptr<float>(std::min(y+k,height-1));

            for( x = 0; x < width; x++ )
            {
                float p = srow0[x] + srow1[x];
                float t0 = row[x*3] + g0*p;
                float t1 = row[x*3+1] + g1*(srow1[x] - srow0[x]);
                float t2 = row[x*3+2] + g2*p;

                row[x*3] = t0;
                row[x*3+1] = t1;
                row[x*3+2] = t2;
            
                drow[x*5] = t0;
                drow[x*5+1] = t1;
                drow[x*5+2] = t2; 
                drow[x*5+3] = 0; 
                drow[x*5+4] = 0; 
            }
        }
    }

   saveDest(dst, "intermediate_result");

   for( y = 0; y < height; y++ )
   {
        auto _row = allBuffers[y];
        float *row = (float*)_row + n*3;
        float g0 = g[0];
        float *drow = dst.ptr<float>(y);

        // horizontal part of convolution
        for( x = 0; x < n*3; x++ )
        {
            row[-1-x] = row[2-x];
            row[width*3+x] = row[width*3+x-3];
        }

        for( x = 0; x < width; x++ )
        {
            g0 = g[0];
            // r1 ~ 1, r2 ~ x, r3 ~ y, r4 ~ x^2, r5 ~ y^2, r6 ~ xy
            double b1 = row[x*3]*g0, b2 = 0, b3 = row[x*3+1]*g0,
                b4 = 0, b5 = row[x*3+2]*g0, b6 = 0;

            for( k = 1; k <= n; k++ )
            {
                double tg = row[(x+k)*3] + row[(x-k)*3];
                g0 = g[k];
                b1 += tg*g0;
                b4 += tg*xxg[k];
                b2 += (row[(x+k)*3] - row[(x-k)*3])*xg[k];
                b3 += (row[(x+k)*3+1] + row[(x-k)*3+1])*g0;
                b6 += (row[(x+k)*3+1] - row[(x-k)*3+1])*xg[k];
                b5 += (row[(x+k)*3+2] + row[(x-k)*3+2])*g0;
            }

            // do not store r1
            drow[x*5+1] = (float)(b2*ig11);
            drow[x*5] = (float)(b3*ig11);
            drow[x*5+3] = (float)(b1*ig03 + b4*ig33);
            drow[x*5+2] = (float)(b1*ig03 + b5*ig33);
            drow[x*5+4] = (float)(b6*ig55);
        }
        row -= n*3;
    }

}
void polyExp(const Mat &in) {
    Mat dest;
    FarnebackPolyExp(in, dest, 7, 1.5);
    saveDest(dest, "final_result");
}

int main() {

    fbCalcAndPrint(7, 1.5);

    Mat src= imread("murmeln.jpg");
    Mat planes[3];
    split(src,planes);  
    Mat fimg; 
    planes[2].convertTo(fimg, CV_32F, 1.0/255.0, 0); 

    polyExp(fimg);
}

