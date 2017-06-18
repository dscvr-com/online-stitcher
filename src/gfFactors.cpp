#include <iostream>
#include <opencv2/video/tracking.hpp>
#include "io/inputImage.hpp"


using namespace std;
using namespace cv;

void saveDest(const Mat &dest, const char* name, float scale = 30) {
    int n = dest.channels();
    Mat planes[n];
    split(dest, planes);  

    for(int i = 0; i < n; i++) {
        Mat out;
        Mat a = abs(planes[i]);
        a.convertTo(out, CV_8UC1, 255*scale, 0);
        imwrite("dbg/" + std::string(name) + optonaut::ToString(i) +  ".jpg", out);
    }
}

static void
FarnebackUpdateMatrices( const Mat& _R0, const Mat& _R1, const Mat& _flow, Mat& matM, int _y0, int _y1 )
{
    const int BORDER = 5;
    static const float border[BORDER] = {0.14f, 0.14f, 0.4472f, 0.4472f, 0.4472f};

    int x, y, width = _flow.cols, height = _flow.rows;
    const float* R1 = _R1.ptr<float>();
    size_t step1 = _R1.step/sizeof(R1[0]);

    matM.create(height, width, CV_32FC(5));

    for( y = _y0; y < _y1; y++ )
    {
        const float* flow = _flow.ptr<float>(y);
        const float* R0 = _R0.ptr<float>(y);
        float* M = matM.ptr<float>(y);

        for( x = 0; x < width; x++ )
        {
            float dx = flow[x*2], dy = flow[x*2+1];
            float fx = x + dx, fy = y + dy;

#if 1
            int x1 = cvFloor(fx), y1 = cvFloor(fy);
            const float* ptr = R1 + y1*step1 + x1*5;
            float r2, r3, r4, r5, r6;

            fx -= x1; fy -= y1;

            if( (unsigned)x1 < (unsigned)(width-1) &&
                (unsigned)y1 < (unsigned)(height-1) )
            {
                float a00 = (1.f-fx)*(1.f-fy), a01 = fx*(1.f-fy),
                      a10 = (1.f-fx)*fy, a11 = fx*fy;

                r2 = a00*ptr[0] + a01*ptr[5] + a10*ptr[step1] + a11*ptr[step1+5];
                r3 = a00*ptr[1] + a01*ptr[6] + a10*ptr[step1+1] + a11*ptr[step1+6];
                r4 = a00*ptr[2] + a01*ptr[7] + a10*ptr[step1+2] + a11*ptr[step1+7];
                r5 = a00*ptr[3] + a01*ptr[8] + a10*ptr[step1+3] + a11*ptr[step1+8];
                r6 = a00*ptr[4] + a01*ptr[9] + a10*ptr[step1+4] + a11*ptr[step1+9];
            
                r4 = (R0[x*5+2] + r4)*0.5f;
                r5 = (R0[x*5+3] + r5)*0.5f;
                r6 = (R0[x*5+4] + r6)*0.25f;
            }
#else
            int x1 = cvRound(fx), y1 = cvRound(fy);
            const float* ptr = R1 + y1*step1 + x1*5;
            float r2, r3, r4, r5, r6;
            if( (unsigned)x1 < (unsigned)width &&
                (unsigned)y1 < (unsigned)height )
            {
                r2 = ptr[0];
                r3 = ptr[1];
                r4 = (R0[x*5+2] + ptr[2])*0.5f;
                r5 = (R0[x*5+3] + ptr[3])*0.5f;
                r6 = (R0[x*5+4] + ptr[4])*0.25f;
            }
#endif
            else
            {
                r2 = r3 = 0.f;
                r4 = R0[x*5+2];
                r5 = R0[x*5+3];
                r6 = R0[x*5+4]*0.5f;
            }

            r2 = (R0[x*5] - r2)*0.5f;
            r3 = (R0[x*5+1] - r3)*0.5f;

            r2 += r4*dy + r6*dx;
            r3 += r6*dy + r5*dx;

            if( (unsigned)(x - BORDER) >= (unsigned)(width - BORDER*2) ||
                (unsigned)(y - BORDER) >= (unsigned)(height - BORDER*2))
            {
                float scale = (x < BORDER ? border[x] : 1.f)*
                    (x >= width - BORDER ? border[width - x - 1] : 1.f)*
                    (y < BORDER ? border[y] : 1.f)*
                    (y >= height - BORDER ? border[height - y - 1] : 1.f);

                r2 *= scale; r3 *= scale; r4 *= scale;
                r5 *= scale; r6 *= scale;
            }

            M[x*5]   = r4*r4 + r6*r6; // G(1,1)
            M[x*5+1] = (r4 + r5)*r6;  // G(1,2)=G(2,1)
            M[x*5+2] = r5*r5 + r6*r6; // G(2,2)
            M[x*5+3] = r4*r2 + r6*r3; // h(1)
            M[x*5+4] = r6*r2 + r5*r3; // h(2)
        }
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
void polyExp(const Mat &in, Mat &dest) {
    FarnebackPolyExp(in, dest, 7, 1.5);
    saveDest(dest, "final_result");
}

int main() {

    fbCalcAndPrint(7, 1.5);

    Mat src = imread("Murmeln_LEFT.jpg");
    Mat src2 = imread("Murmeln_RIGHT.jpg");

    Mat planes[3];
    split(src,planes);  
    Mat fimg, fimg2; 
    planes[2].convertTo(fimg, CV_32F, 1.0/255.0, 0); 
    split(src2,planes);  
    planes[2].convertTo(fimg2, CV_32F, 1.0/255.0, 0); 

    Mat poly1, poly2;

    polyExp(fimg, poly1);
    polyExp(fimg2, poly2);

    Mat flow = Mat::zeros(poly1.rows, poly1.cols, CV_32FC2);
    Mat M;
    
    FarnebackUpdateMatrices( poly1, poly2, flow, M, 0, poly1.rows);

    saveDest(flow, "flow", 100000);
    saveDest(M, "M", 100000);

}

