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

static void
FarnebackUpdateFlow_GaussianBlur( const Mat& _R0, const Mat& _R1,
                                 Mat& _flow, Mat& matM, int block_size,
                                 bool update_matrices )
{
    int x, y, i, width = _flow.cols, height = _flow.rows;
    int m = block_size/2;
    int y0 = 0, y1;
    int min_update_stripe = std::max((1 << 10)/width, block_size);
    double sigma = m*0.3, s = 1;
    
    AutoBuffer<float> _vsum((width+m*2+2)*5 + 16), _hsum(width*5 + 16);
    AutoBuffer<float> _kernel((m+1)*5 + 16);
    AutoBuffer<float*> _srow(m*2+1);
    float *vsum = alignPtr((float*)_vsum + (m+1)*5, 16), *hsum = alignPtr((float*)_hsum, 16);
    float* kernel = (float*)_kernel;
    const float** srow = (const float**)&_srow[0];
    kernel[0] = (float)s;
    
    for( i = 1; i <= m; i++ )
    {
        float t = (float)std::exp(-i*i/(2*sigma*sigma) );
        kernel[i] = t;
        s += t*2;
    }
    
    s = 1./s;
    for( i = 0; i <= m; i++ )
        kernel[i] = (float)(kernel[i]*s);
    
#if CV_SSE2
    float* simd_kernel = alignPtr(kernel + m+1, 16);
    volatile bool useSIMD = checkHardwareSupport(CV_CPU_SSE);
    if( useSIMD )
    {
        for( i = 0; i <= m; i++ )
            _mm_store_ps(simd_kernel + i*4, _mm_set1_ps(kernel[i]));
    }
#endif
    
    // compute blur(G)*flow=blur(h)
    for( y = 0; y < height; y++ )
    {
        double g11, g12, g22, h1, h2;
        float* flow = _flow.ptr<float>(y);
        
        // vertical blur
        for( i = 0; i <= m; i++ )
        {
            srow[m-i] = matM.ptr<float>(std::max(y-i,0));
            srow[m+i] = matM.ptr<float>(std::min(y+i,height-1));
        }
        
        x = 0;
#if CV_SSE2
        if( useSIMD )
        {
            for( ; x <= width*5 - 16; x += 16 )
            {
                const float *sptr0 = srow[m], *sptr1;
                __m128 g4 = _mm_load_ps(simd_kernel);
                __m128 s0, s1, s2, s3;
                s0 = _mm_mul_ps(_mm_loadu_ps(sptr0 + x), g4);
                s1 = _mm_mul_ps(_mm_loadu_ps(sptr0 + x + 4), g4);
                s2 = _mm_mul_ps(_mm_loadu_ps(sptr0 + x + 8), g4);
                s3 = _mm_mul_ps(_mm_loadu_ps(sptr0 + x + 12), g4);
                
                for( i = 1; i <= m; i++ )
                {
                    __m128 x0, x1;
                    sptr0 = srow[m+i], sptr1 = srow[m-i];
                    g4 = _mm_load_ps(simd_kernel + i*4);
                    x0 = _mm_add_ps(_mm_loadu_ps(sptr0 + x), _mm_loadu_ps(sptr1 + x));
                    x1 = _mm_add_ps(_mm_loadu_ps(sptr0 + x + 4), _mm_loadu_ps(sptr1 + x + 4));
                    s0 = _mm_add_ps(s0, _mm_mul_ps(x0, g4));
                    s1 = _mm_add_ps(s1, _mm_mul_ps(x1, g4));
                    x0 = _mm_add_ps(_mm_loadu_ps(sptr0 + x + 8), _mm_loadu_ps(sptr1 + x + 8));
                    x1 = _mm_add_ps(_mm_loadu_ps(sptr0 + x + 12), _mm_loadu_ps(sptr1 + x + 12));
                    s2 = _mm_add_ps(s2, _mm_mul_ps(x0, g4));
                    s3 = _mm_add_ps(s3, _mm_mul_ps(x1, g4));
                }
                
                _mm_store_ps(vsum + x, s0);
                _mm_store_ps(vsum + x + 4, s1);
                _mm_store_ps(vsum + x + 8, s2);
                _mm_store_ps(vsum + x + 12, s3);
            }
            
            for( ; x <= width*5 - 4; x += 4 )
            {
                const float *sptr0 = srow[m], *sptr1;
                __m128 g4 = _mm_load_ps(simd_kernel);
                __m128 s0 = _mm_mul_ps(_mm_loadu_ps(sptr0 + x), g4);
                
                for( i = 1; i <= m; i++ )
                {
                    sptr0 = srow[m+i], sptr1 = srow[m-i];
                    g4 = _mm_load_ps(simd_kernel + i*4);
                    __m128 x0 = _mm_add_ps(_mm_loadu_ps(sptr0 + x), _mm_loadu_ps(sptr1 + x));
                    s0 = _mm_add_ps(s0, _mm_mul_ps(x0, g4));
                }
                _mm_store_ps(vsum + x, s0);
            }
        }
#endif
        for( ; x < width*5; x++ )
        {
            float s0 = srow[m][x]*kernel[0];
            for( i = 1; i <= m; i++ )
                s0 += (srow[m+i][x] + srow[m-i][x])*kernel[i];
            vsum[x] = s0;
        }
        
        // update borders
        for( x = 0; x < m*5; x++ )
        {
            vsum[-1-x] = vsum[4-x];
            vsum[width*5+x] = vsum[width*5+x-5];
        }
        
        // horizontal blur
        x = 0;
#if CV_SSE2
        if( useSIMD )
        {
            for( ; x <= width*5 - 8; x += 8 )
            {
                __m128 g4 = _mm_load_ps(simd_kernel);
                __m128 s0 = _mm_mul_ps(_mm_loadu_ps(vsum + x), g4);
                __m128 s1 = _mm_mul_ps(_mm_loadu_ps(vsum + x + 4), g4);
                
                for( i = 1; i <= m; i++ )
                {
                    g4 = _mm_load_ps(simd_kernel + i*4);
                    __m128 x0 = _mm_add_ps(_mm_loadu_ps(vsum + x - i*5),
                                           _mm_loadu_ps(vsum + x + i*5));
                    __m128 x1 = _mm_add_ps(_mm_loadu_ps(vsum + x - i*5 + 4),
                                           _mm_loadu_ps(vsum + x + i*5 + 4));
                    s0 = _mm_add_ps(s0, _mm_mul_ps(x0, g4));
                    s1 = _mm_add_ps(s1, _mm_mul_ps(x1, g4));
                }
                
                _mm_store_ps(hsum + x, s0);
                _mm_store_ps(hsum + x + 4, s1);
            }
        }
#endif
        for( ; x < width*5; x++ )
        {
            float sum = vsum[x]*kernel[0];
            for( i = 1; i <= m; i++ )
                sum += kernel[i]*(vsum[x - i*5] + vsum[x + i*5]);
            hsum[x] = sum;
        }
        
    //saveDest(flow, "flow_update_before_det", 10000);
    //saveDest(M, "M_update_gefore_det", 10000);
        for( x = 0; x < width; x++ )
        {
            g11 = hsum[x*5];
            g12 = hsum[x*5+1];
            g22 = hsum[x*5+2];
            h1 = hsum[x*5+3];
            h2 = hsum[x*5+4];
            
            double idet = 1./(g11*g22 - g12*g12 + 1e-3);
            
            flow[x*2] = (float)((g11*h2-g12*h1)*idet);
            flow[x*2+1] = (float)((g22*h1-g12*h2)*idet);
        }
        
        y1 = y == height - 1 ? height : y - block_size;
        if( update_matrices && (y1 == height || y1 >= y0 + min_update_stripe) )
        {
            FarnebackUpdateMatrices( _R0, _R1, _flow, matM, y0, y1 );
            y0 = y1;
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

    Mat planes[3], planes2[3];
    split(src,planes);  
    Mat fimg, fimg2; 
    planes[2].convertTo(fimg, CV_32F, 1.0/255.0, 0); 
    split(src2,planes2);  
    planes2[2].convertTo(fimg2, CV_32F, 1.0/255.0, 0); 

    Mat poly1, poly2;

    polyExp(fimg, poly1);
    polyExp(fimg2, poly2);

    Mat flow = Mat::zeros(poly1.rows, poly1.cols, CV_32FC2);
    flow.setTo(Scalar(1, 0));
    Mat M;
    
    FarnebackUpdateMatrices( poly1, poly2, flow, M, 0, poly1.rows);
    saveDest(flow, "flow_update", 10000);
    saveDest(M, "M_update", 10000);
    FarnebackUpdateFlow_GaussianBlur(poly1, poly2,
                                 flow, M, 7, false);
    saveDest(flow, "flow_update_g", 100);
    saveDest(M, "M_update_g", 10000);

    
    FarnebackUpdateMatrices( poly1, poly2, flow, M, 0, poly1.rows);
    saveDest(flow, "flow_update_2", 10000);
    saveDest(M, "M_update_2", 10000);
    FarnebackUpdateFlow_GaussianBlur(poly1, poly2,
                                 flow, M, 7, false);
    saveDest(flow, "flow_update_g_2", 100);
    saveDest(M, "M_update_g_2", 10000);
   

    FarnebackUpdateMatrices( poly1, poly2, flow, M, 0, poly1.rows);
    saveDest(flow, "flow_update_3", 10000);
    saveDest(M, "M_update_3", 10000);
    FarnebackUpdateFlow_GaussianBlur(poly1, poly2,
                                 flow, M, 7, false);
    saveDest(flow, "flow_update_g_3", 100);
    saveDest(M, "M_update_g_3", 10000);

    Mat truth = Mat::zeros(src.size(), CV_32FC2);

    cv::calcOpticalFlowFarneback(planes[2], planes2[2], truth, 0.3, 0, 5, 5, 7, 1.5, 0);

    saveDest(truth, "Truth", 1.0 / 255.0);
}

