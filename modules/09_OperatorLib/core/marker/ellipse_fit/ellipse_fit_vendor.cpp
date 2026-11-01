// ============================================================================
// ellipse_fit_vendor.cpp — 【E3】cv::fitEllipse(5 点 CV_32F) vendor 逐位一致实现
//
// 来源（OpenCV 4.13.0 官方源码逐段转录，行号为该源码树行号）：
//   modules/imgproc/src/shapedescr.cpp
//     getOfs             :289-293（theRNG() 调用模式逐字保留，含 eps=0 轮）
//     fitEllipseNoDirect :295-448 （→ fitEllipseNoDirect5，奇异降级路径）
//     fitEllipseDirect   :658-824 （→ fitEllipse5Direct）
//   modules/core/src/lda.cpp
//     complex_div                        :297-311
//     EigenvalueDecomposition::hqr2      :315-768
//     EigenvalueDecomposition::orthes    :771-858
//     EigenvalueDecomposition::compute   :874-899（n=3 支语义并入 eigen3NonSymmetric）
//     eigenNonSymmetric  :956-1004（fallbackSymmetric=false；排序/写出 → eigen3NonSymmetric）
//
// 适配点（仅限下列，其余逐字转录）：
//   1. n=5 / depth=CV_32F 定死：is_float 三元收敛为 ptsf[i]；含 depth 的
//      CV_Assert 删除（定死后恒真）；n<5 CV_Error 逐字保留（死分支）
//   2. fitEllipseDirect 的 Mat A( n, 6, CV_64F) → 函数级 static thread_local
//      复用（稳态零堆分配）
//   3. cv::mulTransposed( A, DM, true, noArray(), 1.0, -1 ) → 手写 k 严格升序
//      三重循环（scale=1.0d 精确等价跳过）＋ completeSymm 语义镜像
//   4. DM *= (1.0/n) → 逐元素 DM(r,c) = DM(r,c) * (1.0/5.0)（保持乘常量写法）
//   5. eigenNonSymmetric(M, eVal, eVec) → vendor eigen3NonSymmetric（栈上定长
//      数组替 alloc_1d/alloc_2d；sortIdx 走公开 API；行序推导见函数注释）；
//      消费段 eVec.at<double>(r,c) → eVec[r][c]
//   6. cv::fitEllipseNoDirect( points ) → fitEllipseNoDirect5( pts5 )
//      （5 点 Mat 头零拷贝包装，函数体逐字）
//   7. CV_INSTRUMENT_REGION()（private.hpp 内部宏，非 instrumentation 构建
//      展开为空）不转录
//   8. EVD 的 this->n → 形参 n；成员数组 H/V/d/e/ort → 栈数组（形参传递）
// ============================================================================
#include "ellipse_fit_vendor.h"

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <limits>

namespace calib {
using namespace cv;

// —— shapedescr.cpp :289-293 逐字转录（RNG 流红线：含 eps=0 轮每点每轮 2 次 uniform）——
static inline Point2f getOfs(float eps)
{
    RNG& rng = theRNG();
    return Point2f(rng.uniform(-eps, eps), rng.uniform(-eps, eps));
}

// —— shapedescr.cpp :295-448 fitEllipseNoDirect 逐字转录（罕见降级路径，允许临时分配）——
// 【E3 适配】Mat points = _points.getMat() → 5 点 CV_32FC2 零拷贝包装（本地仍名
// points，函数体其余逐字符不变；points.ptr<Point2f>() 即从本包装取）
static RotatedRect fitEllipseNoDirect5( const Point2f* pts5 )
{
    Mat points( 5, 1, CV_32FC2, const_cast<Point2f*>(pts5) );
    int i, n = points.checkVector(2);
    int depth = points.depth();
    CV_Assert( n >= 0 && (depth == CV_32F || depth == CV_32S));

    RotatedRect box;

    if( n < 5 )
        CV_Error( cv::Error::StsBadSize, "There should be at least 5 points to fit the ellipse" );

    // New fitellipse algorithm, contributed by Dr. Daniel Weiss
    Point2f c(0,0);
    double gfp[5] = {0}, rp[5] = {0}, t, vd[25]={0}, wd[5]={0};
    const double min_eps = 1e-8;
    bool is_float = depth == CV_32F;

    AutoBuffer<double> _Ad(n*12+n);
    double *Ad = _Ad.data(), *ud = Ad + n*5, *bd = ud + n*5;
    Point2f* ptsf_copy = (Point2f*)(bd + n);

    // first fit for parameters A - E
    Mat A( n, 5, CV_64F, Ad );
    Mat b( n, 1, CV_64F, bd );
    Mat x( 5, 1, CV_64F, gfp );
    Mat u( n, 1, CV_64F, ud );
    Mat vt( 5, 5, CV_64F, vd );
    Mat w( 5, 1, CV_64F, wd );

    {
    const Point* ptsi = points.ptr<Point>();
    const Point2f* ptsf = points.ptr<Point2f>();
    for( i = 0; i < n; i++ )
    {
        Point2f p = is_float ? ptsf[i] : Point2f((float)ptsi[i].x, (float)ptsi[i].y);
        ptsf_copy[i] = p;
        c += p;
    }
    }
    c.x /= n;
    c.y /= n;

    double s = 0;
    for( i = 0; i < n; i++ )
    {
        Point2f p = ptsf_copy[i];
        p -= c;
        s += fabs(p.x) + fabs(p.y);
    }
    double scale = 100./(s > FLT_EPSILON ? s : FLT_EPSILON);

    for( i = 0; i < n; i++ )
    {
        Point2f p = ptsf_copy[i];
        p -= c;
        double px = p.x*scale;
        double py = p.y*scale;

        bd[i] = 10000.0; // 1.0?
        Ad[i*5] = -px * px; // A - C signs inverted as proposed by APP
        Ad[i*5 + 1] = -py * py;
        Ad[i*5 + 2] = -px * py;
        Ad[i*5 + 3] = px;
        Ad[i*5 + 4] = py;
    }

    SVDecomp(A, w, u, vt);
    if(wd[0]*FLT_EPSILON > wd[4]) {
        float eps = (float)(s/(n*2)*1e-3);
        for( i = 0; i < n; i++ )
        {
            const Point2f p = ptsf_copy[i] + getOfs(eps);
            ptsf_copy[i] = p;
        }

        for( i = 0; i < n; i++ )
        {
            Point2f p = ptsf_copy[i];
            p -= c;
            double px = p.x*scale;
            double py = p.y*scale;
            bd[i] = 10000.0; // 1.0?
            Ad[i*5] = -px * px; // A - C signs inverted as proposed by APP
            Ad[i*5 + 1] = -py * py;
            Ad[i*5 + 2] = -px * py;
            Ad[i*5 + 3] = px;
            Ad[i*5 + 4] = py;
        }
        SVDecomp(A, w, u, vt);
    }
    SVBackSubst(w, u, vt, b, x);

    // now use general-form parameters A - E to find the ellipse center:
    // differentiate general form wrt x/y to get two equations for cx and cy
    A = Mat( 2, 2, CV_64F, Ad );
    b = Mat( 2, 1, CV_64F, bd );
    x = Mat( 2, 1, CV_64F, rp );
    Ad[0] = 2 * gfp[0];
    Ad[1] = Ad[2] = gfp[2];
    Ad[3] = 2 * gfp[1];
    bd[0] = gfp[3];
    bd[1] = gfp[4];
    solve( A, b, x, DECOMP_SVD );

    // re-fit for parameters A - C with those center coordinates
    A = Mat( n, 3, CV_64F, Ad );
    b = Mat( n, 1, CV_64F, bd );
    x = Mat( 3, 1, CV_64F, gfp );
    for( i = 0; i < n; i++ )
    {
        Point2f p = ptsf_copy[i];
        p -= c;
        double px = p.x*scale;
        double py = p.y*scale;
        bd[i] = 1.0;
        Ad[i * 3] = (px - rp[0]) * (px - rp[0]);
        Ad[i * 3 + 1] = (py - rp[1]) * (py - rp[1]);
        Ad[i * 3 + 2] = (px - rp[0]) * (py - rp[1]);
    }
    solve(A, b, x, DECOMP_SVD);

    // store angle and radii
    rp[4] = -0.5 * atan2(gfp[2], gfp[1] - gfp[0]); // convert from APP angle usage
    if( fabs(gfp[2]) > min_eps )
        t = gfp[2]/sin(-2.0 * rp[4]);
    else // ellipse is rotated by an integer multiple of pi/2
        t = gfp[1] - gfp[0];
    rp[2] = fabs(gfp[0] + gfp[1] - t);
    if( rp[2] > min_eps )
        rp[2] = std::sqrt(2.0 / rp[2]);
    rp[3] = fabs(gfp[0] + gfp[1] + t);
    if( rp[3] > min_eps )
        rp[3] = std::sqrt(2.0 / rp[3]);

    box.center.x = (float)(rp[0]/scale) + c.x;
    box.center.y = (float)(rp[1]/scale) + c.y;
    box.size.width = (float)(rp[2]*2/scale);
    box.size.height = (float)(rp[3]*2/scale);
    if( box.size.width > box.size.height )
    {
        float tmp;
        CV_SWAP( box.size.width, box.size.height, tmp );
        box.angle = (float)(90 + rp[4]*180/CV_PI);
    }
    if( box.angle < -180 )
        box.angle += 360;
    if( box.angle > 360 )
        box.angle -= 360;

    return box;
}

// —— lda.cpp :297-311 complex_div 逐字转录 ——
static void complex_div(double xr, double xi, double yr, double yi, double& cdivr, double& cdivi) {
    double r, dv;
    CV_DbgAssert(std::abs(yr) + std::abs(yi) > 0.0);
    if (std::abs(yr) > std::abs(yi)) {
        r = yi / yr;
        dv = yr + r * yi;
        cdivr = (xr + r * xi) / dv;
        cdivi = (xi - r * xr) / dv;
    } else {
        r = yr / yi;
        dv = yi + r * yr;
        cdivr = (r * xr + xi) / dv;
        cdivi = (r * xi - xr) / dv;
    }
}

// —— lda.cpp :771-858 EigenvalueDecomposition::orthes 逐字转录
//    （【E3 适配】成员 n/H/V/ort → 形参；n=3 时 high=2，m 循环仅 m=1）——
static void orthes(int n, double H[3][3], double V[3][3], double ort[3]) {
    //  This is derived from the Algol procedures orthes and ortran,
    //  by Martin and Wilkinson, Handbook for Auto. Comp.,
    //  Vol.ii-Linear Algebra, and the corresponding
    //  Fortran subroutines in EISPACK.
    int low = 0;
    int high = n - 1;

    for (int m = low + 1; m < high; m++) {

        // Scale column.

        double scale = 0.0;
        for (int i = m; i <= high; i++) {
            scale = scale + std::abs(H[i][m - 1]);
        }
        if (scale != 0.0) {

            // Compute Householder transformation.

            double h = 0.0;
            for (int i = high; i >= m; i--) {
                ort[i] = H[i][m - 1] / scale;
                h += ort[i] * ort[i];
            }
            double g = std::sqrt(h);
            if (ort[m] > 0) {
                g = -g;
            }
            h = h - ort[m] * g;
            ort[m] = ort[m] - g;

            // Apply Householder similarity transformation
            // H = (I-u*u'/h)*H*(I-u*u')/h)

            for (int j = m; j < n; j++) {
                double f = 0.0;
                for (int i = high; i >= m; i--) {
                    f += ort[i] * H[i][j];
                }
                f = f / h;
                for (int i = m; i <= high; i++) {
                    H[i][j] -= f * ort[i];
                }
            }

            for (int i = 0; i <= high; i++) {
                double f = 0.0;
                for (int j = high; j >= m; j--) {
                    f += ort[j] * H[i][j];
                }
                f = f / h;
                for (int j = m; j <= high; j++) {
                    H[i][j] -= f * ort[j];
                }
            }
            ort[m] = scale * ort[m];
            H[m][m - 1] = scale * g;
        }
    }

    // Accumulate transformations (Algol's ortran).

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            V[i][j] = (i == j ? 1.0 : 0.0);
        }
    }

    for (int m = high - 1; m > low; m--) {
        if (H[m][m - 1] != 0.0) {
            for (int i = m + 1; i <= high; i++) {
                ort[i] = H[i][m - 1];
            }
            for (int j = m; j <= high; j++) {
                double g = 0.0;
                for (int i = m; i <= high; i++) {
                    g += ort[i] * V[i][j];
                }
                // Double division avoids possible underflow
                g = (g / ort[m]) / H[m][m - 1];
                for (int i = m; i <= high; i++) {
                    V[i][j] += g * ort[i];
                }
            }
        }
    }
}

// —— lda.cpp :315-768 EigenvalueDecomposition::hqr2 逐字转录
//    （【E3 适配】this->n → 形参 n；成员 H/V/d/e → 形参栈数组）——
static void hqr2(int n, double H[3][3], double V[3][3], double d[3], double e[3]) {

    //  This is derived from the Algol procedure hqr2,
    //  by Martin and Wilkinson, Handbook for Auto. Comp.,
    //  Vol.ii-Linear Algebra, and the corresponding
    //  Fortran subroutine in EISPACK.

    // Initialize
    const int max_iters_count = 1000 * n;

    const int nn = n; CV_Assert(nn > 0);
    int n1 = nn - 1;
    const int low = 0;
    const int high = nn - 1;
    const double eps = std::numeric_limits<double>::epsilon();
    double exshift = 0.0;

    // Store roots isolated by balanc and compute matrix norm

    double norm = 0.0;
    for (int i = 0; i < nn; i++) {
#if 0 // 'if' condition is always false
        if (i < low || i > high) {
            d[i] = H[i][i];
            e[i] = 0.0;
        }
#endif
        for (int j = std::max(i - 1, 0); j < nn; j++) {
            norm += std::abs(H[i][j]);
        }
    }

    // Outer loop over eigenvalue index
    int iter = 0;
    while (n1 >= low) {

        // Look for single small sub-diagonal element
        int l = n1;
        while (l > low) {
            if (norm < FLT_EPSILON) {
                break;
            }
            double s = std::abs(H[l - 1][l - 1]) + std::abs(H[l][l]);
            if (s == 0.0) {
                s = norm;
            }
            if (std::abs(H[l][l - 1]) < eps * s) {
                break;
            }
            l--;
        }

        // Check for convergence
        if (l == n1) {
            // One root found
            H[n1][n1] = H[n1][n1] + exshift;
            d[n1] = H[n1][n1];
            e[n1] = 0.0;
            n1--;
            iter = 0;

        } else if (l == n1 - 1) {
            // Two roots found
            double w = H[n1][n1 - 1] * H[n1 - 1][n1];
            double p = (H[n1 - 1][n1 - 1] - H[n1][n1]) * 0.5;
            double q = p * p + w;
            double z = std::sqrt(std::abs(q));
            H[n1][n1] = H[n1][n1] + exshift;
            H[n1 - 1][n1 - 1] = H[n1 - 1][n1 - 1] + exshift;
            double x = H[n1][n1];

            if (q >= 0) {
                // Real pair
                if (p >= 0) {
                    z = p + z;
                } else {
                    z = p - z;
                }
                d[n1 - 1] = x + z;
                d[n1] = d[n1 - 1];
                if (z != 0.0) {
                    d[n1] = x - w / z;
                }
                e[n1 - 1] = 0.0;
                e[n1] = 0.0;
                x = H[n1][n1 - 1];
                double s = std::abs(x) + std::abs(z);
                p = x / s;
                q = z / s;
                double r = std::sqrt(p * p + q * q);
                p = p / r;
                q = q / r;

                // Row modification

                for (int j = n1 - 1; j < nn; j++) {
                    z = H[n1 - 1][j];
                    H[n1 - 1][j] = q * z + p * H[n1][j];
                    H[n1][j] = q * H[n1][j] - p * z;
                }

                // Column modification

                for (int i = 0; i <= n1; i++) {
                    z = H[i][n1 - 1];
                    H[i][n1 - 1] = q * z + p * H[i][n1];
                    H[i][n1] = q * H[i][n1] - p * z;
                }

                // Accumulate transformations

                for (int i = low; i <= high; i++) {
                    z = V[i][n1 - 1];
                    V[i][n1 - 1] = q * z + p * V[i][n1];
                    V[i][n1] = q * V[i][n1] - p * z;
                }

            } else {
                // Complex pair
                d[n1 - 1] = x + p;
                d[n1] = x + p;
                e[n1 - 1] = z;
                e[n1] = -z;
            }
            n1 = n1 - 2;
            iter = 0;

        } else {
            // No convergence yet

            // Form shift
            double x = H[n1][n1];
            double y = 0.0;
            double w = 0.0;
            if (l < n1) {
                y = H[n1 - 1][n1 - 1];
                w = H[n1][n1 - 1] * H[n1 - 1][n1];
            }

            // Wilkinson's original ad hoc shift
            if (iter == 10) {
                exshift += x;
                for (int i = low; i <= n1; i++) {
                    H[i][i] -= x;
                }
                double s = std::abs(H[n1][n1 - 1]) + std::abs(H[n1 - 1][n1 - 2]);
                x = y = 0.75 * s;
                w = -0.4375 * s * s;
            }

            // MATLAB's new ad hoc shift

            if (iter == 30) {
                double s = (y - x) * 0.5;
                s = s * s + w;
                if (s > 0) {
                    s = std::sqrt(s);
                    if (y < x) {
                        s = -s;
                    }
                    s = x - w / ((y - x) * 0.5 + s);
                    for (int i = low; i <= n1; i++) {
                        H[i][i] -= s;
                    }
                    exshift += s;
                    x = y = w = 0.964;
                }
            }

            iter = iter + 1;
            if (iter > max_iters_count)
                CV_Error(Error::StsNoConv, "Algorithm doesn't converge (complex eigen values?)");

            double p = std::numeric_limits<double>::quiet_NaN();
            double q = std::numeric_limits<double>::quiet_NaN();
            double r = std::numeric_limits<double>::quiet_NaN();

            // Look for two consecutive small sub-diagonal elements
            int m = n1 - 2;
            while (m >= l) {
                double z = H[m][m];
                r = x - z;
                double s = y - z;
                p = (r * s - w) / H[m + 1][m] + H[m][m + 1];
                q = H[m + 1][m + 1] - z - r - s;
                r = H[m + 2][m + 1];
                s = std::abs(p) + std::abs(q) + std::abs(r);
                p = p / s;
                q = q / s;
                r = r / s;
                if (m == l) {
                    break;
                }
                if (std::abs(H[m][m - 1]) * (std::abs(q) + std::abs(r)) < eps * (std::abs(p)
                                                                                 * (std::abs(H[m - 1][m - 1]) + std::abs(z) + std::abs(
                                                                                                                                       H[m + 1][m + 1])))) {
                    break;
                }
                m--;
            }

            for (int i = m + 2; i <= n1; i++) {
                H[i][i - 2] = 0.0;
                if (i > m + 2) {
                    H[i][i - 3] = 0.0;
                }
            }

            // Double QR step involving rows l:n and columns m:n

            for (int k = m; k < n1; k++) {

                bool notlast = (k != n1 - 1);
                if (k != m) {
                    p = H[k][k - 1];
                    q = H[k + 1][k - 1];
                    r = (notlast ? H[k + 2][k - 1] : 0.0);
                    x = std::abs(p) + std::abs(q) + std::abs(r);
                    if (x != 0.0) {
                        p = p / x;
                        q = q / x;
                        r = r / x;
                    }
                }
                if (x == 0.0) {
                    break;
                }
                double s = std::sqrt(p * p + q * q + r * r);
                if (p < 0) {
                    s = -s;
                }
                if (s != 0) {
                    if (k != m) {
                        H[k][k - 1] = -s * x;
                    } else if (l != m) {
                        H[k][k - 1] = -H[k][k - 1];
                    }
                    p = p + s;
                    x = p / s;
                    y = q / s;
                    double z = r / s;
                    q = q / p;
                    r = r / p;

                    // Row modification

                    for (int j = k; j < nn; j++) {
                        p = H[k][j] + q * H[k + 1][j];
                        if (notlast) {
                            p = p + r * H[k + 2][j];
                            H[k + 2][j] = H[k + 2][j] - p * z;
                        }
                        H[k][j] -= p * x;
                        H[k + 1][j] -= p * y;
                    }

                    // Column modification

                    for (int i = 0; i <= std::min(n1, k + 3); i++) {
                        p = x * H[i][k] + y * H[i][k + 1];
                        if (notlast) {
                            p = p + z * H[i][k + 2];
                            H[i][k + 2] = H[i][k + 2] - p * r;
                        }
                        H[i][k] -= p;
                        H[i][k + 1] -= p * q;
                    }

                    // Accumulate transformations

                    for (int i = low; i <= high; i++) {
                        p = x * V[i][k] + y * V[i][k + 1];
                        if (notlast) {
                            p = p + z * V[i][k + 2];
                            V[i][k + 2] = V[i][k + 2] - p * r;
                        }
                        V[i][k] = V[i][k] - p;
                        V[i][k + 1] = V[i][k + 1] - p * q;
                    }
                } // (s != 0)
            } // k loop
        } // check convergence
    } // while (n1 >= low)

    // Backsubstitute to find vectors of upper triangular form

    if (norm < FLT_EPSILON) {
        return;
    }

    for (n1 = nn - 1; n1 >= 0; n1--) {
        double p = d[n1];
        double q = e[n1];

        if (q == 0) {
            // Real vector
            double z = std::numeric_limits<double>::quiet_NaN();
            double s = std::numeric_limits<double>::quiet_NaN();

            int l = n1;
            H[n1][n1] = 1.0;
            for (int i = n1 - 1; i >= 0; i--) {
                double w = H[i][i] - p;
                double r = 0.0;
                for (int j = l; j <= n1; j++) {
                    r = r + H[i][j] * H[j][n1];
                }
                if (e[i] < 0.0) {
                    z = w;
                    s = r;
                } else {
                    l = i;
                    if (e[i] == 0.0) {
                        if (w != 0.0) {
                            H[i][n1] = -r / w;
                        } else {
                            H[i][n1] = -r / (eps * norm);
                        }
                    } else {
                        // Solve real equations
                        CV_DbgAssert(!cvIsNaN(z));
                        double x = H[i][i + 1];
                        double y = H[i + 1][i];
                        q = (d[i] - p) * (d[i] - p) + e[i] * e[i];
                        double t = (x * s - z * r) / q;
                        H[i][n1] = t;
                        if (std::abs(x) > std::abs(z)) {
                            H[i + 1][n1] = (-r - w * t) / x;
                        } else {
                            CV_DbgAssert(z != 0.0);
                            H[i + 1][n1] = (-s - y * t) / z;
                        }
                    }

                    // Overflow control
                    double t = std::abs(H[i][n1]);
                    if ((eps * t) * t > 1) {
                        double inv_t = 1.0 / t;
                        for (int j = i; j <= n1; j++) {
                            H[j][n1] *= inv_t;
                        }
                    }
                }
            }
        } else if (q < 0) {
            // Complex vector
            double z = std::numeric_limits<double>::quiet_NaN();
            double r = std::numeric_limits<double>::quiet_NaN();
            double s = std::numeric_limits<double>::quiet_NaN();

            int l = n1 - 1;

            // Last vector component imaginary so matrix is triangular

            if (std::abs(H[n1][n1 - 1]) > std::abs(H[n1 - 1][n1])) {
                H[n1 - 1][n1 - 1] = q / H[n1][n1 - 1];
                H[n1 - 1][n1] = -(H[n1][n1] - p) / H[n1][n1 - 1];
            } else {
                complex_div(
                        0.0, -H[n1 - 1][n1],
                        H[n1 - 1][n1 - 1] - p, q,
                        H[n1 - 1][n1 - 1], H[n1 - 1][n1]
                );
            }
            H[n1][n1 - 1] = 0.0;
            H[n1][n1] = 1.0;
            for (int i = n1 - 2; i >= 0; i--) {
                double ra, sa, vr, vi;
                ra = 0.0;
                sa = 0.0;
                for (int j = l; j <= n1; j++) {
                    ra = ra + H[i][j] * H[j][n1 - 1];
                    sa = sa + H[i][j] * H[j][n1];
                }
                double w = H[i][i] - p;

                if (e[i] < 0.0) {
                    z = w;
                    r = ra;
                    s = sa;
                } else {
                    l = i;
                    if (e[i] == 0) {
                        complex_div(
                                -ra, -sa,
                                w, q,
                                H[i][n1 - 1], H[i][n1]
                        );
                    } else {
                        // Solve complex equations

                        double x = H[i][i + 1];
                        double y = H[i + 1][i];
                        vr = (d[i] - p) * (d[i] - p) + e[i] * e[i] - q * q;
                        vi = (d[i] - p) * 2.0 * q;
                        if (vr == 0.0 && vi == 0.0) {
                            vr = eps * norm * (std::abs(w) + std::abs(q) + std::abs(x)
                                               + std::abs(y) + std::abs(z));
                        }
                        complex_div(
                                x * r - z * ra + q * sa, x * s - z * sa - q * ra,
                                vr, vi,
                                H[i][n1 - 1], H[i][n1]);
                        if (std::abs(x) > (std::abs(z) + std::abs(q))) {
                            H[i + 1][n1 - 1] = (-ra - w * H[i][n1 - 1] + q
                                               * H[i][n1]) / x;
                            H[i + 1][n1] = (-sa - w * H[i][n1] - q * H[i][n1
                                                                        - 1]) / x;
                        } else {
                            complex_div(
                                    -r - y * H[i][n1 - 1], -s - y * H[i][n1],
                                    z, q,
                                    H[i + 1][n1 - 1], H[i + 1][n1]);
                        }
                    }

                    // Overflow control

                    double t = std::max(std::abs(H[i][n1 - 1]), std::abs(H[i][n1]));
                    if ((eps * t) * t > 1) {
                        for (int j = i; j <= n1; j++) {
                            H[j][n1 - 1] = H[j][n1 - 1] / t;
                            H[j][n1] = H[j][n1] / t;
                        }
                    }
                }
            }
        }
    }

    // Vectors of isolated roots

#if 0 // 'if' condition is always false
    for (int i = 0; i < nn; i++) {
        if (i < low || i > high) {
            for (int j = i; j < nn; j++) {
                V[i][j] = H[i][j];
            }
        }
    }
#endif

    // Back transformation to get eigenvectors of original matrix

    for (int j = nn - 1; j >= low; j--) {
        for (int i = low; i <= high; i++) {
            double z = 0.0;
            for (int k = low; k <= std::min(j, high); k++) {
                z += V[i][k] * H[k][j];
            }
            V[i][j] = z;
        }
    }
}

// —— lda.cpp :956-1004 eigenNonSymmetric 的 n=3/CV_64F 支转录
//    （eigenNonSymmetric 内部固定 compute(src64f, false)——恒走非对称路径；
//     本函数即 fitEllipseDirect 消费的 eigenNonSymmetric(M, eVal, eVec) 的 vendor 版）——
// 行序推导（自检项 3）：
//   compute() 写出 _eigenvectors(i,j) = V[i][j]（lda.cpp :895）
//   sorted_eigenvectors(i,j) = eigenvectors.t()(sort_indexes[i], j)（:991-1000）
//     = _eigenvectors(j, sort_indexes[i]) = V[j][sort_indexes[i]]
//   ⇒ evec 第 i 行 = V 的第 sortIdx[i] 列（列→行，memcpy 语义）
static void eigen3NonSymmetric(const cv::Matx33d& src, double eval[3], double evec[3][3])
{
    // compute(src, false) 支：src64f = src（CV_64F 同型免转换）→ H[i][j] = src(i,j)
    //（同型 convertTo 拷贝逐位保真；this->n = 3）
    double H[3][3], V[3][3], d[3], e[3], ort[3];
    const int n = 3;
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++)
            H[i][j] = src(i, j);

    orthes(n, H, V, ort);
    hqr2(n, H, V, d, e);

    // :977-985 排序——sortIdx 走公开 API（与 DLL 同一实现；d 即 eigenvalues 向量）
    Mat eigenvalues64f(1, 3, CV_64F, d);
    Mat sort_indexes(1, 3, CV_32S);
    cv::sortIdx(eigenvalues64f, sort_indexes, SORT_EVERY_ROW | SORT_DESCENDING);
    const int* sortIdx = sort_indexes.ptr<int>();

    // :984-985 sorted_eigenvalues64f[i] = eigenvalues64f[sort_indexes[i]]
    for (int i = 0; i < 3; i++)
        eval[i] = d[sortIdx[i]];

    // :989-1001 sorted_eigenvectors(i,·) = eigenvectors.t().row(sort_indexes[i])
    //   = V 的第 sortIdx[i] 列 → evec[i][j] = V[j][sortIdx[i]]
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++)
            evec[i][j] = V[j][sortIdx[i]];
}

// —— shapedescr.cpp :658-824 fitEllipseDirect 转录（适配点见文件头）——
cv::RotatedRect fitEllipse5Direct( const cv::Point2f* pts5 )
{
    // 【E3 适配】n=5 / depth=CV_32F 定死（含 depth 的 CV_Assert 恒真删除）
    int i;
    const int n = 5;
    float eps = 0;

    RotatedRect box;

    if( n < 5 )
        CV_Error( cv::Error::StsBadSize, "There should be at least 5 points to fit the ellipse" );

    Point2d c(0., 0.);

    // 【E3 适配】is_float==true（CV_32F 定死）：三元收敛为 ptsf[i]；ptsf 直取入参
    const Point2f* ptsf = pts5;

    // 【E3 适配】Mat A( n, 6, CV_64F) → 函数级 static thread_local 复用（稳态零分配）
    static thread_local Mat A;
    if( A.rows != 5 ) A.create( 5, 6, CV_64F );
    Matx<double, 6, 6> DM;
    Matx33d M, TM, Q;
    Matx<double, 3, 1> pVec;

    double x0, y0, a, b, theta, Ts;
    double s = 0;

    for( i = 0; i < n; i++ )
    {
        Point2f p = ptsf[i];
        c.x += p.x;
        c.y += p.y;
    }
    c.x /= n;
    c.y /= n;

    for( i = 0; i < n; i++ )
    {
        Point2f p = ptsf[i];
        s += fabs(p.x - c.x) + fabs(p.y - c.y);
    }
    double scale = 100./(s > FLT_EPSILON ? s : (double)FLT_EPSILON);

    // first, try the original pointset.
    // if it's singular, try to shift the points a bit
    int iter = 0;
    for( iter = 0; iter < 2; iter++ ) {
        for( i = 0; i < n; i++ )
        {
            Point2f p = ptsf[i];
            const Point2f delta = getOfs(eps);
            double px = (p.x + delta.x - c.x)*scale, py = (p.y + delta.y - c.y)*scale;

            A.at<double>(i,0) = px*px;
            A.at<double>(i,1) = px*py;
            A.at<double>(i,2) = py*py;
            A.at<double>(i,3) = px;
            A.at<double>(i,4) = py;
            A.at<double>(i,5) = 1.0;
        }
        // 【E3 适配】cv::mulTransposed( A, DM, true, noArray(), 1.0, -1 ) →
        // 手写 k 严格升序三重循环（scale=1.0d 精确等价跳过）：DM(i,j)=Σ_k A(k,i)*A(k,j)
        for( int ri = 0; ri < 6; ri++ )
            for( int ci = 0; ci <= ri; ci++ )
            {
                double sum = 0.;
                for( int k = 0; k < 5; k++ )
                    sum += A.at<double>(k,ri) * A.at<double>(k,ci);
                DM(ri,ci) = sum;
            }
        // completeSymm 语义：下三角已算，上三角镜像 DM(j,i)=DM(i,j)（i>j）
        for( int ri = 0; ri < 6; ri++ )
            for( int ci = 0; ci < ri; ci++ )
                DM(ci,ri) = DM(ri,ci);
        // 【E3 适配】DM *= (1.0/n) → 逐元素乘常量（保持乘法写法，n=5）
        for( int ri = 0; ri < 6; ri++ )
            for( int ci = 0; ci < 6; ci++ )
                DM(ri,ci) = DM(ri,ci) * (1.0/5.0);

        TM(0,0) = DM(0,5)*DM(3,5)*DM(4,4) - DM(0,5)*DM(3,4)*DM(4,5) - DM(0,4)*DM(3,5)*DM(5,4) + \
                  DM(0,3)*DM(4,5)*DM(5,4) + DM(0,4)*DM(3,4)*DM(5,5) - DM(0,3)*DM(4,4)*DM(5,5);
        TM(0,1) = DM(1,5)*DM(3,5)*DM(4,4) - DM(1,5)*DM(3,4)*DM(4,5) - DM(1,4)*DM(3,5)*DM(5,4) + \
                  DM(1,3)*DM(4,5)*DM(5,4) + DM(1,4)*DM(3,4)*DM(5,5) - DM(1,3)*DM(4,4)*DM(5,5);
        TM(0,2) = DM(2,5)*DM(3,5)*DM(4,4) - DM(2,5)*DM(3,4)*DM(4,5) - DM(2,4)*DM(3,5)*DM(5,4) + \
                  DM(2,3)*DM(4,5)*DM(5,4) + DM(2,4)*DM(3,4)*DM(5,5) - DM(2,3)*DM(4,4)*DM(5,5);
        TM(1,0) = DM(0,5)*DM(3,3)*DM(4,5) - DM(0,5)*DM(3,5)*DM(4,3) + DM(0,4)*DM(3,5)*DM(5,3) - \
                  DM(0,3)*DM(4,5)*DM(5,3) - DM(0,4)*DM(3,3)*DM(5,5) + DM(0,3)*DM(4,3)*DM(5,5);
        TM(1,1) = DM(1,5)*DM(3,3)*DM(4,5) - DM(1,5)*DM(3,5)*DM(4,3) + DM(1,4)*DM(3,5)*DM(5,3) - \
                  DM(1,3)*DM(4,5)*DM(5,3) - DM(1,4)*DM(3,3)*DM(5,5) + DM(1,3)*DM(4,3)*DM(5,5);
        TM(1,2) = DM(2,5)*DM(3,3)*DM(4,5) - DM(2,5)*DM(3,5)*DM(4,3) + DM(2,4)*DM(3,5)*DM(5,3) - \
                  DM(2,3)*DM(4,5)*DM(5,3) - DM(2,4)*DM(3,3)*DM(5,5) + DM(2,3)*DM(4,3)*DM(5,5);
        TM(2,0) = DM(0,5)*DM(3,4)*DM(4,3) - DM(0,5)*DM(3,3)*DM(4,4) - DM(0,4)*DM(3,4)*DM(5,3) + \
                  DM(0,3)*DM(4,4)*DM(5,3) + DM(0,4)*DM(3,3)*DM(5,4) - DM(0,3)*DM(4,3)*DM(5,4);
        TM(2,1) = DM(1,5)*DM(3,4)*DM(4,3) - DM(1,5)*DM(3,3)*DM(4,4) - DM(1,4)*DM(3,4)*DM(5,3) + \
                  DM(1,3)*DM(4,4)*DM(5,3) + DM(1,4)*DM(3,3)*DM(5,4) - DM(1,3)*DM(4,3)*DM(5,4);
        TM(2,2) = DM(2,5)*DM(3,4)*DM(4,3) - DM(2,5)*DM(3,3)*DM(4,4) - DM(2,4)*DM(3,4)*DM(5,3) + \
                  DM(2,3)*DM(4,4)*DM(5,3) + DM(2,4)*DM(3,3)*DM(5,4) - DM(2,3)*DM(4,3)*DM(5,4);

        Ts=(-(DM(3,5)*DM(4,4)*DM(5,3)) + DM(3,4)*DM(4,5)*DM(5,3) + DM(3,5)*DM(4,3)*DM(5,4) - \
              DM(3,3)*DM(4,5)*DM(5,4)  - DM(3,4)*DM(4,3)*DM(5,5) + DM(3,3)*DM(4,4)*DM(5,5));

        M(0,0) = (DM(2,0) + (DM(2,3)*TM(0,0) + DM(2,4)*TM(1,0) + DM(2,5)*TM(2,0))/Ts)/2.;
        M(0,1) = (DM(2,1) + (DM(2,3)*TM(0,1) + DM(2,4)*TM(1,1) + DM(2,5)*TM(2,1))/Ts)/2.;
        M(0,2) = (DM(2,2) + (DM(2,3)*TM(0,2) + DM(2,4)*TM(1,2) + DM(2,5)*TM(2,2))/Ts)/2.;
        M(1,0) = -DM(1,0) - (DM(1,3)*TM(0,0) + DM(1,4)*TM(1,0) + DM(1,5)*TM(2,0))/Ts;
        M(1,1) = -DM(1,1) - (DM(1,3)*TM(0,1) + DM(1,4)*TM(1,1) + DM(1,5)*TM(2,1))/Ts;
        M(1,2) = -DM(1,2) - (DM(1,3)*TM(0,2) + DM(1,4)*TM(1,2) + DM(1,5)*TM(2,2))/Ts;
        M(2,0) = (DM(0,0) + (DM(0,3)*TM(0,0) + DM(0,4)*TM(1,0) + DM(0,5)*TM(2,0))/Ts)/2.;
        M(2,1) = (DM(0,1) + (DM(0,3)*TM(0,1) + DM(0,4)*TM(1,1) + DM(0,5)*TM(2,1))/Ts)/2.;
        M(2,2) = (DM(0,2) + (DM(0,3)*TM(0,2) + DM(0,4)*TM(1,2) + DM(0,5)*TM(2,2))/Ts)/2.;

        double det = fabs(cv::determinant(M));
        if (fabs(det) > 1.0e-10)
            break;
        eps = (float)(s/(n*2)*1e-2);
    }

    if( iter < 2 ) {
        // 【E3 适配】Mat eVal, eVec; + eigenNonSymmetric(M, eVal, eVec)
        // → vendor eigen3NonSymmetric（输出双精度数组；eVec 行序＝原 eVec Mat 行序）
        double eVal[3];
        double eVec[3][3];
        eigen3NonSymmetric(M, eVal, eVec);

        // Select the eigen vector {a,b,c} which satisfies 4ac-b^2 > 0
        double cond[3];
        cond[0]=(4.0 * eVec[0][0] * eVec[0][2] - eVec[0][1] * eVec[0][1]);
        cond[1]=(4.0 * eVec[1][0] * eVec[1][2] - eVec[1][1] * eVec[1][1]);
        cond[2]=(4.0 * eVec[2][0] * eVec[2][2] - eVec[2][1] * eVec[2][1]);
        if (cond[0]<cond[1]) {
            i = (cond[1]<cond[2]) ? 2 : 1;
        } else {
            i = (cond[0]<cond[2]) ? 2 : 0;
        }
        double norm = std::sqrt(eVec[i][0]*eVec[i][0] + eVec[i][1]*eVec[i][1] + eVec[i][2]*eVec[i][2]);
        if (((eVec[i][0]<0.0  ? -1 : 1) * (eVec[i][1]<0.0  ? -1 : 1) * (eVec[i][2]<0.0  ? -1 : 1)) <= 0.0) {
                norm=-1.0*norm;
            }
        pVec(0) =eVec[i][0]/norm; pVec(1) =eVec[i][1]/norm;pVec(2) =eVec[i][2]/norm;

    //  Q = (TM . pVec)/Ts;
        Q(0,0) = (TM(0,0)*pVec(0) +TM(0,1)*pVec(1) +TM(0,2)*pVec(2) )/Ts;
        Q(0,1) = (TM(1,0)*pVec(0) +TM(1,1)*pVec(1) +TM(1,2)*pVec(2) )/Ts;
        Q(0,2) = (TM(2,0)*pVec(0) +TM(2,1)*pVec(1) +TM(2,2)*pVec(2) )/Ts;

    // We compute the ellipse properties in the shifted coordinates as doing so improves the numerical accuracy.

        double u1 = pVec(2)*Q(0,0)*Q(0,0) - pVec(1)*Q(0,0)*Q(0,1) + pVec(0)*Q(0,1)*Q(0,1) + pVec(1)*pVec(1)*Q(0,2);
        double u2 = pVec(0)*pVec(2)*Q(0,2);
        double l1 = sqrt(pVec(1)*pVec(1) + (pVec(0) - pVec(2))*(pVec(0) - pVec(2)));
        double l2 = pVec(0) + pVec(2) ;
        double l3 = pVec(1)*pVec(1) - 4*pVec(0)*pVec(2) ;
        double p1 = 2*pVec(2)*Q(0,0) - pVec(1)*Q(0,1);
        double p2 = 2*pVec(0)*Q(0,1) - pVec(1)*Q(0,0);

        x0 = (p1/l3/scale) + c.x;
        y0 = (p2/l3/scale) + c.y;
        a = sqrt(2.)*sqrt((u1 - 4.0*u2)/((l1 - l2)*l3))/scale;
        b = sqrt(2.)*sqrt(-1.0*((u1 - 4.0*u2)/((l1 + l2)*l3)))/scale;
        if (pVec(1)  == 0) {
            if (pVec(0)  < pVec(2) ) {
                theta = 0;
            } else {
                theta = CV_PI/2.;
            }
        } else {
                theta = CV_PI/2. + 0.5*std::atan2(pVec(1) , (pVec(0)  - pVec(2) ));
        }

        box.center.x = (float)x0;
        box.center.y = (float)y0;
        box.size.width = (float)(2.0*a);
        box.size.height = (float)(2.0*b);
        if( box.size.width > box.size.height )
        {
            float tmp;
            CV_SWAP( box.size.width, box.size.height, tmp );
            box.angle = (float)(fmod((90 + theta*180/CV_PI),180.0)) ;
        } else {
            box.angle = (float)(fmod(theta*180/CV_PI,180.0));
        };
    } else {
        box = fitEllipseNoDirect5( pts5 );
    }
    return box;
}

} // namespace calib
