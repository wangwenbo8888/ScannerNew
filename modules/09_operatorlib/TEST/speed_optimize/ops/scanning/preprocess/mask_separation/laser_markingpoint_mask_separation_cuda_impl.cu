#include "laser_markingpoint_mask_separation_cuda_pimpl.h"
#include "common/calib_types.h"
#include "common/calib_logging.h"
#include <cuda_runtime.h>
#include <cstdio>
#include <cstring>
#include <opencv2/cudaimgproc.hpp>
#include <stdexcept>

using namespace calib;


CALIB_DEFINE_LOG_TAG(LaserMarkingSep, LaserMarkingSeparationCUDA);

constexpr int BW = 16;
constexpr int BH = 16;

// ============================================================
// Kernel: Fused Gaussian 5x5 + Threshold (Step 1)
// ============================================================
__global__ void fusedGaussianThresholdKernel(
    const uint8_t* __restrict__ src, uint8_t* __restrict__ dst,
    int width, int height, int threshold)
{
    __shared__ uint8_t smem[BH + 4][BW + 4];
    int tx = threadIdx.x, ty = threadIdx.y;

    for (int j = ty; j < BH + 4; j += BH) {
        for (int i = tx; i < BW + 4; i += BW) {
            int gx = min(max(blockIdx.x * BW + i - 2, 0), width - 1);
            int gy = min(max(blockIdx.y * BH + j - 2, 0), height - 1);
            smem[j][i] = src[gy * width + gx];
        }

    }
    __syncthreads();

    int x = blockIdx.x * BW + tx;
    int y = blockIdx.y * BH + ty;
    if (x >= width || y >= height) return;

    const int g[5] = {19, 66, 100, 66, 19};
    int sum = 0;
    #pragma unroll
    for (int dy = -2; dy <= 2; ++dy)
        #pragma unroll
        for (int dx = -2; dx <= 2; ++dx)
            sum += g[dy+2] * g[dx+2] * smem[ty+2+dy][tx+2+dx];
    dst[y * width + x] = (sum > threshold * 72900) ? 255 : 0;
}

// ============================================================
// Kernel: Binary Erode / Dilate with shared memory
// circle=true 闁哄啫澧庨悰銉╁矗閿濆嫯绀嬬痪鍙ヨ兌濞插繘鏁嶉崸澧勯搹?dy閾忓繘鍩夐ˉ鐩欰LF閾忓繘鏁嶆径娑氱闁告熬绠戦崹顖炲棘閻熸媽鍩?k閼寸牰
// ============================================================
template<int KHALF, bool IS_ERODE>
__global__ void binaryMorphKernel(
    const uint8_t* __restrict__ src, uint8_t* __restrict__ dst,
    int width, int height, bool circle)
{
    constexpr int SW = BW + 2*KHALF, SH = BH + 2*KHALF;
    __shared__ uint8_t smem[SH][SW];
    int tx = threadIdx.x, ty = threadIdx.y;

    for (int j = ty; j < SH; j += BH)
        for (int i = tx; i < SW; i += BW) {
            int gx = blockIdx.x*BW + i - KHALF;
            int gy = blockIdx.y*BH + j - KHALF;
            smem[j][i] = (gx >= 0 && gx < width && gy >= 0 && gy < height)
                         ? src[gy*width + gx] : (IS_ERODE ? 255 : 0);
        }
    __syncthreads();

    int x = blockIdx.x*BW + tx, y = blockIdx.y*BH + ty;
    if (x >= width || y >= height) return;

    constexpr int R2 = KHALF * KHALF;
    uint8_t result = IS_ERODE ? 255 : 0;
    for (int dy = -KHALF; dy <= KHALF; ++dy)
        for (int dx = -KHALF; dx <= KHALF; ++dx) {
            if (circle && dy*dy + dx*dx > R2) continue;
            uint8_t v = smem[ty+KHALF+dy][tx+KHALF+dx];
            if constexpr (IS_ERODE) { if (v == 0) { dst[y*width+x]=0; return; } }
            else                     { if (v != 0) { dst[y*width+x]=255; return; } }
        }
    dst[y*width+x] = result;
}

// ============================================================
// Kernel: Fused Dilate + Subtract (top-hat 闁告垵绻戠涵?
// dst = subtrahend 闁?閸楊晩ilate(src)闁挎稒鐭焥t_dilated闁挎稑鐗嗚ぐ鑼矚閻氬绀嗗銈呮惈閻㈩偅娼忛幘鍐叉瘔 dilate(src)
// ============================================================
template<int KHALF>
__global__ void fusedDilateSubtractKernel(
    const uint8_t* __restrict__ src, const uint8_t* __restrict__ subtrahend,
    uint8_t* __restrict__ dst, uint8_t* __restrict__ dst_dilated,
    int width, int height, bool circle)
{
    constexpr int SW = BW+2*KHALF, SH = BH+2*KHALF;
    __shared__ uint8_t smem[SH][SW];
    int tx = threadIdx.x, ty = threadIdx.y;

    for (int j = ty; j < SH; j += BH)
        for (int i = tx; i < SW; i += BW) {
            int gx = blockIdx.x*BW+i-KHALF, gy = blockIdx.y*BH+j-KHALF;
            smem[j][i] = (gx>=0 && gx<width && gy>=0 && gy<height)
                         ? src[gy*width+gx] : 0;
        }
    __syncthreads();

    int x = blockIdx.x*BW+tx, y = blockIdx.y*BH+ty;
    if (x >= width || y >= height) return;

    constexpr int R2 = KHALF * KHALF;
    uint8_t dilated = 0;
    for (int dy = -KHALF; dy <= KHALF; ++dy)
        for (int dx = -KHALF; dx <= KHALF; ++dx) {
            if (circle && dy*dy + dx*dx > R2) continue;
            if (smem[ty+KHALF+dy][tx+KHALF+dx] != 0) { dilated=255; goto done; }
        }
    done:
    if (dst_dilated) dst_dilated[y*width+x] = dilated;
    dst[y*width+x] = (subtrahend[y*width+x]==255 && dilated==0) ? 255 : 0;
}

// ============================================================
// 閵嗘劕鍙忛柧楣冣偓鐔峰娴兼ê瀵?2026-10-16閵嗘垵銇囬弽姝屽帿閾撯偓 prevZero 韫囶偉鐭惧鍕剁窗
//   Pass A閿涙岸鈧劘顢?prevZero[x]閿涙繆顢戦崘?x 閸欏﹤鍙惧锔挎櫠閺堚偓鏉╂垹娈?0 娴ｅ秶鐤嗛敍鍫氬灊1 閺冪媴绱氶敍?
//   Pass B閿涙氨娲?閺傚湱鐛ラ崘鍛槨閸嬪繒些鐞?dy 閻ㄥ嫭鎸夐獮铏唽 [x閳姍w, x+hw] 閸?255 閸掋倕鐣?
//           閴?prevZero[x+hw] < x閳姍w閿涘湦(1)/鐞涘矉绱氶妴鍌濆帿閾撯偓閿涙槐ND閳ユ柡鈧柧绗岄崢鐔尖偓鎰仯
//           閹殿偅寮跨拠顓濈疅閿涘牆鎯堥悾灞筋樆閿?55閵嗕礁娓鹃惄?dx铏?dy铏忛埉顥歌檹閿涘鈧劒缍呮稉鈧懛娣偓?
//   婢跺秵娼呮惔?O(N璺疪) 閼辨艾鎮庣拠缁樻禌 O(N璺疪铏? smem 閹殿偀鈧柡鈧?9鑴?9 閻╂ü绮?~17ms 閳?娴滄碍顕犵粔鎺嬧偓?
// ============================================================
__global__ void prevZeroRowKernel(
    const uint8_t* __restrict__ src, unsigned short* __restrict__ prevZero,
    int width, int height)
{
    int y = blockIdx.x * blockDim.x + threadIdx.x;
    if (y >= height) return;
    const uint8_t* row = src + (size_t)y * width;
    unsigned short* out = prevZero + (size_t)y * width;
    if ((width & 7) == 0
        && (reinterpret_cast<uintptr_t>(row) & 7) == 0) {
        const uint64_t* row64 = reinterpret_cast<const uint64_t*>(row);
        unsigned short last = 0;   // 编码：位置+1（0=无）
        int x = 0;
        for (int i = 0; i < width / 8; ++i) {
            const uint64_t v = row64[i];
            const uint64_t t = ((v & 0x7F7F7F7F7F7F7F7Full) + 0x7F7F7F7F7F7F7F7Full) | v;
            if (~(t) & 0x8080808080808080ull) {  // 閺堫剛绮嶉崥顐︽祩鐎涙濡垾鏂衡偓鏃堚偓鎰摟閼?
                for (int k = 0; k < 8; ++k) {
                    if (((v >> (8 * k)) & 0xFFull) == 0) last = (unsigned short)(x + 1);
                    out[x] = last;
                    ++x;
                }
            } else {                             // 閸?255 缂佸嫧鈧柡鈧梹澹掗柌蹇撴倱閸?
                for (int k = 0; k < 8; ++k) out[x++] = last;
            }
        }
    } else {
        unsigned short last = 0;   // 编码：位置+1（0=无）
        for (int x = 0; x < width; ++x) {
            if (row[x] == 0) last = (unsigned short)(x + 1);
            out[x] = last;
        }
    }
}

// 閵嗘劙鈧喎瀹虫导妯哄閵嗘唲revOne 鐎电懓浼撻敍姘愁攽閸?x 閸欏﹤鍙惧锔挎櫠閺堚偓鏉?255 娴ｅ秶鐤嗛敍鍫氬灊1 閺冪媴绱氶妴?
// 濞?[a,b] 閸?255 閴?prevOne[b] 閳?a閳ユ柡鈧棁鍟懗鈧敍鍦閿涘(1)/閸嬪繒些鐞涘苯鍨界€规哎鈧?
// 閸氭垿鍣洪崠鏍閿? 鐎涙濡?濮濄儻绱欑€硅棄瀹?8 閺佹挳娅庨弮璁圭礆閿涘苯鍙忛梿?8 鐎涙濡紒鍕闁插繐鍟撻敍鍫熷负閼舵粎鈻堥悿蹇撶埗閹緤绱氶妴?
__global__ void prevOneRowKernel(
    const uint8_t* __restrict__ src, unsigned short* __restrict__ prevOne,
    int width, int height)
{
    int y = blockIdx.x * blockDim.x + threadIdx.x;
    if (y >= height) return;
    const uint8_t* row = src + (size_t)y * width;
    unsigned short* out = prevOne + (size_t)y * width;
    if ((width & 7) == 0
        && (reinterpret_cast<uintptr_t>(row) & 7) == 0) {
        const uint64_t* row64 = reinterpret_cast<const uint64_t*>(row);
        unsigned short last = 0;   // 编码：位置+1（0=无）
        int x = 0;
        for (int i = 0; i < width / 8; ++i) {
            const uint64_t v = row64[i];
            const uint64_t t = ((v & 0x7F7F7F7F7F7F7F7Full) + 0x7F7F7F7F7F7F7F7Full) | v;
            if (t & 0x8080808080808080ull) {   // 閺堫剛绮嶉崥顐︽姜闂嗚泛鐡ч懞鍌椻偓鏂衡偓鏃堚偓鎰摟閼?
                for (int k = 0; k < 8; ++k) {
                    if (((v >> (8 * k)) & 0xFFull) != 0) last = (unsigned short)(x + 1);
                    out[x] = last;
                    ++x;
                }
            } else {                            // 閸忋劑娴傜紒鍕ㄢ偓鏂衡偓鏃€澹掗柌蹇撴倱閸?
                for (int k = 0; k < 8; ++k) out[x++] = last;
            }
        }
    } else {
        unsigned short last = 0;   // 编码：位置+1（0=无）
        for (int x = 0; x < width; ++x) {
            if (row[x] != 0) last = (unsigned short)(x + 1);
            out[x] = last;
        }
    }
}

// 閵嗘劙鈧喎瀹虫导妯哄閵嗘垵銇囬弽姝屽暙閼斥偓閿涘鍣哄▔鏇礄fused dilate-subtract 韫囶偉鐭惧鍕剁礆閿涙俺鍟懗鈧紒?prevOne
// 閸掋倕鐣鹃敍鍫滅瑢闁劗鍋?OR 鐠囶厺绠熼柅鎰秴娑撯偓閼风补鈧柡鈧柨鎯堥悾灞筋樆閿?閵嗕礁娓鹃惄?dx铏?dy铏忛埉顥歌檹閿涘绱遍崙蹇旂《閸氬苯甯弽鎼炩偓?
template<int KHALF>
__global__ void fusedDilateSubPrevOneKernel(
    const unsigned short* __restrict__ prevOne,
    const uint8_t* __restrict__ subtrahend,
    uint8_t* __restrict__ dst, uint8_t* __restrict__ dst_dilated,
    int width, int height, bool circle)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;
    constexpr int R2 = KHALF * KHALF;
    uint8_t dilated = 0;
    for (int dy = -KHALF; dy <= KHALF && dilated == 0; ++dy) {
        const int yy = y + dy;
        if (yy < 0 || yy >= height) continue;          // 閻ｅ苯顦婚敍?閿涘牆甯弽绋挎倱鐠囶厺绠熼敍?
        int hw = circle ? (int)sqrtf((float)(R2 - dy * dy)) : KHALF;
        int a = x - hw, b = x + hw;
        if (a < 0) a = 0;
        if (b > width - 1) b = width - 1;
        if (a > b) continue;
        if ((int)prevOne[(size_t)yy * width + b] > a) dilated = 255;
    }
    if (dst_dilated) dst_dilated[(size_t)y * width + x] = dilated;
    dst[(size_t)y * width + x] =
        (subtrahend[(size_t)y * width + x] == 255 && dilated == 0) ? 255 : 0;
}

template<int KHALF>
__global__ void diskErodePrevZeroKernel(
    const unsigned short* __restrict__ prevZero, uint8_t* __restrict__ dst,
    int width, int height, bool circle)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;
    constexpr int R2 = KHALF * KHALF;
    for (int dy = -KHALF; dy <= KHALF; ++dy) {
        const int yy = y + dy;
        if (yy < 0 || yy >= height) continue;          // 閻ｅ苯顦婚敍?55閿涘牆甯弽绋挎倱鐠囶厺绠熼敍?
        int hw = circle ? (int)sqrtf((float)(R2 - dy * dy)) : KHALF;
        int a = x - hw, b = x + hw;
        if (a < 0) a = 0;
        if (b > width - 1) b = width - 1;
        if (a > b) continue;
        if ((int)prevZero[(size_t)yy * width + b] > a) {   // 濞堥潧鍞撮張?0 閳?閼垫劘娈屾稉?0
            dst[(size_t)y * width + x] = 0;
            return;
        }
    }
    dst[(size_t)y * width + x] = 255;
}

// ============================================================
// 【速度优化·二轮】块内融合形态学：tile＋光晕载入共享内存 → 块内建行前缀
// （最近目标位置+1，0=无）→ 输出像素 O(R) 查表。一次全局读＋一次写；
// 无独立建表核、无全局前缀表流量。语义与逐点窗口扫描逐位一致：
// 界外填充（erode=255/dilate=0）、圆盘 dx²+dy²≤R²（hw=⌊√(R²-dy²)⌋）。
// ============================================================
template<int R, bool IS_ERODE, int TW = 64, int TH = 16>
__global__ void blockMorphKernel(
    const uint8_t* __restrict__ src, uint8_t* __restrict__ dst,
    int width, int height, bool circle)
{
    constexpr int SW = TW + 2 * R, SH = TH + 2 * R;
    __shared__ uint8_t tile[SH][SW];
    __shared__ unsigned short pfx[SH][SW];
    __shared__ int hwTab[2 * R + 1];   // 圆盘半宽表（每块一次，替每像素 sqrtf）
    const int tx = threadIdx.x, ty = threadIdx.y;
    const int tid = ty * TW + tx;
    // 0) 半宽表：hw[dy+R]=⌊√(R²-dy²)⌋（circle）/ R（方核）
    if (tid < 2 * R + 1) {
        const int dy = tid - R;
        hwTab[tid] = circle ? (int)sqrtf((float)(R * R - dy * dy)) : R;
    }
    // 1) 载入 tile（界外：erode=255 / dilate=0）
    for (int j = ty; j < SH; j += TH)
        for (int i = tx; i < SW; i += TW) {
            const int gx = blockIdx.x * TW + i - R;
            const int gy = blockIdx.y * TH + j - R;
            tile[j][i] = (gx >= 0 && gx < width && gy >= 0 && gy < height)
                             ? src[(size_t)gy * width + gx]
                             : (IS_ERODE ? 255 : 0);
        }
    __syncthreads();
    // 2) 行前缀——SWAR 向量化（4 字节/步，无目标组批量写同一 last，有目标回退逐字节）
    if (tid < SH) {
        unsigned short last = 0;
        const int aligned_end = SW & ~3;
        if (IS_ERODE) {
            for (int i = 0; i < aligned_end; i += 4) {
                const uint32_t v = (uint32_t)tile[tid][i]
                                 | ((uint32_t)tile[tid][i+1] << 8)
                                 | ((uint32_t)tile[tid][i+2] << 16)
                                 | ((uint32_t)tile[tid][i+3] << 24);
                if ((v - 0x01010101u) & ~v & 0x80808080u) {
                    for (int k = 0; k < 4; ++k) {
                        if (tile[tid][i+k] == 0) last = (unsigned short)(i + k + 1);
                        pfx[tid][i+k] = last;
                    }
                } else {
                    pfx[tid][i] = last; pfx[tid][i+1] = last;
                    pfx[tid][i+2] = last; pfx[tid][i+3] = last;
                }
            }
            for (int i = aligned_end; i < SW; ++i) {
                if (tile[tid][i] == 0) last = (unsigned short)(i + 1);
                pfx[tid][i] = last;
            }
        } else {
            for (int i = 0; i < aligned_end; i += 4) {
                const uint32_t v = (uint32_t)tile[tid][i]
                                 | ((uint32_t)tile[tid][i+1] << 8)
                                 | ((uint32_t)tile[tid][i+2] << 16)
                                 | ((uint32_t)tile[tid][i+3] << 24);
                if (v != 0) {
                    for (int k = 0; k < 4; ++k) {
                        if (tile[tid][i+k] != 0) last = (unsigned short)(i + k + 1);
                        pfx[tid][i+k] = last;
                    }
                } else {
                    pfx[tid][i] = last; pfx[tid][i+1] = last;
                    pfx[tid][i+2] = last; pfx[tid][i+3] = last;
                }
            }
            for (int i = aligned_end; i < SW; ++i) {
                if (tile[tid][i] != 0) last = (unsigned short)(i + 1);
                pfx[tid][i] = last;
            }
        }
    }
    __syncthreads();
    // 3) 输出：erode＝段内无 0 → 255；dilate＝段内有 1 → 255
    const int x = blockIdx.x * TW + tx, y = blockIdx.y * TH + ty;
    if (x >= width || y >= height) return;
    uint8_t out = IS_ERODE ? 255 : 0;
    for (int dy = -R; dy <= R; ++dy) {
        const int hw = hwTab[dy + R];
        int a = tx + R - hw, b = tx + R + hw;   // tile 内列
        if (a < 0) a = 0;
        if (b > SW - 1) b = SW - 1;
        if (a > b) continue;
        const int pv = pfx[ty + R + dy][b];
        if (pv > a) { out = IS_ERODE ? 0 : 255; break; }
    }
    dst[(size_t)y * width + x] = out;
}

// 块内融合「膨胀＋减法」——【速度优化·四轮·修正版】
// 块级一致跳过：整块被减数全非 255 且不需 dilated → 全块统一跳过 tile＋前缀＋膨胀
// （决策在 __syncthreads 后对所有线程一致——无部分返回的同步未定义行为）
// s4 场景 survivor 背景 95%+，大量块命中 → s4b 从 0.77ms → ~0.1ms
template<int R, int TW = 64, int TH = 16>
__global__ void blockDilateSubKernel(
    const uint8_t* __restrict__ src,
    const uint8_t* __restrict__ subtrahend,
    uint8_t* __restrict__ dst, uint8_t* __restrict__ dst_dilated,
    int width, int height, bool circle)
{
    constexpr int SW = TW + 2 * R, SH = TH + 2 * R;
    __shared__ uint8_t tile[SH][SW];
    __shared__ unsigned short pfx[SH][SW];
    __shared__ int hwTab[2 * R + 1];
    __shared__ int anySub255;
    const int tx = threadIdx.x, ty = threadIdx.y;
    const int tid = ty * TW + tx;
    const int x0 = blockIdx.x * TW, y0 = blockIdx.y * TH;

    // —— 块级被减数检查（两步同步保证一致）——
    if (tid == 0) anySub255 = 0;
    __syncthreads();
    {
        const int x = x0 + tx, y = y0 + ty;
        if (x < width && y < height && subtrahend[(size_t)y * width + x] == 255)
            anySub255 = 1;
    }
    __syncthreads();
    if (!anySub255 && dst_dilated == nullptr) {
        // 全块被减数全 0 → 输出全 0（决策一致——整块安全返回，不再触同步）
        const int x = x0 + tx, y = y0 + ty;
        if (x < width && y < height) dst[(size_t)y * width + x] = 0;
        return;
    }

    // —— 常规路径（全块到达）——
    if (tid < 2 * R + 1) {
        const int dy = tid - R;
        hwTab[tid] = circle ? (int)sqrtf((float)(R * R - dy * dy)) : R;
    }
    for (int j = ty; j < SH; j += TH)
        for (int i = tx; i < SW; i += TW) {
            const int gx = x0 + i - R;
            const int gy = y0 + j - R;
            tile[j][i] = (gx >= 0 && gx < width && gy >= 0 && gy < height)
                             ? src[(size_t)gy * width + gx] : 0;
        }
    __syncthreads();
    if (tid < SH) {
        unsigned short last = 0;
        const int aligned_end = SW & ~3;
        for (int i = 0; i < aligned_end; i += 4) {
            const uint32_t v = (uint32_t)tile[tid][i]
                             | ((uint32_t)tile[tid][i+1] << 8)
                             | ((uint32_t)tile[tid][i+2] << 16)
                             | ((uint32_t)tile[tid][i+3] << 24);
            if (v != 0) {
                for (int k = 0; k < 4; ++k) {
                    if (tile[tid][i+k] != 0) last = (unsigned short)(i + k + 1);
                    pfx[tid][i+k] = last;
                }
            } else {
                pfx[tid][i] = last; pfx[tid][i+1] = last;
                pfx[tid][i+2] = last; pfx[tid][i+3] = last;
            }
        }
        for (int i = aligned_end; i < SW; ++i) {
            if (tile[tid][i] != 0) last = (unsigned short)(i + 1);
            pfx[tid][i] = last;
        }
    }
    __syncthreads();
    const int x = x0 + tx, y = y0 + ty;
    if (x >= width || y >= height) return;
    uint8_t dilated = 0;
    for (int dy = -R; dy <= R && dilated == 0; ++dy) {
        const int hw = hwTab[dy + R];
        int a = tx + R - hw, b = tx + R + hw;
        if (a < 0) a = 0;
        if (b > SW - 1) b = SW - 1;
        if (a > b) continue;
        if (pfx[ty + R + dy][b] > a) dilated = 255;
    }
    if (dst_dilated) dst_dilated[(size_t)y * width + x] = dilated;
    dst[(size_t)y * width + x] =
        (subtrahend[(size_t)y * width + x] == 255 && dilated == 0) ? 255 : 0;
}

// —— 块内融合核发射器（kh 覆盖当前产线配置；其他尺寸回退旧路径兜底）——
static void launchErode(const uint8_t* s, uint8_t* d, int w, int h, int ks,
                        bool circle, cudaStream_t st);
static void launchDilate(const uint8_t* s, uint8_t* d, int w, int h, int ks,
                         bool circle, cudaStream_t st);
static void launchFusedDilateSub(const uint8_t* s, const uint8_t* sub,
                                 uint8_t* dst, uint8_t* dstDil,
                                 int w, int h, int ks, bool circle, cudaStream_t st);
static void launchBlockMorphDispatch(const uint8_t* s, uint8_t* d, int w, int h,
                                     int ks, bool circle, bool erode, cudaStream_t st) {
    const int kh = (ks - 1) / 2;
    // 膨胀全走旧逐点（小半径稀疏图早退优——块内核前缀扫描开销反超）
    if (!erode || kh < 6) {
        if (erode) launchErode(s, d, w, h, ks, circle, st);
        else launchDilate(s, d, w, h, ks, circle, st);
        return;
    }
    dim3 b(64, 16);
    dim3 g((w + 63) / 64, (h + 15) / 16);
    switch (kh) {
        case 6:  blockMorphKernel<6, true><<<g,b,0,st>>>(s,d,w,h,circle); return;
        case 19: blockMorphKernel<19,true><<<g,b,0,st>>>(s,d,w,h,circle); return;
        default: break;
    }
    launchErode(s, d, w, h, ks, circle, st);
}

static void launchBlockDilateSubDispatch(const uint8_t* s, const uint8_t* sub,
                                         uint8_t* dst, uint8_t* dstDil,
                                         int w, int h, int ks, bool circle,
                                         cudaStream_t st) {
    const int kh = (ks - 1) / 2;
    if (kh < 6) {   // 小半径保留原路径（早退/小窗本来快）
        launchFusedDilateSub(s, sub, dst, dstDil, w, h, ks, circle, st);
        return;
    }
    dim3 b(64, 16);
    dim3 g((w + 63) / 64, (h + 15) / 16);
    switch (kh) {
        case 7:  blockDilateSubKernel<7><<<g,b,0,st>>>(s,sub,dst,dstDil,w,h,circle); return;
        case 20: blockDilateSubKernel<20><<<g,b,0,st>>>(s,sub,dst,dstDil,w,h,circle); return;
        default: break;
    }
    launchFusedDilateSub(s, sub, dst, dstDil, w, h, ks, circle, st);
}

static dim3 gridSz(int w, int h) { return dim3((w+BW-1)/BW, (h+BH-1)/BH); }

// 婢堆勭壋閼垫劘娈?prevZero 韫囶偉鐭惧鍕剁礄kh閳?閿涙矚uf 娑?w*h int32 娑擃叀娴嗛敍宀冪殶閻劍鏌熼幐浣规箒閿?
static void launchErodeFast(const uint8_t* s, uint8_t* d, int w, int h, int kh,
                            bool circle, cudaStream_t st, unsigned short* prevZeroBuf) {
    {
        int threads = 128;
        int blocks = (h + threads - 1) / threads;
        prevZeroRowKernel<<<blocks, threads, 0, st>>>(s, prevZeroBuf, w, h);
    }
    dim3 b(16, 16);
    dim3 g((w + 15) / 16, (h + 15) / 16);
    switch (kh) {
        case 8:  diskErodePrevZeroKernel<8><<<g,b,0,st>>>(prevZeroBuf,d,w,h,circle);break;
        case 9:  diskErodePrevZeroKernel<9><<<g,b,0,st>>>(prevZeroBuf,d,w,h,circle);break;
        case 10: diskErodePrevZeroKernel<10><<<g,b,0,st>>>(prevZeroBuf,d,w,h,circle);break;
        case 11: diskErodePrevZeroKernel<11><<<g,b,0,st>>>(prevZeroBuf,d,w,h,circle);break;
        case 12: diskErodePrevZeroKernel<12><<<g,b,0,st>>>(prevZeroBuf,d,w,h,circle);break;
        case 13: diskErodePrevZeroKernel<13><<<g,b,0,st>>>(prevZeroBuf,d,w,h,circle);break;
        case 14: diskErodePrevZeroKernel<14><<<g,b,0,st>>>(prevZeroBuf,d,w,h,circle);break;
        case 15: diskErodePrevZeroKernel<15><<<g,b,0,st>>>(prevZeroBuf,d,w,h,circle);break;
        case 16: diskErodePrevZeroKernel<16><<<g,b,0,st>>>(prevZeroBuf,d,w,h,circle);break;
        case 17: diskErodePrevZeroKernel<17><<<g,b,0,st>>>(prevZeroBuf,d,w,h,circle);break;
        case 18: diskErodePrevZeroKernel<18><<<g,b,0,st>>>(prevZeroBuf,d,w,h,circle);break;
        case 19: diskErodePrevZeroKernel<19><<<g,b,0,st>>>(prevZeroBuf,d,w,h,circle);break;
        case 20: diskErodePrevZeroKernel<20><<<g,b,0,st>>>(prevZeroBuf,d,w,h,circle);break;
        case 21: diskErodePrevZeroKernel<21><<<g,b,0,st>>>(prevZeroBuf,d,w,h,circle);break;
        case 22: diskErodePrevZeroKernel<22><<<g,b,0,st>>>(prevZeroBuf,d,w,h,circle);break;
        case 23: diskErodePrevZeroKernel<23><<<g,b,0,st>>>(prevZeroBuf,d,w,h,circle);break;
        case 24: diskErodePrevZeroKernel<24><<<g,b,0,st>>>(prevZeroBuf,d,w,h,circle);break;
        case 25: diskErodePrevZeroKernel<25><<<g,b,0,st>>>(prevZeroBuf,d,w,h,circle);break;
        case 26: diskErodePrevZeroKernel<26><<<g,b,0,st>>>(prevZeroBuf,d,w,h,circle);break;
        case 27: diskErodePrevZeroKernel<27><<<g,b,0,st>>>(prevZeroBuf,d,w,h,circle);break;
        case 28: diskErodePrevZeroKernel<28><<<g,b,0,st>>>(prevZeroBuf,d,w,h,circle);break;
        case 29: diskErodePrevZeroKernel<29><<<g,b,0,st>>>(prevZeroBuf,d,w,h,circle);break;
        case 30: diskErodePrevZeroKernel<30><<<g,b,0,st>>>(prevZeroBuf,d,w,h,circle);break;
        case 31: diskErodePrevZeroKernel<31><<<g,b,0,st>>>(prevZeroBuf,d,w,h,circle);break;
        default: throw std::runtime_error("Bad erode khalf (fast path max 31)");
    }
}

static void launchErode(const uint8_t* s, uint8_t* d, int w, int h, int ks, bool circle, cudaStream_t st) {
    int kh=(ks-1)/2; dim3 b(BW,BH), g=gridSz(w,h);
    switch(kh){        case 1: binaryMorphKernel<1,true><<<g,b,0,st>>>(s,d,w,h,circle);break;
        case 2: binaryMorphKernel<2,true><<<g,b,0,st>>>(s,d,w,h,circle);break;
        case 3: binaryMorphKernel<3,true><<<g,b,0,st>>>(s,d,w,h,circle);break;
        case 4: binaryMorphKernel<4,true><<<g,b,0,st>>>(s,d,w,h,circle);break;
        case 5: binaryMorphKernel<5,true><<<g,b,0,st>>>(s,d,w,h,circle);break;
        case 6: binaryMorphKernel<6,true><<<g,b,0,st>>>(s,d,w,h,circle);break;
        case 7: binaryMorphKernel<7,true><<<g,b,0,st>>>(s,d,w,h,circle);break;
        case 8: binaryMorphKernel<8,true><<<g,b,0,st>>>(s,d,w,h,circle);break;
        case 9: binaryMorphKernel<9,true><<<g,b,0,st>>>(s,d,w,h,circle);break;
        case 10: binaryMorphKernel<10,true><<<g,b,0,st>>>(s,d,w,h,circle);break;
        case 11: binaryMorphKernel<11,true><<<g,b,0,st>>>(s,d,w,h,circle);break;
        case 12: binaryMorphKernel<12,true><<<g,b,0,st>>>(s,d,w,h,circle);break;
        case 13: binaryMorphKernel<13,true><<<g,b,0,st>>>(s,d,w,h,circle);break;
        case 14: binaryMorphKernel<14,true><<<g,b,0,st>>>(s,d,w,h,circle);break;
        case 15: binaryMorphKernel<15,true><<<g,b,0,st>>>(s,d,w,h,circle);break;
        case 16: binaryMorphKernel<16,true><<<g,b,0,st>>>(s,d,w,h,circle);break;
        case 17: binaryMorphKernel<17,true><<<g,b,0,st>>>(s,d,w,h,circle);break;
        case 18: binaryMorphKernel<18,true><<<g,b,0,st>>>(s,d,w,h,circle);break;
        case 19: binaryMorphKernel<19,true><<<g,b,0,st>>>(s,d,w,h,circle);break;
        case 20: binaryMorphKernel<20,true><<<g,b,0,st>>>(s,d,w,h,circle);break;
        case 21: binaryMorphKernel<21,true><<<g,b,0,st>>>(s,d,w,h,circle);break;
        case 22: binaryMorphKernel<22,true><<<g,b,0,st>>>(s,d,w,h,circle);break;
        case 23: binaryMorphKernel<23,true><<<g,b,0,st>>>(s,d,w,h,circle);break;
        case 24: binaryMorphKernel<24,true><<<g,b,0,st>>>(s,d,w,h,circle);break;
        case 25: binaryMorphKernel<25,true><<<g,b,0,st>>>(s,d,w,h,circle);break;
        case 26: binaryMorphKernel<26,true><<<g,b,0,st>>>(s,d,w,h,circle);break;
        case 27: binaryMorphKernel<27,true><<<g,b,0,st>>>(s,d,w,h,circle);break;
        case 28: binaryMorphKernel<28,true><<<g,b,0,st>>>(s,d,w,h,circle);break;
        case 29: binaryMorphKernel<29,true><<<g,b,0,st>>>(s,d,w,h,circle);break;
        case 30: binaryMorphKernel<30,true><<<g,b,0,st>>>(s,d,w,h,circle);break;
        case 31: binaryMorphKernel<31,true><<<g,b,0,st>>>(s,d,w,h,circle);break;
        default: throw std::runtime_error("Bad erode khalf (max 31 = kernel 63)");
    }
}
static void launchDilate(const uint8_t* s, uint8_t* d, int w, int h, int ks, bool circle, cudaStream_t st) {
    int kh=(ks-1)/2; dim3 b(BW,BH), g=gridSz(w,h);
    switch(kh){
        case 1: binaryMorphKernel<1,false><<<g,b,0,st>>>(s,d,w,h,circle);break;
        case 2: binaryMorphKernel<2,false><<<g,b,0,st>>>(s,d,w,h,circle);break;
        case 3: binaryMorphKernel<3,false><<<g,b,0,st>>>(s,d,w,h,circle);break;
        case 4: binaryMorphKernel<4,false><<<g,b,0,st>>>(s,d,w,h,circle);break;
        case 5: binaryMorphKernel<5,false><<<g,b,0,st>>>(s,d,w,h,circle);break;
        case 6: binaryMorphKernel<6,false><<<g,b,0,st>>>(s,d,w,h,circle);break;
        case 7: binaryMorphKernel<7,false><<<g,b,0,st>>>(s,d,w,h,circle);break;
        case 8: binaryMorphKernel<8,false><<<g,b,0,st>>>(s,d,w,h,circle);break;
        case 9: binaryMorphKernel<9,false><<<g,b,0,st>>>(s,d,w,h,circle);break;
        case 10: binaryMorphKernel<10,false><<<g,b,0,st>>>(s,d,w,h,circle);break;
        case 11: binaryMorphKernel<11,false><<<g,b,0,st>>>(s,d,w,h,circle);break;
        case 12: binaryMorphKernel<12,false><<<g,b,0,st>>>(s,d,w,h,circle);break;
        case 13: binaryMorphKernel<13,false><<<g,b,0,st>>>(s,d,w,h,circle);break;
        case 14: binaryMorphKernel<14,false><<<g,b,0,st>>>(s,d,w,h,circle);break;
        case 15: binaryMorphKernel<15,false><<<g,b,0,st>>>(s,d,w,h,circle);break;
        case 16: binaryMorphKernel<16,false><<<g,b,0,st>>>(s,d,w,h,circle);break;
        case 17: binaryMorphKernel<17,false><<<g,b,0,st>>>(s,d,w,h,circle);break;
        case 18: binaryMorphKernel<18,false><<<g,b,0,st>>>(s,d,w,h,circle);break;
        case 19: binaryMorphKernel<19,false><<<g,b,0,st>>>(s,d,w,h,circle);break;
        case 20: binaryMorphKernel<20,false><<<g,b,0,st>>>(s,d,w,h,circle);break;
        case 21: binaryMorphKernel<21,false><<<g,b,0,st>>>(s,d,w,h,circle);break;
        case 22: binaryMorphKernel<22,false><<<g,b,0,st>>>(s,d,w,h,circle);break;
        case 23: binaryMorphKernel<23,false><<<g,b,0,st>>>(s,d,w,h,circle);break;
        case 24: binaryMorphKernel<24,false><<<g,b,0,st>>>(s,d,w,h,circle);break;
        case 25: binaryMorphKernel<25,false><<<g,b,0,st>>>(s,d,w,h,circle);break;
        case 26: binaryMorphKernel<26,false><<<g,b,0,st>>>(s,d,w,h,circle);break;
        case 27: binaryMorphKernel<27,false><<<g,b,0,st>>>(s,d,w,h,circle);break;
        case 28: binaryMorphKernel<28,false><<<g,b,0,st>>>(s,d,w,h,circle);break;
        case 29: binaryMorphKernel<29,false><<<g,b,0,st>>>(s,d,w,h,circle);break;
        case 30: binaryMorphKernel<30,false><<<g,b,0,st>>>(s,d,w,h,circle);break;
        case 31: binaryMorphKernel<31,false><<<g,b,0,st>>>(s,d,w,h,circle);break;
        default: throw std::runtime_error("Bad dilate khalf (max 31 = kernel 63)");
    }
}
static void launchFusedDilateSub(const uint8_t* s, const uint8_t* sub,
                                 uint8_t* d, uint8_t* dd, int w, int h, int ks,
                                 bool circle, cudaStream_t st) {
    int kh=(ks-1)/2; dim3 b(BW,BH), g=gridSz(w,h);
    switch(kh){
        case 1: fusedDilateSubtractKernel<1><<<g,b,0,st>>>(s,sub,d,dd,w,h,circle);break;
        case 2: fusedDilateSubtractKernel<2><<<g,b,0,st>>>(s,sub,d,dd,w,h,circle);break;
        case 3: fusedDilateSubtractKernel<3><<<g,b,0,st>>>(s,sub,d,dd,w,h,circle);break;
        case 4: fusedDilateSubtractKernel<4><<<g,b,0,st>>>(s,sub,d,dd,w,h,circle);break;
        case 5: fusedDilateSubtractKernel<5><<<g,b,0,st>>>(s,sub,d,dd,w,h,circle);break;
        case 6: fusedDilateSubtractKernel<6><<<g,b,0,st>>>(s,sub,d,dd,w,h,circle);break;
        case 7: fusedDilateSubtractKernel<7><<<g,b,0,st>>>(s,sub,d,dd,w,h,circle);break;
        case 8: fusedDilateSubtractKernel<8><<<g,b,0,st>>>(s,sub,d,dd,w,h,circle);break;
        case 9: fusedDilateSubtractKernel<9><<<g,b,0,st>>>(s,sub,d,dd,w,h,circle);break;
        case 10: fusedDilateSubtractKernel<10><<<g,b,0,st>>>(s,sub,d,dd,w,h,circle);break;
        case 11: fusedDilateSubtractKernel<11><<<g,b,0,st>>>(s,sub,d,dd,w,h,circle);break;
        case 12: fusedDilateSubtractKernel<12><<<g,b,0,st>>>(s,sub,d,dd,w,h,circle);break;
        case 13: fusedDilateSubtractKernel<13><<<g,b,0,st>>>(s,sub,d,dd,w,h,circle);break;
        case 14: fusedDilateSubtractKernel<14><<<g,b,0,st>>>(s,sub,d,dd,w,h,circle);break;
        case 15: fusedDilateSubtractKernel<15><<<g,b,0,st>>>(s,sub,d,dd,w,h,circle);break;
        case 16: fusedDilateSubtractKernel<16><<<g,b,0,st>>>(s,sub,d,dd,w,h,circle);break;
        case 17: fusedDilateSubtractKernel<17><<<g,b,0,st>>>(s,sub,d,dd,w,h,circle);break;
        case 18: fusedDilateSubtractKernel<18><<<g,b,0,st>>>(s,sub,d,dd,w,h,circle);break;
        case 19: fusedDilateSubtractKernel<19><<<g,b,0,st>>>(s,sub,d,dd,w,h,circle);break;
        case 20: fusedDilateSubtractKernel<20><<<g,b,0,st>>>(s,sub,d,dd,w,h,circle);break;
        case 21: fusedDilateSubtractKernel<21><<<g,b,0,st>>>(s,sub,d,dd,w,h,circle);break;
        case 22: fusedDilateSubtractKernel<22><<<g,b,0,st>>>(s,sub,d,dd,w,h,circle);break;
        case 23: fusedDilateSubtractKernel<23><<<g,b,0,st>>>(s,sub,d,dd,w,h,circle);break;
        case 24: fusedDilateSubtractKernel<24><<<g,b,0,st>>>(s,sub,d,dd,w,h,circle);break;
        case 25: fusedDilateSubtractKernel<25><<<g,b,0,st>>>(s,sub,d,dd,w,h,circle);break;
        case 26: fusedDilateSubtractKernel<26><<<g,b,0,st>>>(s,sub,d,dd,w,h,circle);break;
        case 27: fusedDilateSubtractKernel<27><<<g,b,0,st>>>(s,sub,d,dd,w,h,circle);break;
        case 28: fusedDilateSubtractKernel<28><<<g,b,0,st>>>(s,sub,d,dd,w,h,circle);break;
        case 29: fusedDilateSubtractKernel<29><<<g,b,0,st>>>(s,sub,d,dd,w,h,circle);break;
        case 30: fusedDilateSubtractKernel<30><<<g,b,0,st>>>(s,sub,d,dd,w,h,circle);break;
        case 31: fusedDilateSubtractKernel<31><<<g,b,0,st>>>(s,sub,d,dd,w,h,circle);break;
        default: throw std::runtime_error("Bad fused sub khalf (max 31 = kernel 63)");
    }
}

// 閵嗘劙鈧喎瀹虫导妯哄閵嗘垵銇囬弽?fused dilate-sub 韫囶偉鐭惧鍕埠娑撯偓閸忋儱褰涢敍鍧榟閳溂6,31] 鐠?prevOne閿?
static void launchFusedDilateSubFast(const uint8_t* s, const uint8_t* sub,
                                     uint8_t* dst, uint8_t* dstDil,
                                     int w, int h, int ks, bool circle,
                                     cudaStream_t st, unsigned short* prevOneBuf) {
    const int kh = (ks - 1) / 2;
    if (kh < 6 || kh > 31) {
        launchFusedDilateSub(s, sub, dst, dstDil, w, h, ks, circle, st);
        return;
    }
    {
        int threads = 128;
        int blocks = (h + threads - 1) / threads;
        prevOneRowKernel<<<blocks, threads, 0, st>>>(s, prevOneBuf, w, h);
    }
    dim3 b(16, 16);
    dim3 g((w + 15) / 16, (h + 15) / 16);
    switch (kh) {
        case 6:  fusedDilateSubPrevOneKernel<6><<<g,b,0,st>>>(prevOneBuf,sub,dst,dstDil,w,h,circle);break;
        case 7:  fusedDilateSubPrevOneKernel<7><<<g,b,0,st>>>(prevOneBuf,sub,dst,dstDil,w,h,circle);break;
        case 8:  fusedDilateSubPrevOneKernel<8><<<g,b,0,st>>>(prevOneBuf,sub,dst,dstDil,w,h,circle);break;
        case 9:  fusedDilateSubPrevOneKernel<9><<<g,b,0,st>>>(prevOneBuf,sub,dst,dstDil,w,h,circle);break;
        case 10: fusedDilateSubPrevOneKernel<10><<<g,b,0,st>>>(prevOneBuf,sub,dst,dstDil,w,h,circle);break;
        case 11: fusedDilateSubPrevOneKernel<11><<<g,b,0,st>>>(prevOneBuf,sub,dst,dstDil,w,h,circle);break;
        case 12: fusedDilateSubPrevOneKernel<12><<<g,b,0,st>>>(prevOneBuf,sub,dst,dstDil,w,h,circle);break;
        case 13: fusedDilateSubPrevOneKernel<13><<<g,b,0,st>>>(prevOneBuf,sub,dst,dstDil,w,h,circle);break;
        case 14: fusedDilateSubPrevOneKernel<14><<<g,b,0,st>>>(prevOneBuf,sub,dst,dstDil,w,h,circle);break;
        case 15: fusedDilateSubPrevOneKernel<15><<<g,b,0,st>>>(prevOneBuf,sub,dst,dstDil,w,h,circle);break;
        case 16: fusedDilateSubPrevOneKernel<16><<<g,b,0,st>>>(prevOneBuf,sub,dst,dstDil,w,h,circle);break;
        case 17: fusedDilateSubPrevOneKernel<17><<<g,b,0,st>>>(prevOneBuf,sub,dst,dstDil,w,h,circle);break;
        case 18: fusedDilateSubPrevOneKernel<18><<<g,b,0,st>>>(prevOneBuf,sub,dst,dstDil,w,h,circle);break;
        case 19: fusedDilateSubPrevOneKernel<19><<<g,b,0,st>>>(prevOneBuf,sub,dst,dstDil,w,h,circle);break;
        case 20: fusedDilateSubPrevOneKernel<20><<<g,b,0,st>>>(prevOneBuf,sub,dst,dstDil,w,h,circle);break;
        case 21: fusedDilateSubPrevOneKernel<21><<<g,b,0,st>>>(prevOneBuf,sub,dst,dstDil,w,h,circle);break;
        case 22: fusedDilateSubPrevOneKernel<22><<<g,b,0,st>>>(prevOneBuf,sub,dst,dstDil,w,h,circle);break;
        case 23: fusedDilateSubPrevOneKernel<23><<<g,b,0,st>>>(prevOneBuf,sub,dst,dstDil,w,h,circle);break;
        case 24: fusedDilateSubPrevOneKernel<24><<<g,b,0,st>>>(prevOneBuf,sub,dst,dstDil,w,h,circle);break;
        case 25: fusedDilateSubPrevOneKernel<25><<<g,b,0,st>>>(prevOneBuf,sub,dst,dstDil,w,h,circle);break;
        case 26: fusedDilateSubPrevOneKernel<26><<<g,b,0,st>>>(prevOneBuf,sub,dst,dstDil,w,h,circle);break;
        case 27: fusedDilateSubPrevOneKernel<27><<<g,b,0,st>>>(prevOneBuf,sub,dst,dstDil,w,h,circle);break;
        case 28: fusedDilateSubPrevOneKernel<28><<<g,b,0,st>>>(prevOneBuf,sub,dst,dstDil,w,h,circle);break;
        case 29: fusedDilateSubPrevOneKernel<29><<<g,b,0,st>>>(prevOneBuf,sub,dst,dstDil,w,h,circle);break;
        case 30: fusedDilateSubPrevOneKernel<30><<<g,b,0,st>>>(prevOneBuf,sub,dst,dstDil,w,h,circle);break;
        case 31: fusedDilateSubPrevOneKernel<31><<<g,b,0,st>>>(prevOneBuf,sub,dst,dstDil,w,h,circle);break;
        default: break;
    }
}
static const uint8_t* gp(const cv::cuda::GpuMat& m) { return m.ptr<uint8_t>(); }
static uint8_t* gp(cv::cuda::GpuMat& m) { return m.ptr<uint8_t>(); }

// ============================================================
// Impl
// ============================================================
LaserMarkingSeparationCUDA::Impl::Impl(const LaserMarkingSeparationParams& p) : params_(p) {
    params_.validate();
    if (cv::cuda::getCudaEnabledDeviceCount() <= 0)
        throw std::runtime_error("No CUDA GPU");
    createEvents();
}
LaserMarkingSeparationCUDA::Impl::~Impl() {
    destroyEvents();
    if (h_grayPin_) { cudaFreeHost(h_grayPin_); h_grayPin_ = nullptr; grayPinCap_ = 0; }
    if (graphExec_) { cudaGraphExecDestroy(graphExec_); graphExec_ = nullptr; }
    if (graph_) { cudaGraphDestroy(graph_); graph_ = nullptr; }
        if (graphExecDev_) { cudaGraphExecDestroy(graphExecDev_); graphExecDev_ = nullptr; }
        if (graphDev_) { cudaGraphDestroy(graphDev_); graphDev_ = nullptr; }
}

void LaserMarkingSeparationCUDA::Impl::createEvents() {
    if (events_created_) return;
    cudaEventCreate(&event_start_);
    cudaEventCreate(&event_upload_done_);
    cudaEventCreate(&event_step1_done_);
    cudaEventCreate(&event_step2_done_);
    cudaEventCreate(&event_step3_done_);
    cudaEventCreate(&event_step4_done_);
    cudaEventCreate(&event_step5_done_);
    cudaEventCreate(&ev_dbgA_);
    cudaEventCreate(&ev_dbgB_);
    events_created_ = true;
}
void LaserMarkingSeparationCUDA::Impl::destroyEvents() {
    if (!events_created_) return;
    cudaEventDestroy(event_start_);   cudaEventDestroy(event_upload_done_);
    cudaEventDestroy(event_step1_done_); cudaEventDestroy(event_step2_done_);
    cudaEventDestroy(event_step3_done_); cudaEventDestroy(event_step4_done_);
    cudaEventDestroy(event_step5_done_);
    cudaEventDestroy(ev_dbgA_); cudaEventDestroy(ev_dbgB_);
    events_created_ = false;
}

// ============================================================
// executePipeline
// ============================================================
void LaserMarkingSeparationCUDA::Impl::executePipeline(
    cv::cuda::Stream& stream, MaskSeparationTimings& timings, bool recordEvents)
{
    auto cs = cv::cuda::StreamAccessor::getStream(stream);
    int w = d_inputBuffer.cols, h = d_inputBuffer.rows;
    const bool circle = (params_.kernelShape == 1);   // 0=閺傜懓鑸?1=閸﹀棗鑸扮壕浣烘磸
    // 鏉╂ê甯弽?= 閺夆偓閺?2閿涙岸妲婚獮绋跨摠閼?1px 鏉堝湱绱悳顖涚濠曞繗绻橀崙蹇旂《鏉堟挸鍤敍?2 閹媽藟楠?閳?閻?-2px
    // 閳?閳?闂冨牆鈧壈顕㈡稊?= 閻喎鐤勭€硅棄瀹抽崺?
    const int laserRestore  = params_.laserKillSize + 2;
    const int markerRestore = params_.markerKillSize + 2;

    // 濞夘煉绱癳vent_start_ 閻㈣精鐨熼悽銊︽煙閿涘澃eparate/warmup閿涘婀稉濠佺炊閸撳秷顔囪ぐ鏇椻偓鏂衡偓鏃€顒濇径鍕瑝闁插秴缍嶉敍?
    // 閸氾箑鍨?upload_ms 閻ㄥ嫬灏梻纾嬫崳閻愯娅勬禍搴ｇ矒閻愮櫢绱橦istorical bug閿涙碍娴橀崷銊︻劃闁插秴缍嶉懛?upload_ms 閹帟绀嬮敍?

    // 閳?Fused Gaussian + Threshold
    fusedGaussianThresholdKernel<<<gridSz(w,h),dim3(BW,BH),0,cs>>>(
        gp(d_inputBuffer), gp(d_binary), w, h, params_.threshold);
    if (recordEvents) cudaEventRecord(event_step1_done_, cs);

    // 閳?閸樿鐨崳顏嗗仯閼垫劘娈?閳?閸忋劑鍣虹紒鎾寸€崶?X閿涘牆鎮囩紒鎾寸€?-1px/鏉堢櫢绱?
    // ② 去小噪点腐蚀 → 全量结构图 X（各结构 -1px/边）——【块内融合核】
    launchBlockMorphDispatch(gp(d_binary), gp(d_structures), w, h,
                             params_.noiseErodeSize, circle, true, cs);
    if (recordEvents) cudaEventRecord(event_step2_done_, cs);

    // ③ 杀激光：腐蚀 → 融合[膨胀还原+减法] → laserRaw + survivor
    //    laserRaw = X − 还原幸存者（激光线 4-7px 死透落入）
    //    survivor = 标记点+中/大噪点（还原至真实宽度域）
    launchBlockMorphDispatch(gp(d_structures), gp(d_temp), w, h,
                             params_.laserKillSize, circle, true, cs);
    if (recordEvents) cudaEventRecord(ev_dbgA_, cs);
    launchBlockDilateSubDispatch(gp(d_temp), gp(d_structures), gp(d_laserRaw),
                                 gp(d_survivor), w, h, laserRestore, circle, cs);
    if (recordEvents) cudaEventRecord(event_step3_done_, cs);

    // 閳?閺夆偓閺嶅洩顔囬悙鐧哥窗閼垫劘娈?閳?閾诲秴鎮嶽閼躲劏鍎夋潻妯哄斧+閸戝繑纭禲 閳?markingRaw閿涘牆銇囬崳顏嗗仯閳妿_discard 瀵喛绱?
    //    markingRaw = survivor 閳?鏉╂ê甯獮绋跨摠閼板拑绱欓弽鍥唶閻愯顒撮柅蹇氭儰閸忋儻绱辨稉顓炴珨閻愮懓鎮撻崗銉礉
    //    闂堢姳绗呭〒?ccl 闂堛垻袧闂傛瓕绻冨銈忕礆
    // ④ 杀标记点：腐蚀 → 融合[膨胀还原+减法] → markingRaw（大噪点→d_discard 弃）
    //    markingRaw = survivor − 还原幸存者（标记点死透落入；中噪点同入，
    //    靠下游 ccl 面积闸过滤）——【块内融合核】
    launchBlockMorphDispatch(gp(d_survivor), gp(d_temp), w, h,
                             params_.markerKillSize, circle, true, cs);
    if (recordEvents) cudaEventRecord(ev_dbgB_, cs);
    launchBlockDilateSubDispatch(gp(d_temp), gp(d_survivor), gp(d_markingRaw),
                                 nullptr, w, h, markerRestore, circle, cs);  // discard 无消费方——停写省一遍全局写
    if (recordEvents) cudaEventRecord(event_step4_done_, cs);

    // ⑤ 边缘保留膨胀（供下游亚像素边缘提取；净外扩 ~3px/边）——【块内融合核】
    launchBlockMorphDispatch(gp(d_laserRaw), gp(d_laserFinal), w, h,
                             params_.laserEdgeDilateSize, circle, false, cs);
    launchBlockMorphDispatch(gp(d_markingRaw), gp(d_markingFinal), w, h,
                             params_.markingEdgeDilateSize, circle, false, cs);
    if (recordEvents) cudaEventRecord(event_step5_done_, cs);
    // 【速度优化】内部计时同步改 env 门控——计时跑（JMW_SEPTIME）才同步；
    // 常规回放免同步（harness 事件计时制下链路直通）
    if (getenv("JMW_SEPTIME")) {
        cudaEventSynchronize(event_step5_done_);
        float ms;
        float dbg3a = 0, dbg3b = 0, dbg4a = 0, dbg4b = 0;
        cudaEventElapsedTime(&ms, event_start_, event_upload_done_);      timings.upload_ms = ms;
        cudaEventElapsedTime(&ms, event_upload_done_, event_step1_done_); timings.step1_gaussian_threshold_ms = ms;
        cudaEventElapsedTime(&ms, event_step1_done_, event_step2_done_);  timings.step2_noise_erode_ms = ms;
        cudaEventElapsedTime(&ms, event_step2_done_, ev_dbgA_);           dbg3a = ms;
        cudaEventElapsedTime(&ms, ev_dbgA_, event_step3_done_);           dbg3b = ms;
        cudaEventElapsedTime(&ms, event_step2_done_, event_step3_done_);  timings.step3_laser_extract_ms = ms;
        cudaEventElapsedTime(&ms, event_step3_done_, ev_dbgB_);           dbg4a = ms;
        cudaEventElapsedTime(&ms, ev_dbgB_, event_step4_done_);           dbg4b = ms;
        cudaEventElapsedTime(&ms, event_step3_done_, event_step4_done_);  timings.step4_marking_extract_ms = ms;
        cudaEventElapsedTime(&ms, event_step4_done_, event_step5_done_);  timings.step5_edge_dilate_ms = ms;
        cudaEventElapsedTime(&ms, event_start_, event_step5_done_);       timings.total_pipeline_ms = ms;
        fprintf(stderr, "[sep-t] up=%.2f s1=%.2f s2=%.2f s3=%.2f(%.2f+%.2f) s4=%.2f(%.2f+%.2f) s5=%.2f tot=%.2f\n",
                timings.upload_ms, timings.step1_gaussian_threshold_ms,
                timings.step2_noise_erode_ms, timings.step3_laser_extract_ms,
                dbg3a, dbg3b, timings.step4_marking_extract_ms, dbg4a, dbg4b,
                timings.step5_edge_dilate_ms,
                timings.total_pipeline_ms);
    }
}

void LaserMarkingSeparationCUDA::Impl::warmup(int rows, int cols) {
    auto a = [&](cv::cuda::GpuMat& b){ cv::cuda::createContinuous(rows,cols,CV_8UC1,b); };
    a(d_inputBuffer); a(d_binary); a(d_structures); a(d_temp);
    a(d_survivor); a(d_laserRaw); a(d_markingRaw); a(d_discard);
    a(d_laserFinal); a(d_markingFinal);
    warmup_rows_=rows; warmup_cols_=cols;
    // 【P1 修复】warmup 重分配 GpuMat → 旧图持有悬垂指针，必须失效
    if (graphExec_) { cudaGraphExecDestroy(graphExec_); graphExec_ = nullptr; }
    if (graph_) { cudaGraphDestroy(graph_); graph_ = nullptr; }
        if (graphExecDev_) { cudaGraphExecDestroy(graphExecDev_); graphExecDev_ = nullptr; }
        if (graphDev_) { cudaGraphDestroy(graphDev_); graphDev_ = nullptr; }

    cv::Mat z = cv::Mat::zeros(rows,cols,CV_8UC1);
    d_inputBuffer.upload(z);
    cv::cuda::Stream s;
    MaskSeparationTimings t;
    cudaEventRecord(event_start_, cv::cuda::StreamAccessor::getStream(s));
    d_inputBuffer.upload(z, s);
    executePipeline(s, t);
#ifndef NDEBUG
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess)
        throw std::runtime_error(std::string("warmup failed: ")+cudaGetErrorString(err));
#endif
    warmed_up_ = true;
}

void LaserMarkingSeparationCUDA::Impl::setParams(const LaserMarkingSeparationParams& p) {
#ifndef NDEBUG
    assert(!inProcess_.load());
#endif
    params_ = p; params_.validate(); warmed_up_ = false;
    // 参数变化 → 捕获的图失效，下次 separate 重新捕获
    if (graphExec_) { cudaGraphExecDestroy(graphExec_); graphExec_ = nullptr; }
    if (graph_) { cudaGraphDestroy(graph_); graph_ = nullptr; }
        if (graphExecDev_) { cudaGraphExecDestroy(graphExecDev_); graphExecDev_ = nullptr; }
        if (graphDev_) { cudaGraphDestroy(graphDev_); graphDev_ = nullptr; }
}

LaserMarkingSeparationResult LaserMarkingSeparationCUDA::Impl::separate(
    const cv::Mat& grayImage, cv::cuda::Stream& stream)
{
#ifndef NDEBUG
    assert(!inProcess_.load());
    struct SF { std::atomic<bool>*f; SF(std::atomic<bool>*f_):f(f_){} ~SF(){f->store(false);} };
    SF guard(&inProcess_); inProcess_.store(true);
#endif

    LaserMarkingSeparationResult result;
    try {
        int r=grayImage.rows, c=grayImage.cols;
        if (!warmed_up_||warmup_rows_!=r||warmup_cols_!=c) {
            auto a = [&](cv::cuda::GpuMat& b){ cv::cuda::createContinuous(r,c,CV_8UC1,b); };
            a(d_inputBuffer); a(d_binary); a(d_structures); a(d_temp);
            a(d_survivor); a(d_laserRaw); a(d_markingRaw); a(d_discard);
            a(d_laserFinal); a(d_markingFinal);
            warmup_rows_=r; warmup_cols_=c; warmed_up_=true;
            // 尺寸变化 → 捕获的图持有旧指针/旧尺寸，必须失效重捕获
            if (graphExec_) { cudaGraphExecDestroy(graphExec_); graphExec_ = nullptr; }
            if (graph_) { cudaGraphDestroy(graph_); graph_ = nullptr; }
        if (graphExecDev_) { cudaGraphExecDestroy(graphExecDev_); graphExecDev_ = nullptr; }
        if (graphDev_) { cudaGraphDestroy(graphDev_); graphDev_ = nullptr; }
        }
        auto cs = cv::cuda::StreamAccessor::getStream(stream);
        // 【速度优化·二轮】pinned 暂存上传（免 pageable staging/隐式同步）
        {
            const size_t n = static_cast<size_t>(r) * c;
            if (grayPinCap_ < n) {
                if (h_grayPin_) cudaFreeHost(h_grayPin_);
                cudaMallocHost(&h_grayPin_, n);
                grayPinCap_ = n;
                if (graphExec_) { cudaGraphExecDestroy(graphExec_); graphExec_ = nullptr; }
                if (graph_) { cudaGraphDestroy(graph_); graph_ = nullptr; }
        if (graphExecDev_) { cudaGraphExecDestroy(graphExecDev_); graphExecDev_ = nullptr; }
        if (graphDev_) { cudaGraphDestroy(graphDev_); graphDev_ = nullptr; }
            }
            memcpy(h_grayPin_, grayImage.data, n);
        }
        const bool diagMode = getenv("JMW_SEPTIME") != nullptr;
        if (!diagMode && graphExec_) {
            // 稳态：整管线一次提交（图内含 H2D 上传——pinned 源地址恒定）
            cudaGraphLaunch(graphExec_, cs);
            goto pipelineDone;   // 【P0 修复】防 fall-through 到 legacy 双执行
        } else if (!diagMode) {
            // 首帧：捕获完整序列（上传＋全部分步核），实例化后立即执行
            cudaError_t ce = cudaStreamBeginCapture(cs, cudaStreamCaptureModeGlobal);
            if (ce != cudaSuccess) goto legacyPath;
            {
                const cv::Mat wrap(r, c, CV_8UC1, h_grayPin_);
                d_inputBuffer.upload(wrap, stream);
            }
            executePipeline(stream, result.timings, /*recordEvents=*/false);
            if (cudaStreamEndCapture(cs, &graph_) != cudaSuccess) { graph_ = nullptr; goto legacyPath; }
            if (cudaGraphInstantiate(&graphExec_, graph_, nullptr, nullptr, 0) != cudaSuccess || !graphExec_) {
                graphExec_ = nullptr; goto legacyPath;
            }
            if (cudaGraphLaunch(graphExec_, cs) != cudaSuccess) goto legacyPath;
            goto pipelineDone;
        }
        { legacyPath:;
            cudaEventRecord(event_start_, cs);
            {
                const cv::Mat wrap(r, c, CV_8UC1, h_grayPin_);
                d_inputBuffer.upload(wrap, stream);
            }
            cudaEventRecord(event_upload_done_, cs);
            executePipeline(stream, result.timings, /*recordEvents=*/true);
        }
        pipelineDone:;

        result.success = true;
        result.message = "Separation successful";
        result.qualityFlag = calib::QualityFlag::Normal;
        // 【速度优化·二轮】结果浅引用替 clone（省 4 次 2.36MB 显存拷贝）；
        // 有效性契约不变：至本实例下一次 Execute 前（harness 每眼一实例、每帧一次）
        result.d_laserMask = std::make_shared<cv::cuda::GpuMat>(d_laserFinal);
        result.d_markingPointMask = std::make_shared<cv::cuda::GpuMat>(d_markingFinal);
        result.d_combinedMask = std::make_shared<cv::cuda::GpuMat>(d_structures);
        // ④ 回显掩膜（给 harness S0 双链矫正系一致性检查用）
        result.d_step1_binary = std::make_shared<cv::cuda::GpuMat>(d_binary);
        if (params_.dumpAllSteps) {          // 携带分步中间掩膜（额外显存拷贝，诊断期才付）
            result.d_structuresMask = std::make_shared<cv::cuda::GpuMat>(d_structures);
            result.d_laserRaw       = std::make_shared<cv::cuda::GpuMat>(d_laserRaw);
            result.d_survivorMask   = std::make_shared<cv::cuda::GpuMat>(d_survivor);
            result.d_markingRaw     = std::make_shared<cv::cuda::GpuMat>(d_markingRaw);
        }

        CALIB_LOG_DEBUG("Pipeline timings (ms): upload={:.3f} step1={:.3f} "
            "step2={:.3f} step3={:.3f} step4={:.3f} step5={:.3f} total={:.3f}",
            result.timings.upload_ms, result.timings.step1_gaussian_threshold_ms,
            result.timings.step2_noise_erode_ms, result.timings.step3_laser_extract_ms,
            result.timings.step4_marking_extract_ms, result.timings.step5_edge_dilate_ms,
            result.timings.total_pipeline_ms);
    } catch (const cv::Exception& e) {
        result.success=false; result.message=std::string("OpenCV: ")+e.what();
        result.qualityFlag=calib::QualityFlag::Degraded;
    } catch (const std::exception& e) {
        result.success=false; result.message=std::string("Err: ")+e.what();
        result.qualityFlag=calib::QualityFlag::Degraded;
    }
    return result;
}

// 【速度优化】GpuMat 直输：输入已在显存（steger 同源帧）——免每帧二次 H2D 上传。
// 图＝纯管线（无上传节点）；D2D 入 d_inputBuffer 图外发射（源指针可变）。
// 逐内核序列/参数与 separate() 完全一致 → 结果逐位一致。
LaserMarkingSeparationResult LaserMarkingSeparationCUDA::Impl::separateDev(
    const cv::cuda::GpuMat& d_grayImage, cv::cuda::Stream& stream)
{
#ifndef NDEBUG
    assert(!inProcess_.load());
    struct SF { std::atomic<bool>*f; SF(std::atomic<bool>*f_):f(f_){} ~SF(){f->store(false);} };
    SF guard(&inProcess_); inProcess_.store(true);
#endif

    LaserMarkingSeparationResult result;
    try {
        if (d_grayImage.empty()) throw std::runtime_error("device input is empty");
        if (d_grayImage.type() != CV_8UC1)
            throw std::runtime_error("device input must be CV_8UC1");
        int r = d_grayImage.rows, c = d_grayImage.cols;
        if (!warmed_up_ || warmup_rows_ != r || warmup_cols_ != c) {
            auto a = [&](cv::cuda::GpuMat& b){ cv::cuda::createContinuous(r,c,CV_8UC1,b); };
            a(d_inputBuffer); a(d_binary); a(d_structures); a(d_temp);
            a(d_survivor); a(d_laserRaw); a(d_markingRaw); a(d_discard);
            a(d_laserFinal); a(d_markingFinal);
            warmup_rows_ = r; warmup_cols_ = c; warmed_up_ = true;
            if (graphExec_) { cudaGraphExecDestroy(graphExec_); graphExec_ = nullptr; }
            if (graph_) { cudaGraphDestroy(graph_); graph_ = nullptr; }
            if (graphExecDev_) { cudaGraphExecDestroy(graphExecDev_); graphExecDev_ = nullptr; }
            if (graphDev_) { cudaGraphDestroy(graphDev_); graphDev_ = nullptr; }
        }
        auto cs = cv::cuda::StreamAccessor::getStream(stream);
        const size_t n = static_cast<size_t>(r) * c;

        const bool diagMode = getenv("JMW_SEPTIME") != nullptr;
        if (!diagMode && graphExecDev_) {
            // 稳态：D2D 入驻 + 一次 graphLaunch。
            // ⚠ 必须用 copyTo：源 GpuMat 可能带行距（pitch），按字节裸拷会压扁布局
            d_grayImage.copyTo(d_inputBuffer, stream);
            cudaGraphLaunch(graphExecDev_, cs);
        } else if (!diagMode) {
            // 首次：捕获纯管线（无上传节点），实例化后立即执行；
            // 捕获/实例化失败 → 直接发射兜底（捕获模式不执行，必须重跑管线）
            d_grayImage.copyTo(d_inputBuffer, stream);
            // 【WDDM 边界修复】同流前序图仍在飞时紧接 BeginCapture 会触发驱动
            // 崩溃——捕获仅每实例一次，排空等待代价可忽略
            stream.waitForCompletion();
            cudaError_t ce = cudaStreamBeginCapture(cs, cudaStreamCaptureModeGlobal);
            if (ce == cudaSuccess) {
                executePipeline(stream, result.timings, /*recordEvents=*/false);
                if (cudaStreamEndCapture(cs, &graphDev_) != cudaSuccess) graphDev_ = nullptr;
            }
            if (graphDev_ &&
                cudaGraphInstantiate(&graphExecDev_, graphDev_, nullptr, nullptr, 0)
                    == cudaSuccess && graphExecDev_) {
                cudaGraphLaunch(graphExecDev_, cs);
            } else {
                executePipeline(stream, result.timings, /*recordEvents=*/false);
            }
        } else {
            // 诊断路径（与 host 版一致：逐核事件＋同步计时）
            cudaEventRecord(event_start_, cs);
            d_grayImage.copyTo(d_inputBuffer, stream);   // ⚠ copyTo 处理行距
            cudaEventRecord(event_upload_done_, cs);
            executePipeline(stream, result.timings, /*recordEvents=*/true);
        }

        result.success = true;
        result.message = "Separation successful";
        result.qualityFlag = calib::QualityFlag::Normal;
        result.d_laserMask = std::make_shared<cv::cuda::GpuMat>(d_laserFinal);
        result.d_markingPointMask = std::make_shared<cv::cuda::GpuMat>(d_markingFinal);
        result.d_combinedMask = std::make_shared<cv::cuda::GpuMat>(d_structures);
        result.d_step1_binary = std::make_shared<cv::cuda::GpuMat>(d_binary);
        if (params_.dumpAllSteps) {
            result.d_structuresMask = std::make_shared<cv::cuda::GpuMat>(d_structures);
            result.d_laserRaw       = std::make_shared<cv::cuda::GpuMat>(d_laserRaw);
            result.d_survivorMask   = std::make_shared<cv::cuda::GpuMat>(d_survivor);
            result.d_markingRaw     = std::make_shared<cv::cuda::GpuMat>(d_markingRaw);
        }
    } catch (const cv::Exception& e) {
        result.success=false; result.message=std::string("OpenCV: ")+e.what();
        result.qualityFlag=calib::QualityFlag::Degraded;
    } catch (const std::exception& e) {
        result.success=false; result.message=std::string("Err: ")+e.what();
        result.qualityFlag=calib::QualityFlag::Degraded;
    }
    return result;
}
