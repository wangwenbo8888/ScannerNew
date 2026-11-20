#pragma once
#include <ceres/ceres.h>
#include <Eigen/Core>
#include <Eigen/Geometry>

namespace calib {

// so3 李代数 log: 从旋转矩阵 R 取旋转向量。模板化(支持 ceres Jet)。
//
// 实现: 小角度 vee 近似  log(R) ≈ vee((R - Rᵀ)/2) = 0.5 * [R21-R12, R02-R20, R10-R01] = sin(θ)*axis。
// 当 θ 较小时 sin(θ)≈θ, 上式即 so3 log 的精确值; θ 较大时为近似(只取一阶项)。
//
// ========================= 有效窗口(硬限制, 使用方必读) =========================
//   该 vee 形式返回 sin(θ)·axis, 仅在 |θ| 较小时接近真实 so3 log。它有病态的稳定点:
//     - θ=π 时 sin(π)=0 → 残差读数为零(180° 漂移完全不可见!);
//     - θ=π/2 处为鞍点(sin 达峰后回落), 梯度方向错误。
//   实践有效窗口: |θ| ≲ ~0.3 rad(~17°)。超出该窗口, PGO 可能静默欠矫正大旋转漂移。
//   之所以选该形式而非含 acos+钳位的"精确" Rodrigues:
//     (1) 对 ceres::Jet 全程光滑(纯多项式), 适合 AutoDiff;
//     (2) PGO 设计为 warm-start: 初值是真实位姿的小扰动, 旋转残差本应很小, 近似足够;
//         优化过程中残差持续收缩, 近似误差进一步减小。
//   风险: 若某闭环初始位姿的旋转漂移已很大(长闭环累积), vee 形式无法表达, PGO 会欠矫正。
//   缓解: 调用方(global_ba_cpu.cpp PGO 块)在求解前对初始位姿算各边旋转残差模, 超窗口时
//         spdlog::warn 告警(best-effort: PGO+GBA 不中止, 精密由 GBA plain-LM 兜底)。
//   切勿将该残差用于 |θ| 可能接近 π 的场景。
// ==============================================================================
template <typename T>
inline Eigen::Matrix<T, 3, 1> so3Log(const Eigen::Matrix<T, 3, 3>& R) {
    Eigen::Matrix<T, 3, 1> v;
    v(0) = R(2, 1) - R(1, 2);
    v(1) = R(0, 2) - R(2, 0);
    v(2) = R(1, 0) - R(0, 1);
    return v * T(0.5);
}

// 相对位姿残差(6 维)。autodiff-friendly, 不依赖 Sophus。
//
// 约定:
//   测量值 T_ik=(R_ik, t_ik) 是 "frame-k-local → frame-i-local" 的相对位姿(由 Kabsch 求得,
//   即满足 z_i ≈ R_ik z_k + t_ik, 其中 z_i/z_k 为同一 landmark 在 i/k 帧的本机系坐标)。
//   估计位姿 T_i=(R_i,t_i)、T_k=(R_k,t_k) 为 scanner-local → global。
//   估计的 "k-local → i-local" 相对位姿 = T_i⁻¹ T_k = (R_iᵀ R_k, R_iᵀ(t_k − t_i))。
//
// 残差:
//   旋转: r_rot = so3Log( R_ikᵀ · (R_iᵀ R_k) )         (零 ⇔ R_iᵀ R_k = R_ik)
//   平移: r_tr  = R_iᵀ(t_k − t_i) − t_ik               (零 ⇔ 估计相对平移 = 测量)
//
// 块布局: qi4[frame i 四元数 w,x,y,z], ti3[frame i 平移], qk4, tk3。
struct RelativePoseResidual {
    Eigen::Matrix3d R_ik;
    Eigen::Vector3d t_ik;
    double rotWeight;
    double transWeight;

    RelativePoseResidual(const Eigen::Matrix3d& Rik, const Eigen::Vector3d& tik,
                         double rotW, double transW)
        : R_ik(Rik), t_ik(tik), rotWeight(rotW), transWeight(transW) {}

    template <typename T>
    bool operator()(const T* const qi4,   // frame i 四元数 [w,x,y,z]
                    const T* const ti3,   // frame i 平移
                    const T* const qk4,   // frame k 四元数 [w,x,y,z]
                    const T* const tk3,   // frame k 平移
                    T* residual) const {
        Eigen::Quaternion<T> Qi(qi4[0], qi4[1], qi4[2], qi4[3]);
        Eigen::Quaternion<T> Qk(qk4[0], qk4[1], qk4[2], qk4[3]);
        Eigen::Matrix<T, 3, 1> ti(ti3[0], ti3[1], ti3[2]);
        Eigen::Matrix<T, 3, 1> tk(tk3[0], tk3[1], tk3[2]);

        Eigen::Matrix<T, 3, 3> Ri = Qi.toRotationMatrix();
        Eigen::Matrix<T, 3, 3> Rk = Qk.toRotationMatrix();
        Eigen::Matrix<T, 3, 3> R_ik_t = R_ik.template cast<T>().transpose();
        Eigen::Matrix<T, 3, 3> R_rel_est = Ri.transpose() * Rk;  // R_iᵀ R_k
        // so3 log of R_ikᵀ · R_rel_est
        Eigen::Matrix<T, 3, 3> Rerr = R_ik_t * R_rel_est;
        Eigen::Matrix<T, 3, 1> rrot = so3Log(Rerr);
        // 平移残差: R_iᵀ(t_k − t_i) − t_ik
        Eigen::Matrix<T, 3, 1> rtr = Ri.transpose() * (tk - ti) - t_ik.template cast<T>();

        residual[0] = rrot(0) * T(rotWeight);
        residual[1] = rrot(1) * T(rotWeight);
        residual[2] = rrot(2) * T(rotWeight);
        residual[3] = rtr(0) * T(transWeight);
        residual[4] = rtr(1) * T(transWeight);
        residual[5] = rtr(2) * T(transWeight);
        return true;
    }
};

} // namespace calib
