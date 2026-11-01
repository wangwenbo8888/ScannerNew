#pragma once
#include <ceres/ceres.h>
#include <Eigen/Core>
#include <Eigen/Geometry>

namespace calib {

// 残差 r = R(q)ᵀ (X - t) - z, 旋转由四元数 q(w,x,y,z) 构造,内部归一化
// 信息量加权: 残差乘 1/sigma
struct PointPairResidual {
    Eigen::Vector3d z;      // 本机系观测
    double inv_sigma;       // 1/sigma

    PointPairResidual(const Eigen::Vector3d& obs, double sigma)
        : z(obs), inv_sigma(1.0 / sigma) {}

    template <typename T>
    bool operator()(const T* const q4,   // 四元数 [w,x,y,z], 4
                    const T* const t3,   // 平移, 3
                    const T* const X3,   // 全局点, 3
                    T* residual) const {
        // 归一化四元数(防数值漂移)
        T qn[4] = {q4[0], q4[1], q4[2], q4[3]};
        T n = ceres::sqrt(qn[0]*qn[0]+qn[1]*qn[1]+qn[2]*qn[2]+qn[3]*qn[3]) + T(1e-30);
        for (int i = 0; i < 4; ++i) qn[i] = qn[i] / n;

        Eigen::Quaternion<T> Q(qn[0], qn[1], qn[2], qn[3]);
        Eigen::Matrix<T,3,1> t(t3[0], t3[1], t3[2]);
        Eigen::Matrix<T,3,1> X(X3[0], X3[1], X3[2]);
        // Rᵀ(X - t) = Q.conjugate() * (X - t)
        Eigen::Matrix<T,3,1> pred = Q.conjugate() * (X - t);
        residual[0] = (pred(0) - T(z(0))) * T(inv_sigma);
        residual[1] = (pred(1) - T(z(1))) * T(inv_sigma);
        residual[2] = (pred(2) - T(z(2))) * T(inv_sigma);
        return true;
    }
};

// P4-T24 高精度已有點软先验残差（客户端扫描流水线.md §5.3）: r[i] = (X[0][i] − X_e[i]) / sigma
// 动态参数块 functor 形态（T const* const*）, 配合
//   ceres::DynamicAutoDiffCostFunction<MarkerPriorCost, 3> + AddParameterBlock(3) 使用;
// 不加鲁棒损失（先验本身可信, 不应被 Tukey 压制）。
// 升级路径（仅注释未实现）: 某子集需绝对固定时, 对相应参数块改用
//   problem.SetParameterBlockConstant(X) 替代本软先验。
struct MarkerPriorCost {
    double X_e[3];   // 先验位置（与 X 同坐标系）
    double sigma;    // 先验 σ（极小=高权重）

    template <typename T>
    bool operator()(T const* const* X, T* r) const {
        for (int i = 0; i < 3; ++i) r[i] = (X[0][i] - T(X_e[i])) / T(sigma);
        return true;
    }
};

} // namespace calib
