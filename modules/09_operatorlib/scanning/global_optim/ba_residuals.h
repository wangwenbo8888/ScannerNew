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

} // namespace calib
