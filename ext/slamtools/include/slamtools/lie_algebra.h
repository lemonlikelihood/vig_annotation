#pragma once

#include <slamtools/common.h>

// 将向量转换为反对称矩阵
inline Eigen::Matrix3d hat(const Eigen::Vector3d &w) {
    return (Eigen::Matrix3d() << 0, -w.z(), w.y(),
            w.z(), 0, -w.x(),
            -w.y(), w.x(), 0)
        .finished();
}

// 轴角变旋转
inline Eigen::Quaterniond expmap(const Eigen::Vector3d &w) {
    Eigen::AngleAxisd aa(w.norm(), w.stableNormalized());
    Eigen::Quaterniond q;
    q = aa;
    return q;
}

// 旋转变轴角
inline Eigen::Vector3d logmap(const Eigen::Quaterniond &q) {
    Eigen::AngleAxisd aa(q);
    return aa.angle() * aa.axis();
}

Eigen::Matrix3d right_jacobian(const Eigen::Vector3d &w);
Eigen::Matrix<double, 3, 2> s2_tangential_basis(const Eigen::Vector3d &x);
