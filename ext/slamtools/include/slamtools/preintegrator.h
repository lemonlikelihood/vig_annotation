#pragma once

#include <slamtools/common.h>
#include <slamtools/state.h>

struct PreIntegrator {
    struct Delta {
        double t;
        Eigen::Quaterniond q;
        Eigen::Vector3d p;
        Eigen::Vector3d v;
        Eigen::Matrix<double, 15, 15> cov; // ordered in q, p, v, bg, ba
        Eigen::Matrix<double, 15, 15> sqrt_inv_cov;
    };

    struct Jacobian {
        Eigen::Matrix3d dq_dbg;
        Eigen::Matrix3d dp_dbg;
        Eigen::Matrix3d dp_dba;
        Eigen::Matrix3d dv_dbg;
        Eigen::Matrix3d dv_dba;
    };

    void reset();
    void increment(double dt, const IMUData &data, const Eigen::Vector3d &bg, const Eigen::Vector3d &ba, bool compute_jacobian, bool compute_covariance);
    bool integrate(double t, const Eigen::Vector3d &bg, const Eigen::Vector3d &ba, bool compute_jacobian, bool compute_covariance);
    void compute_sqrt_inv_cov();

    Eigen::Matrix3d cov_w; // continuous noise covariance                     // 连续白噪声协方差，离散值 = 连续值 / delta t
    Eigen::Matrix3d cov_a;
    Eigen::Matrix3d cov_bg; // continuous random walk noise covariance
    Eigen::Matrix3d cov_ba;

    Delta delta;                  // 在进行increment和integrate后当前帧会获得delta和jacobian信息
    Jacobian jacobian;

    std::vector<IMUData> data;   // 相邻两个图片帧之间的IMU数据
};
