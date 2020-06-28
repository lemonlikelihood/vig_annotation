#include <slamtools/preintegrator.h>
#include <slamtools/lie_algebra.h>

using namespace Eigen;

void PreIntegrator::reset() {                                                   // 全部重置为0
    delta.t = 0;
    delta.q.setIdentity();
    delta.p.setZero();
    delta.v.setZero();
    delta.cov.setZero();
    delta.sqrt_inv_cov.setZero();

    jacobian.dq_dbg.setZero();
    jacobian.dp_dbg.setZero();
    jacobian.dp_dba.setZero();
    jacobian.dv_dbg.setZero();
    jacobian.dv_dba.setZero();
}

void PreIntegrator::increment(double dt, const IMUData &data, const Vector3d &bg, const Vector3d &ba, bool compute_jacobian, bool compute_covariance) {
    assert(("dt must > 0") && (dt >= 0));

    Vector3d w = data.w - bg;
    Vector3d a = data.a - ba;

    if (compute_covariance) {
        Matrix<double, 9, 9> A;
        A.setIdentity();
        A.block<3, 3>(ES_Q, ES_Q) = expmap(w * dt).conjugate().matrix();                       // p[0,0]
        A.block<3, 3>(ES_V, ES_Q) = -dt * delta.q.matrix() * hat(a);
        A.block<3, 3>(ES_P, ES_Q) = -0.5 * dt * dt * delta.q.matrix() * hat(a);
        A.block<3, 3>(ES_P, ES_V) = dt * Matrix3d::Identity();

        Matrix<double, 9, 6> B;
        B.setZero();
        B.block<3, 3>(ES_Q, ES_BG - ES_BG) = dt * right_jacobian(w * dt);
        B.block<3, 3>(ES_V, ES_BA - ES_BG) = dt * delta.q.matrix();
        B.block<3, 3>(ES_P, ES_BA - ES_BG) = 0.5 * dt * dt * delta.q.matrix();

        Matrix<double, 6, 6> white_noise_cov;
        double inv_dt = 1.0 / std::max(dt, 1.0e-7);
        white_noise_cov.setZero();
        white_noise_cov.block<3, 3>(ES_BG - ES_BG, ES_BG - ES_BG) = cov_w * inv_dt;
        white_noise_cov.block<3, 3>(ES_BA - ES_BG, ES_BA - ES_BG) = cov_a * inv_dt;

        delta.cov.block<9, 9>(ES_Q, ES_Q) = A * delta.cov.block<9, 9>(0, 0) * A.transpose() + B * white_noise_cov * B.transpose();
        delta.cov.block<3, 3>(ES_BG, ES_BG) += cov_bg * dt;
        delta.cov.block<3, 3>(ES_BA, ES_BA) += cov_ba * dt;
    }

    if (compute_jacobian) {
        jacobian.dp_dbg += dt * jacobian.dv_dbg - 0.5 * dt * dt * delta.q.matrix() * hat(a) * jacobian.dq_dbg;
        jacobian.dp_dba += dt * jacobian.dv_dba - 0.5 * dt * dt * delta.q.matrix();
        jacobian.dv_dbg -= dt * delta.q.matrix() * hat(a) * jacobian.dq_dbg;
        jacobian.dv_dba -= dt * delta.q.matrix();
        jacobian.dq_dbg = expmap(w * dt).conjugate().matrix() * jacobian.dq_dbg - dt * right_jacobian(w * dt);
    }


    // 欧拉法积分 假设预积分初始图片为i，结束图片为i+1；中间有m个IMU数据
    // 相邻两个IMU之间的递推公式k->k+1  q 表示k时刻到i时刻的相对旋转
    delta.t = delta.t + dt;
    delta.p = delta.p + dt * delta.v + 0.5 * dt * dt * (delta.q * a);        // q_k_to_i * a_k = a_i
    delta.v = delta.v + dt * (delta.q * a);
    delta.q = (delta.q * expmap(w * dt)).normalized();   // q_k_to_i * q_k+1_to_k = q_k+1_to_i
}

// t 表示图片的时间戳 data 表示相邻两帧图片之间的imu数据，有很多个IMU数据，bg和ba都是上一帧图像的，
bool PreIntegrator::integrate(double t, const Vector3d &bg, const Vector3d &ba, bool compute_jacobian, bool compute_covariance) {
    if (data.size() == 0) return false;
    reset();
    for (size_t i = 0; i + 1 < data.size(); ++i) {       // 将相邻两帧图片之间的所有imu逐个积分
        const IMUData &d = data[i];
        increment(data[i + 1].t - d.t, d, bg, ba, compute_jacobian, compute_covariance);
    }
    // 最后一次进行插值预积分
    assert(("Image time cannot be earlier than last imu.") && (t >= data.back().t));
    increment(t - data.back().t, data.back(), bg, ba, compute_jacobian, compute_covariance);
    if (compute_covariance) {
        compute_sqrt_inv_cov();
    }
    return true;
}

void PreIntegrator::compute_sqrt_inv_cov() {
    delta.sqrt_inv_cov = LLT<Matrix<double, 15, 15>>(delta.cov.inverse()).matrixL().transpose();
}
