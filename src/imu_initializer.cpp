/**************************************************************************
* VIG-Init
*
* Copyright SenseTime. All Rights Reserved.
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
**************************************************************************/

#include "imu_initializer.h"
#include <slamtools/frame.h>
#include <slamtools/track.h>
#include <slamtools/sliding_window.h>
#include <slamtools/configurator.h>
#include <slamtools/lie_algebra.h>
#include <slamtools/stereo.h>

using namespace Eigen;

#define init_recipie(                                                  \
    solve_gyro_bias,                                                   \
    solve_scale_given_gravity,                                         \
    solve_gravity_scale,                                               \
    refine_gravity,                                                    \
    refine_scale)                                                      \
    do {                                                               \
        clear();                                                       \
                                                                       \
        solve_gyro_bias();                                             \
                                                                       \
        solve_gravity_scale();                                         \
                                                                       \
        if (scale < 0.02 || scale > 1.0) return false;                 \
                                                                       \
        if (!config->init_refine_imu()) {                              \
            return apply_init();                                       \
        }                                                              \
                                                                       \
        refine_gravity();                                              \
        refine_scale();                                                \
                                                                       \
        if (scale < 0.02 || scale > 1.0) return false;                 \
                                                                       \
        return apply_init();                                           \
    } while (0)

inline void skip() {
}

ImuInitializer::ImuInitializer(SlidingWindow *map, std::shared_ptr<Configurator> config) :
    map(map), config(config) {
}

ImuInitializer::~ImuInitializer() = default;

void ImuInitializer::clear() {          // bg ba gravity scale velocities 全部设置为0
    bg.setZero();
    ba.setZero();
    gravity.setZero();
    scale = 1;
    velocities.resize(map->frame_num(), Vector3d::Zero());
}


//  1. 求解角速度bias,bg
//  2. 求解重力方向g,尺度s,速度v
//  3. 判断尺度信息是否合理 if (scale < 0.02 || scale > 1.0) return false;  
//  4. 利用垂直边优化重力方向
//  5. 固定重力方向和大小，重新求解尺度s和速度v，即固定g，重新求解第二步
//  6. 再次判断尺度信息是否合理 if (scale < 0.02 || scale > 1.0) return false;  
//  7. 应用初始化信息 apply_init()
bool ImuInitializer::init() {        
    // Recipie of VINS-Mono
    // init_recipie(
    //     solve_gyro_bias,
    //     skip,
    //     solve_gravity_scale_velocity,
    //     skip,
    //     refine_scale_velocity_via_gravity);

    // Recipie of ORB-SLAM
    // init_recipie(
    //     solve_gyro_bias,
    //     solve_scale_given_gravity,
    //     solve_gravity_scale,
    //     skip,
    //     refine_scale_ba_via_gravity);

    init_recipie(
        solve_gyro_bias,
        skip,
        solve_gravity_scale_velocity,
        refine_gravity_via_visual,
        solve_scale_velocity_given_gravity);

}

// 对所有图像帧进行预积分
void ImuInitializer::integrate() {
    for (size_t j = 1; j < map->frame_num(); ++j) {
        Frame *frame_j = map->get_frame(j);
        frame_j->preintegration.integrate(frame_j->image->t, bg, ba, true, false);
    }
}


// 1. 求解角速度bias,bg
void ImuInitializer::solve_gyro_bias() {
    integrate();
    Matrix3d A = Matrix3d::Zero();
    Vector3d b = Vector3d::Zero();

    for (size_t j = 1; j < map->frame_num(); ++j) {
        const size_t i = j - 1;

        const Frame *frame_i = map->get_frame(i);
        const Frame *frame_j = map->get_frame(j);

        const PoseState pose_i = frame_i->get_pose(frame_i->imu);        // 利用IMU和camera外参得到世界坐标系下IMU的位姿
        const PoseState pose_j = frame_j->get_pose(frame_j->imu);

        const Quaterniond &dq = frame_j->preintegration.delta.q;         // 第i帧到第j帧之间的旋转，利用IMU得到
        const Matrix3d &dq_dbg = frame_j->preintegration.jacobian.dq_dbg;  // dq_dbg 表示当第i帧的bg变化了一点之后，第j帧和第i帧之间的旋转dq变化增量，这个量存在于第j帧
        A += dq_dbg.transpose() * dq_dbg;
        b += dq_dbg.transpose() * logmap((pose_i.q * dq).conjugate() * pose_j.q);  // || AX -b ||
    }

    JacobiSVD<Matrix3d> svd(A, ComputeFullU | ComputeFullV);
    bg = svd.solve(b);
}

void ImuInitializer::solve_gyro_bias_exhaust() {
    Matrix3d A = Matrix3d::Zero();
    Vector3d b = Vector3d::Zero();

    for (size_t i = 0; i + 1 < map->frame_num(); ++i) {
        const Frame *frame_i = map->get_frame(i);
        for (size_t j = i + 1; j < map->frame_num(); ++j) {
            const Frame *frame_j = map->get_frame(j);

            PreIntegrator pre_ij;
            for (size_t u = i + 1; u <= j; ++u) {
                const Frame *frame_u = map->get_frame(u);
                pre_ij.data.insert(pre_ij.data.end(), frame_u->preintegration.data.begin(), frame_u->preintegration.data.end());
            }
            pre_ij.integrate(frame_j->image->t, bg, ba, true, false);

            const PoseState pose_i = frame_i->get_pose(frame_i->imu);
            const PoseState pose_j = frame_j->get_pose(frame_j->imu);

            const Quaterniond &dq = frame_j->preintegration.delta.q;
            const Matrix3d &dq_dbg = frame_j->preintegration.jacobian.dq_dbg;
            A += dq_dbg.transpose() * dq_dbg;
            b += dq_dbg.transpose() * logmap((pose_i.q * dq).conjugate() * pose_j.q);
        }
    }

    JacobiSVD<Matrix3d> svd(A, ComputeFullU | ComputeFullV);
    bg = svd.solve(b);
}

void ImuInitializer::solve_gravity_from_external() {
    Vector3d new_gravity = Vector3d::Zero();

    for (size_t i = 0; i < map->frame_num(); ++i) {
        const Frame *frame = map->get_frame(i);
        const PoseState pose = frame->get_pose(frame->imu);
        Vector3d frame_gravity = pose.q * frame->external_gravity;
        new_gravity += frame_gravity;
    }

    new_gravity /= (double)map->frame_num();
    gravity = new_gravity.normalized() * GRAVITY_NOMINAL;
}

void ImuInitializer::solve_gravity_scale() {
    integrate();
    Matrix4d A = Matrix4d::Zero();
    Vector4d b = Vector4d::Zero();

    for (size_t j = 1; j + 1 < map->frame_num(); ++j) {
        const size_t i = j - 1;
        const size_t k = j + 1;

        const Frame *frame_i = map->get_frame(i);
        const Frame *frame_j = map->get_frame(j);
        const Frame *frame_k = map->get_frame(k);
        const PreIntegrator::Delta &delta_ij = frame_j->preintegration.delta;
        const PreIntegrator::Delta &delta_jk = frame_k->preintegration.delta;
        const PreIntegrator::Jacobian &jacobian_ij = frame_j->preintegration.jacobian;
        const PreIntegrator::Jacobian &jacobian_jk = frame_k->preintegration.jacobian;

        const PoseState pose_i = frame_i->get_pose(frame_i->imu);
        const PoseState pose_j = frame_j->get_pose(frame_j->imu);
        const PoseState pose_k = frame_k->get_pose(frame_k->imu);

        Matrix<double, 3, 4> C;
        C.block<3, 3>(0, 0) = -0.5 * delta_ij.t * delta_jk.t * (delta_ij.t + delta_jk.t) * Matrix3d::Identity();
        C.block<3, 1>(0, 3) = delta_ij.t * (pose_k.p - pose_j.p) - delta_jk.t * (pose_j.p - pose_i.p);
        Vector3d d = delta_ij.t * (pose_j.q * delta_jk.p) + delta_ij.t * delta_jk.t * (pose_i.q * delta_ij.v) - delta_jk.t * (pose_i.q * delta_ij.p);
        A += C.transpose() * C;
        b += C.transpose() * d;
    }

    JacobiSVD<Matrix4d> svd(A, ComputeFullU | ComputeFullV);
    Vector4d x = svd.solve(b);
    gravity = x.segment<3>(0).normalized() * GRAVITY_NOMINAL;
    scale = x(3);
}

void ImuInitializer::solve_gravity_scale_exhaust() {
    Matrix4d A = Matrix4d::Zero();
    Vector4d b = Vector4d::Zero();

    for (size_t i = 0; i + 2 < map->frame_num(); ++i) {
        const Frame *frame_i = map->get_frame(i);
        for (size_t j = i + 1; j + 1 < map->frame_num(); ++j) {
            const Frame *frame_j = map->get_frame(j);

            PreIntegrator pre_ij;
            for (size_t u = i + 1; u <= j; ++u) {
                const Frame *frame_u = map->get_frame(u);
                pre_ij.data.insert(pre_ij.data.end(), frame_u->preintegration.data.begin(), frame_u->preintegration.data.end());
            }
            pre_ij.integrate(frame_j->image->t, bg, ba, true, false);

            for (size_t k = j + 1; k < map->frame_num(); ++k) {
                const Frame *frame_k = map->get_frame(k);

                PreIntegrator pre_jk;
                for (size_t u = j + 1; u <= k; ++u) {
                    const Frame *frame_u = map->get_frame(u);
                    pre_jk.data.insert(pre_jk.data.end(), frame_u->preintegration.data.begin(), frame_u->preintegration.data.end());
                }
                pre_jk.integrate(frame_k->image->t, bg, ba, true, false);

                const PreIntegrator::Delta &delta_ij = pre_ij.delta;
                const PreIntegrator::Delta &delta_jk = pre_jk.delta;
                const PreIntegrator::Jacobian &jacobian_ij = pre_ij.jacobian;
                const PreIntegrator::Jacobian &jacobian_jk = pre_jk.jacobian;

                const PoseState pose_i = frame_i->get_pose(frame_i->imu);
                const PoseState pose_j = frame_j->get_pose(frame_j->imu);
                const PoseState pose_k = frame_k->get_pose(frame_k->imu);

                Matrix<double, 3, 4> C;
                C.block<3, 3>(0, 0) = -0.5 * delta_ij.t * delta_jk.t * (delta_ij.t + delta_jk.t) * Matrix3d::Identity();
                C.block<3, 1>(0, 3) = delta_ij.t * (pose_k.p - pose_j.p) - delta_jk.t * (pose_j.p - pose_i.p);
                Vector3d d = delta_ij.t * (pose_j.q * delta_jk.p) + delta_ij.t * delta_jk.t * (pose_i.q * delta_ij.v) - delta_jk.t * (pose_i.q * delta_ij.p);
                A += C.transpose() * C;
                b += C.transpose() * d;
            }
        }
    }

    JacobiSVD<Matrix4d> svd(A, ComputeFullU | ComputeFullV);
    Vector4d x = svd.solve(b);
    gravity = x.segment<3>(0).normalized() * GRAVITY_NOMINAL;
    scale = x(3);
}

//  2. 求解重力g,尺度s,速度v          最终求解出来的g只保留了方向   [g,s,v1,v2,...vn]
void ImuInitializer::solve_gravity_scale_velocity() {
    integrate();
    int N = (int)map->frame_num();
    MatrixXd A;
    VectorXd b;
    A.resize((N - 1) * 6, 3 + 1 + 3 * N);
    b.resize((N - 1) * 6);
    A.setZero();
    b.setZero();

    for (size_t j = 1; j < map->frame_num(); ++j) {
        const size_t i = j - 1;

        const Frame *frame_i = map->get_frame(i);
        const Frame *frame_j = map->get_frame(j);
        const PreIntegrator::Delta &delta = frame_j->preintegration.delta;
        const PoseState pose_i = frame_i->get_pose(frame_i->imu);
        const PoseState pose_j = frame_j->get_pose(frame_j->imu);

        A.block<3, 3>(i * 6, 0) = -0.5 * delta.t * delta.t * Matrix3d::Identity();
        A.block<3, 1>(i * 6, 3) = pose_j.p - pose_i.p;
        A.block<3, 3>(i * 6, 4 + i * 3) = -delta.t * Matrix3d::Identity();
        b.segment<3>(i * 6) = pose_i.q * delta.p;

        A.block<3, 3>(i * 6 + 3, 0) = -delta.t * Matrix3d::Identity();
        A.block<3, 3>(i * 6 + 3, 4 + i * 3) = -Matrix3d::Identity();
        A.block<3, 3>(i * 6 + 3, 4 + j * 3) = Matrix3d::Identity();
        b.segment<3>(i * 6 + 3) = pose_i.q * delta.v;
    }

    VectorXd x = A.fullPivHouseholderQr().solve(b);
    gravity = x.segment<3>(0).normalized() * GRAVITY_NOMINAL;
    scale = x(3);
    for (size_t i = 0; i < map->frame_num(); ++i) {
        velocities[i] = x.segment<3>(4 + i * 3);
    }
}

void ImuInitializer::solve_gravity_scale_velocity_exhaust() {
    int N = (int)map->frame_num();
    int M = N * (N - 1) / 2;
    MatrixXd A;
    VectorXd b;
    A.resize(M * 6, 3 + 1 + 3 * N);
    b.resize(M * 6);
    A.setZero();
    b.setZero();

    size_t m = 0;
    for (size_t i = 0; i + 1 < map->frame_num(); ++i) {
        const Frame *frame_i = map->get_frame(i);
        for (size_t j = i + 1; j < map->frame_num(); ++j) {
            const Frame *frame_j = map->get_frame(j);

            PreIntegrator pre_ij;
            for (size_t u = i + 1; u <= j; ++u) {
                const Frame *frame_u = map->get_frame(u);
                pre_ij.data.insert(pre_ij.data.end(), frame_u->preintegration.data.begin(), frame_u->preintegration.data.end());
            }
            pre_ij.integrate(frame_j->image->t, bg, ba, true, false);

            const PreIntegrator::Delta &delta = pre_ij.delta;
            const PoseState pose_i = frame_i->get_pose(frame_i->imu);
            const PoseState pose_j = frame_j->get_pose(frame_j->imu);

            A.block<3, 3>(m * 6, 0) = -0.5 * delta.t * delta.t * Matrix3d::Identity();
            A.block<3, 1>(m * 6, 3) = pose_j.p - pose_i.p;
            A.block<3, 3>(m * 6, 4 + i * 3) = -delta.t * Matrix3d::Identity();
            b.segment<3>(m * 6) = pose_i.q * delta.p;

            A.block<3, 3>(m * 6 + 3, 0) = -delta.t * Matrix3d::Identity();
            A.block<3, 3>(m * 6 + 3, 4 + i * 3) = -Matrix3d::Identity();
            A.block<3, 3>(m * 6 + 3, 4 + j * 3) = Matrix3d::Identity();
            b.segment<3>(m * 6 + 3) = pose_i.q * delta.v;

            m++;
        }
    }
    assert(("Linear system is not correctly filled.") && ((int)m == M));

    VectorXd x = A.fullPivHouseholderQr().solve(b);
    gravity = x.segment<3>(0).normalized() * GRAVITY_NOMINAL;
    scale = x(3);
    for (size_t i = 0; i < map->frame_num(); ++i) {
        velocities[i] = x.segment<3>(4 + i * 3);
    }
}

void ImuInitializer::solve_scale_given_gravity() {
    integrate();
    double A = 0;
    double b = 0;

    for (size_t j = 1; j + 1 < map->frame_num(); ++j) {
        const size_t i = j - 1;
        const size_t k = j + 1;

        const Frame *frame_i = map->get_frame(i);
        const Frame *frame_j = map->get_frame(j);
        const Frame *frame_k = map->get_frame(k);
        const PreIntegrator::Delta &delta_ij = frame_j->preintegration.delta;
        const PreIntegrator::Delta &delta_jk = frame_k->preintegration.delta;
        const PoseState pose_i = frame_i->get_pose(frame_i->imu);
        const PoseState pose_j = frame_j->get_pose(frame_j->imu);
        const PoseState pose_k = frame_k->get_pose(frame_k->imu);

        Vector3d C = delta_ij.t * (pose_k.p - pose_j.p) - delta_jk.t * (pose_j.p - pose_i.p);
        Vector3d d = 0.5 * delta_ij.t * delta_jk.t * (delta_ij.t + delta_jk.t) * gravity + delta_ij.t * (pose_j.q * delta_jk.p) + delta_ij.t * delta_jk.t * (pose_i.q * delta_ij.v) - delta_jk.t * (pose_i.q * delta_ij.p);
        A += C.transpose() * C;
        b += C.transpose() * d;
    }

    scale = b / A;
}

void ImuInitializer::solve_scale_given_gravity_exhaust() {
    double A = 0;
    double b = 0;

    for (size_t i = 0; i + 2 < map->frame_num(); ++i) {
        const Frame *frame_i = map->get_frame(i);
        for (size_t j = i + 1; j + 1 < map->frame_num(); ++j) {
            const Frame *frame_j = map->get_frame(j);

            PreIntegrator pre_ij;
            for (size_t u = i + 1; u <= j; ++u) {
                const Frame *frame_u = map->get_frame(u);
                pre_ij.data.insert(pre_ij.data.end(), frame_u->preintegration.data.begin(), frame_u->preintegration.data.end());
            }
            pre_ij.integrate(frame_j->image->t, bg, ba, true, false);

            for (size_t k = j + 1; k < map->frame_num(); ++k) {
                const Frame *frame_k = map->get_frame(k);

                PreIntegrator pre_jk;
                for (size_t u = j + 1; u <= k; ++u) {
                    const Frame *frame_u = map->get_frame(u);
                    pre_jk.data.insert(pre_jk.data.end(), frame_u->preintegration.data.begin(), frame_u->preintegration.data.end());
                }
                pre_jk.integrate(frame_k->image->t, bg, ba, true, false);

                const PreIntegrator::Delta &delta_ij = pre_ij.delta;
                const PreIntegrator::Delta &delta_jk = pre_jk.delta;
                const PoseState pose_i = frame_i->get_pose(frame_i->imu);
                const PoseState pose_j = frame_j->get_pose(frame_j->imu);
                const PoseState pose_k = frame_k->get_pose(frame_k->imu);

                Vector3d C = delta_ij.t * (pose_k.p - pose_j.p) - delta_jk.t * (pose_j.p - pose_i.p);
                Vector3d d = 0.5 * delta_ij.t * delta_jk.t * (delta_ij.t + delta_jk.t) * gravity + delta_ij.t * (pose_j.q * delta_jk.p) + delta_ij.t * delta_jk.t * (pose_i.q * delta_ij.v) - delta_jk.t * (pose_i.q * delta_ij.p);
                A += C.transpose() * C;
                b += C.transpose() * d;
            }
        }
    }

    scale = b / A;
}

void ImuInitializer::solve_scale_ba_given_gravity() {
    integrate();
    Matrix4d A = Matrix4d::Zero();
    Vector4d b = Vector4d::Zero();

    for (size_t j = 1; j + 1 < map->frame_num(); ++j) {
        const size_t i = j - 1;
        const size_t k = j + 1;

        const Frame *frame_i = map->get_frame(i);
        const Frame *frame_j = map->get_frame(j);
        const Frame *frame_k = map->get_frame(k);
        const PreIntegrator::Delta &delta_ij = frame_j->preintegration.delta;
        const PreIntegrator::Delta &delta_jk = frame_k->preintegration.delta;
        const PreIntegrator::Jacobian &jacobian_ij = frame_j->preintegration.jacobian;
        const PreIntegrator::Jacobian &jacobian_jk = frame_k->preintegration.jacobian;

        const PoseState pose_i = frame_i->get_pose(frame_i->imu);
        const PoseState pose_j = frame_j->get_pose(frame_j->imu);
        const PoseState pose_k = frame_k->get_pose(frame_k->imu);

        Matrix<double, 3, 4> C;
        C.block<3, 1>(0, 0) = delta_ij.t * (pose_k.p - pose_j.p) - delta_jk.t * (pose_j.p - pose_i.p);
        C.block<3, 3>(0, 1) = -(pose_j.q * jacobian_jk.dp_dba * delta_ij.t + pose_i.q * jacobian_ij.dv_dba * delta_ij.t * delta_jk.t - pose_i.q * jacobian_ij.dp_dba * delta_jk.t);
        Vector3d d = 0.5 * delta_ij.t * delta_jk.t * (delta_ij.t + delta_jk.t) * gravity + delta_ij.t * (pose_j.q * delta_jk.p) + delta_ij.t * delta_jk.t * (pose_i.q * delta_ij.v) - delta_jk.t * (pose_i.q * delta_ij.p);
        A += C.transpose() * C;
        b += C.transpose() * d;
    }

    JacobiSVD<Matrix4d> svd(A, ComputeFullU | ComputeFullV);
    Vector4d x = svd.solve(b);
    scale = x(0);
    ba = x.segment<3>(1);
}

void ImuInitializer::solve_scale_ba_given_gravity_exhaust() {
    Matrix4d A = Matrix4d::Zero();
    Vector4d b = Vector4d::Zero();

    for (size_t i = 0; i + 2 < map->frame_num(); ++i) {
        const Frame *frame_i = map->get_frame(i);
        for (size_t j = i + 1; j + 1 < map->frame_num(); ++j) {
            const Frame *frame_j = map->get_frame(j);

            PreIntegrator pre_ij;
            for (size_t u = i + 1; u <= j; ++u) {
                const Frame *frame_u = map->get_frame(u);
                pre_ij.data.insert(pre_ij.data.end(), frame_u->preintegration.data.begin(), frame_u->preintegration.data.end());
            }
            pre_ij.integrate(frame_j->image->t, bg, ba, true, false);

            for (size_t k = j + 1; k < map->frame_num(); ++k) {
                const Frame *frame_k = map->get_frame(k);

                PreIntegrator pre_jk;
                for (size_t u = j + 1; u <= k; ++u) {
                    const Frame *frame_u = map->get_frame(u);
                    pre_jk.data.insert(pre_jk.data.end(), frame_u->preintegration.data.begin(), frame_u->preintegration.data.end());
                }
                pre_jk.integrate(frame_k->image->t, bg, ba, true, false);

                const PreIntegrator::Delta &delta_ij = pre_ij.delta;
                const PreIntegrator::Delta &delta_jk = pre_jk.delta;
                const PreIntegrator::Jacobian &jacobian_ij = pre_ij.jacobian;
                const PreIntegrator::Jacobian &jacobian_jk = pre_jk.jacobian;

                const PoseState pose_i = frame_i->get_pose(frame_i->imu);
                const PoseState pose_j = frame_j->get_pose(frame_j->imu);
                const PoseState pose_k = frame_k->get_pose(frame_k->imu);

                Matrix<double, 3, 4> C;
                C.block<3, 1>(0, 0) = delta_ij.t * (pose_k.p - pose_j.p) - delta_jk.t * (pose_j.p - pose_i.p);
                C.block<3, 3>(0, 1) = -(pose_j.q * jacobian_jk.dp_dba * delta_ij.t + pose_i.q * jacobian_ij.dv_dba * delta_ij.t * delta_jk.t - pose_i.q * jacobian_ij.dp_dba * delta_jk.t);
                Vector3d d = 0.5 * delta_ij.t * delta_jk.t * (delta_ij.t + delta_jk.t) * gravity + delta_ij.t * (pose_j.q * delta_jk.p) + delta_ij.t * delta_jk.t * (pose_i.q * delta_ij.v) - delta_jk.t * (pose_i.q * delta_ij.p);
                A += C.transpose() * C;
                b += C.transpose() * d;
            }
        }
    }

    JacobiSVD<Matrix4d> svd(A, ComputeFullU | ComputeFullV);
    Vector4d x = svd.solve(b);
    scale = x(0);
    ba = x.segment<3>(1);
}

// 5. 固定重力方向和大小，重新求解尺度s和速度v，即固定g，重新求解第二步
void ImuInitializer::solve_scale_velocity_given_gravity() {
    integrate();
    int N = (int)map->frame_num();
    MatrixXd A;
    VectorXd b;
    A.resize((N - 1) * 6, 1 + 3 * N);
    b.resize((N - 1) * 6);
    A.setZero();
    b.setZero();

    for (size_t j = 1; j < map->frame_num(); ++j) {
        const size_t i = j - 1;

        const Frame *frame_i = map->get_frame(i);
        const Frame *frame_j = map->get_frame(j);
        const PreIntegrator::Delta &delta = frame_j->preintegration.delta;
        const PoseState pose_i = frame_i->get_pose(frame_i->imu);
        const PoseState pose_j = frame_j->get_pose(frame_j->imu);

        A.block<3, 1>(i * 6, 0) = pose_j.p - pose_i.p;
        A.block<3, 3>(i * 6, 1 + i * 3) = -delta.t * Matrix3d::Identity();
        b.segment<3>(i * 6) = 0.5 * delta.t * delta.t * gravity + pose_i.q * delta.p;

        A.block<3, 3>(i * 6 + 3, 1 + i * 3) = -Matrix3d::Identity();
        A.block<3, 3>(i * 6 + 3, 1 + j * 3) = Matrix3d::Identity();
        b.segment<3>(i * 6 + 3) = delta.t * gravity + pose_i.q * delta.v;
    }

    VectorXd x = A.fullPivHouseholderQr().solve(b);
    scale = x(0);
    for (size_t i = 0; i < map->frame_num(); ++i) {
        velocities[i] = x.segment<3>(1 + i * 3);
    }
}

void ImuInitializer::solve_scale_velocity_given_gravity_exhaust() {
    int N = (int)map->frame_num();
    int M = N * (N - 1) / 2;
    MatrixXd A;
    VectorXd b;
    A.resize(M * 6, 1 + 3 * N);
    b.resize(M * 6);
    A.setZero();
    b.setZero();

    size_t m = 0;
    for (size_t i = 0; i + 1 < map->frame_num(); ++i) {
        const Frame *frame_i = map->get_frame(i);
        for (size_t j = i + 1; j < map->frame_num(); ++j) {
            const Frame *frame_j = map->get_frame(j);

            PreIntegrator pre_ij;
            for (size_t u = i + 1; u <= j; ++u) {
                const Frame *frame_u = map->get_frame(u);
                pre_ij.data.insert(pre_ij.data.end(), frame_u->preintegration.data.begin(), frame_u->preintegration.data.end());
            }
            pre_ij.integrate(frame_j->image->t, bg, ba, true, false);

            const PreIntegrator::Delta &delta = pre_ij.delta;
            const PoseState pose_i = frame_i->get_pose(frame_i->imu);
            const PoseState pose_j = frame_j->get_pose(frame_j->imu);

            A.block<3, 1>(m * 6, 0) = pose_j.p - pose_i.p;
            A.block<3, 3>(m * 6, 1 + i * 3) = -delta.t * Matrix3d::Identity();
            b.segment<3>(m * 6) = 0.5 * delta.t * delta.t * gravity + pose_i.q * delta.p;

            A.block<3, 3>(m * 6 + 3, 1 + i * 3) = -Matrix3d::Identity();
            A.block<3, 3>(m * 6 + 3, 1 + j * 3) = Matrix3d::Identity();
            b.segment<3>(m * 6 + 3) = delta.t * gravity + pose_i.q * delta.v;

            m++;
        }
    }
    assert(("Linear system is not correctly filled.") && ((int)m == M));

    VectorXd x = A.fullPivHouseholderQr().solve(b);
    scale = x(0);
    for (size_t i = 0; i < map->frame_num(); ++i) {
        velocities[i] = x.segment<3>(1 + i * 3);
    }
}

void ImuInitializer::solve_scale_ba_velocity_given_gravity() {
    integrate();
    int N = (int)map->frame_num();
    MatrixXd A;
    VectorXd b;
    A.resize((N - 1) * 6, 1 + 3 + 3 * N);
    b.resize((N - 1) * 6);
    A.setZero();
    b.setZero();

    for (size_t j = 1; j < map->frame_num(); ++j) {
        const size_t i = j - 1;

        const Frame *frame_i = map->get_frame(i);
        const Frame *frame_j = map->get_frame(j);
        const PreIntegrator::Delta &delta = frame_j->preintegration.delta;
        const PreIntegrator::Jacobian &jacobian = frame_j->preintegration.jacobian;
        const PoseState pose_i = frame_i->get_pose(frame_i->imu);
        const PoseState pose_j = frame_j->get_pose(frame_j->imu);

        A.block<3, 1>(i * 6, 0) = pose_j.p - pose_i.p;
        A.block<3, 3>(i * 6, 1) = -pose_i.q.matrix() * jacobian.dp_dba;
        A.block<3, 3>(i * 6, 4 + i * 3) = -delta.t * Matrix3d::Identity();
        b.segment<3>(i * 6) = 0.5 * delta.t * delta.t * gravity + pose_i.q * delta.p;

        A.block<3, 3>(i * 6 + 3, 1) = -pose_i.q.matrix() * jacobian.dv_dba;
        A.block<3, 3>(i * 6 + 3, 4 + i * 3) = -Matrix3d::Identity();
        A.block<3, 3>(i * 6 + 3, 4 + j * 3) = Matrix3d::Identity();
        b.segment<3>(i * 6 + 3) = delta.t * gravity + pose_i.q * delta.v;
    }

    VectorXd x = A.fullPivHouseholderQr().solve(b);
    scale = x(0);
    ba = x.segment<3>(1);
    for (size_t i = 0; i < map->frame_num(); ++i) {
        velocities[i] = x.segment<3>(4 + i * 3);
    }
}

void ImuInitializer::solve_scale_ba_velocity_given_gravity_exhaust() {
    int N = (int)map->frame_num();
    int M = N * (N - 1) / 2;
    MatrixXd A;
    VectorXd b;
    A.resize(M * 6, 1 + 3 + 3 * N);
    b.resize(M * 6);
    A.setZero();
    b.setZero();

    size_t m = 0;
    for (size_t i = 0; i + 1 < map->frame_num(); ++i) {
        const Frame *frame_i = map->get_frame(i);
        for (size_t j = i + 1; j < map->frame_num(); ++j) {
            const Frame *frame_j = map->get_frame(j);

            PreIntegrator pre_ij;
            for (size_t u = i + 1; u <= j; ++u) {
                const Frame *frame_u = map->get_frame(u);
                pre_ij.data.insert(pre_ij.data.end(), frame_u->preintegration.data.begin(), frame_u->preintegration.data.end());
            }
            pre_ij.integrate(frame_j->image->t, bg, ba, true, false);

            const PreIntegrator::Delta &delta = pre_ij.delta;
            const PreIntegrator::Jacobian &jacobian = pre_ij.jacobian;
            const PoseState pose_i = frame_i->get_pose(frame_i->imu);
            const PoseState pose_j = frame_j->get_pose(frame_j->imu);

            A.block<3, 1>(m * 6, 0) = pose_j.p - pose_i.p;
            A.block<3, 3>(m * 6, 1) = -pose_i.q.matrix() * jacobian.dp_dba;
            A.block<3, 3>(m * 6, 4 + i * 3) = -delta.t * Matrix3d::Identity();
            b.segment<3>(m * 6) = 0.5 * delta.t * delta.t * gravity + pose_i.q * delta.p;

            A.block<3, 3>(m * 6 + 3, 1) = -pose_i.q.matrix() * jacobian.dv_dba;
            A.block<3, 3>(m * 6 + 3, 4 + i * 3) = -Matrix3d::Identity();
            A.block<3, 3>(m * 6 + 3, 4 + j * 3) = Matrix3d::Identity();
            b.segment<3>(m * 6 + 3) = delta.t * gravity + pose_i.q * delta.v;

            m++;
        }
    }
    assert(("Linear system is not correctly filled.") && ((int)m == M));

    VectorXd x = A.fullPivHouseholderQr().solve(b);
    scale = x(0);
    ba = x.segment<3>(1);
    for (size_t i = 0; i < map->frame_num(); ++i) {
        velocities[i] = x.segment<3>(4 + i * 3);
    }
}

void ImuInitializer::refine_scale_ba_via_gravity() {
    static const double damp = 0.1;
    Matrix<double, 6, 6> A;
    Matrix<double, 6, 1> b;
    for (size_t iter = 0; iter < 1; ++iter) {
        integrate();
        A.setZero();
        b.setZero();
        Matrix<double, 3, 2> Tg = s2_tangential_basis(gravity);

        for (size_t j = 1; j + 1 < map->frame_num(); ++j) {
            const size_t i = j - 1;
            const size_t k = j + 1;

            const Frame *frame_i = map->get_frame(i);
            const Frame *frame_j = map->get_frame(j);
            const Frame *frame_k = map->get_frame(k);
            const PreIntegrator::Delta &delta_ij = frame_j->preintegration.delta;
            const PreIntegrator::Delta &delta_jk = frame_k->preintegration.delta;
            const PreIntegrator::Jacobian &jacobian_ij = frame_j->preintegration.jacobian;
            const PreIntegrator::Jacobian &jacobian_jk = frame_k->preintegration.jacobian;

            const PoseState pose_i = frame_i->get_pose(frame_i->imu);
            const PoseState pose_j = frame_j->get_pose(frame_j->imu);
            const PoseState pose_k = frame_k->get_pose(frame_k->imu);

            Matrix<double, 3, 6> C;
            C.block<3, 1>(0, 0) = delta_ij.t * (pose_k.p - pose_j.p) - delta_jk.t * (pose_j.p - pose_i.p);
            C.block<3, 3>(0, 1) = -(pose_j.q * jacobian_jk.dp_dba * delta_ij.t + pose_i.q * jacobian_ij.dv_dba * delta_ij.t * delta_jk.t - pose_i.q * jacobian_ij.dp_dba * delta_jk.t);
            C.block<3, 2>(0, 4) = -0.5 * delta_ij.t * delta_jk.t * (delta_ij.t + delta_jk.t) * Tg;
            Vector3d d = 0.5 * delta_ij.t * delta_jk.t * (delta_ij.t + delta_jk.t) * gravity + delta_ij.t * (pose_j.q * delta_jk.p) + delta_ij.t * delta_jk.t * (pose_i.q * delta_ij.v) - delta_jk.t * (pose_i.q * delta_ij.p);
            A += C.transpose() * C;
            b += C.transpose() * d;
        }

        JacobiSVD<Matrix<double, 6, 6>> svd(A, ComputeFullU | ComputeFullV);
        Matrix<double, 6, 1> x = svd.solve(b);
        scale = x(0);
        ba += damp * x.segment<3>(1);
        gravity = (gravity + damp * Tg * x.segment<2>(4)).normalized() * GRAVITY_NOMINAL;
    }
}

void ImuInitializer::refine_scale_ba_via_gravity_exhaust() {
    static const double damp = 0.1;
    Matrix<double, 6, 6> A;
    Matrix<double, 6, 1> b;

    for (size_t iter = 0; iter < 1; ++iter) {
        A.setZero();
        b.setZero();
        Matrix<double, 3, 2> Tg = s2_tangential_basis(gravity);

        for (size_t i = 0; i + 2 < map->frame_num(); ++i) {
            const Frame *frame_i = map->get_frame(i);
            for (size_t j = i + 1; j + 1 < map->frame_num(); ++j) {
                const Frame *frame_j = map->get_frame(j);

                PreIntegrator pre_ij;
                for (size_t u = i + 1; u <= j; ++u) {
                    const Frame *frame_u = map->get_frame(u);
                    pre_ij.data.insert(pre_ij.data.end(), frame_u->preintegration.data.begin(), frame_u->preintegration.data.end());
                }
                pre_ij.integrate(frame_j->image->t, bg, ba, true, false);

                for (size_t k = j + 1; k < map->frame_num(); ++k) {
                    const Frame *frame_k = map->get_frame(k);

                    PreIntegrator pre_jk;
                    for (size_t u = j + 1; u <= k; ++u) {
                        const Frame *frame_u = map->get_frame(u);
                        pre_jk.data.insert(pre_jk.data.end(), frame_u->preintegration.data.begin(), frame_u->preintegration.data.end());
                    }
                    pre_jk.integrate(frame_k->image->t, bg, ba, true, false);

                    const PreIntegrator::Delta &delta_ij = pre_ij.delta;
                    const PreIntegrator::Delta &delta_jk = pre_jk.delta;
                    const PreIntegrator::Jacobian &jacobian_ij = pre_ij.jacobian;
                    const PreIntegrator::Jacobian &jacobian_jk = pre_jk.jacobian;

                    const PoseState pose_i = frame_i->get_pose(frame_i->imu);
                    const PoseState pose_j = frame_j->get_pose(frame_j->imu);
                    const PoseState pose_k = frame_k->get_pose(frame_k->imu);

                    Matrix<double, 3, 6> C;
                    C.block<3, 1>(0, 0) = delta_ij.t * (pose_k.p - pose_j.p) - delta_jk.t * (pose_j.p - pose_i.p);
                    C.block<3, 3>(0, 1) = -(pose_j.q * jacobian_jk.dp_dba * delta_ij.t + pose_i.q * jacobian_ij.dv_dba * delta_ij.t * delta_jk.t - pose_i.q * jacobian_ij.dp_dba * delta_jk.t);
                    C.block<3, 2>(0, 4) = -0.5 * delta_ij.t * delta_jk.t * (delta_ij.t + delta_jk.t) * Tg;
                    Vector3d d = 0.5 * delta_ij.t * delta_jk.t * (delta_ij.t + delta_jk.t) * gravity + delta_ij.t * (pose_j.q * delta_jk.p) + delta_ij.t * delta_jk.t * (pose_i.q * delta_ij.v) - delta_jk.t * (pose_i.q * delta_ij.p);
                    A += C.transpose() * C;
                    b += C.transpose() * d;
                }
            }
        }

        JacobiSVD<Matrix<double, 6, 6>> svd(A, ComputeFullU | ComputeFullV);
        Matrix<double, 6, 1> x = svd.solve(b);
        scale = x(0);
        ba += damp * x.segment<3>(1);
        gravity = (gravity + damp * Tg * x.segment<2>(4)).normalized() * GRAVITY_NOMINAL;
    }
}

void ImuInitializer::refine_scale_velocity_via_gravity() {
    static const double damp = 0.1;
    integrate();
    int N = (int)map->frame_num();
    MatrixXd A;
    VectorXd b;
    VectorXd x;
    A.resize((N - 1) * 6, 2 + 1 + 3 * N);
    b.resize((N - 1) * 6);
    x.resize(2 + 1 + 3 * N);

    for (size_t iter = 0; iter < 1; ++iter) {
        A.setZero();
        b.setZero();
        Matrix<double, 3, 2> Tg = s2_tangential_basis(gravity);

        for (size_t j = 1; j < map->frame_num(); ++j) {
            const size_t i = j - 1;

            const Frame *frame_i = map->get_frame(i);
            const Frame *frame_j = map->get_frame(j);
            const PreIntegrator::Delta &delta = frame_j->preintegration.delta;
            const PoseState pose_i = frame_i->get_pose(frame_i->imu);
            const PoseState pose_j = frame_j->get_pose(frame_j->imu);

            A.block<3, 2>(i * 6, 0) = -0.5 * delta.t * delta.t * Tg;
            A.block<3, 1>(i * 6, 2) = pose_j.p - pose_i.p;
            A.block<3, 3>(i * 6, 3 + i * 3) = -delta.t * Matrix3d::Identity();
            b.segment<3>(i * 6) = 0.5 * delta.t * delta.t * gravity + pose_i.q * delta.p;

            A.block<3, 2>(i * 6 + 3, 0) = -delta.t * Tg;
            A.block<3, 3>(i * 6 + 3, 3 + i * 3) = -Matrix3d::Identity();
            A.block<3, 3>(i * 6 + 3, 3 + j * 3) = Matrix3d::Identity();
            b.segment<3>(i * 6 + 3) = delta.t * gravity + pose_i.q * delta.v;
        }

        x = A.fullPivHouseholderQr().solve(b);
        Vector2d dg = x.segment<2>(0);
        gravity = (gravity + damp * Tg * dg).normalized() * GRAVITY_NOMINAL;
    }

    scale = x(2);
    for (size_t i = 0; i < map->frame_num(); ++i) {
        velocities[i] = x.segment<3>(3 + i * 3);
    }
}

void ImuInitializer::refine_scale_velocity_via_gravity_exhaust() {
    static const double damp = 0.1;
    int N = (int)map->frame_num();
    int M = N * (N - 1) / 2;
    MatrixXd A;
    VectorXd b;
    VectorXd x;
    A.resize(M * 6, 2 + 1 + 3 * N);
    b.resize(M * 6);
    x.resize(2 + 1 + 3 * N);

    for (size_t iter = 0; iter < 1; ++iter) {
        A.setZero();
        b.setZero();
        Matrix<double, 3, 2> Tg = s2_tangential_basis(gravity);

        size_t m = 0;
        for (size_t i = 0; i + 1 < map->frame_num(); ++i) {
            const Frame *frame_i = map->get_frame(i);
            for (size_t j = i + 1; j < map->frame_num(); ++j) {
                const Frame *frame_j = map->get_frame(j);

                PreIntegrator pre_ij;
                for (size_t u = i + 1; u <= j; ++u) {
                    const Frame *frame_u = map->get_frame(u);
                    pre_ij.data.insert(pre_ij.data.end(), frame_u->preintegration.data.begin(), frame_u->preintegration.data.end());
                }
                pre_ij.integrate(frame_j->image->t, bg, ba, true, false);

                const PreIntegrator::Delta &delta = pre_ij.delta;
                const PreIntegrator::Jacobian &jacobian = pre_ij.jacobian;
                const PoseState pose_i = frame_i->get_pose(frame_i->imu);
                const PoseState pose_j = frame_j->get_pose(frame_j->imu);

                A.block<3, 2>(m * 6, 0) = -0.5 * delta.t * delta.t * Tg;
                A.block<3, 1>(m * 6, 2) = pose_j.p - pose_i.p;
                A.block<3, 3>(m * 6, 3 + i * 3) = -delta.t * Matrix3d::Identity();
                b.segment<3>(m * 6) = 0.5 * delta.t * delta.t * gravity + pose_i.q * delta.p;

                A.block<3, 2>(m * 6 + 3, 0) = -delta.t * Tg;
                A.block<3, 3>(m * 6 + 3, 3 + i * 3) = -Matrix3d::Identity();
                A.block<3, 3>(m * 6 + 3, 3 + j * 3) = Matrix3d::Identity();
                b.segment<3>(m * 6 + 3) = delta.t * gravity + pose_i.q * delta.v;

                m++;
            }
        }
        assert(("Linear system is not correctly filled.") && ((int)m == M));

        x = A.fullPivHouseholderQr().solve(b);
        Vector2d dg = x.segment<2>(0);
        gravity = (gravity + damp * Tg * dg).normalized() * GRAVITY_NOMINAL;
    }

    scale = x(2);
    for (size_t i = 0; i < map->frame_num(); ++i) {
        velocities[i] = x.segment<3>(3 + i * 3);
    }
}

// 4. 利用垂直边优化重力方向
bool ImuInitializer::refine_gravity_via_visual() {
    Matrix3d cdt = Matrix3d::Zero();
    Vector3d down = gravity.normalized();
    size_t segnum = 0;
    for (size_t i = 0; i < map->frame_num(); ++i) {
        const Frame *frame = map->get_frame(i);
        for (size_t j = 0; j < frame->segments.size(); ++j) {
            const Vector2d &p1 = std::get<0>(frame->segments[j]);
            const Vector2d &p2 = std::get<1>(frame->segments[j]);
            double seglen = (p1 - p2).norm();
            if (seglen * frame->K(0, 0) < 20) continue;
            Vector3d param = frame->get_pose(frame->camera).q * p1.homogeneous().cross(p2.homogeneous()).normalized(); // 从图像坐标系下的坐标转换到世界坐标系下的坐标直线坐标，归一化后得到方向向量
            double cosangle = param.dot(down);        // 得到直线边和重力的夹角cos值
            if (abs(cosangle) < sin(10.0 * M_PI / 180.0)) {   // cos值越小越接近90度,这里直线边与重力夹角得小于10度认为是垂直边
                cdt += seglen * seglen * param * param.transpose();    // 
                segnum++;
            }
        }
    }
    if (segnum < 10 * map->frame_num()) {              // 垂直边的数目必须足够多
        return false;
    }

    JacobiSVD<Matrix3d> svd(cdt, ComputeFullU | ComputeFullV);
    Vector3d down_visual = svd.matrixV().col(2);
    if (down_visual.dot(down) < 0) {
        down_visual = -down_visual;
    }
    gravity = down_visual * GRAVITY_NOMINAL;         // 得到新的重力值
    return true;
}

// 7. 应用初始化信息 apply_init() 重力的bias没有使用，IMU速度可以使用
bool ImuInitializer::apply_init(bool apply_ba, bool apply_velocity) {
    static const Vector3d gravity_nominal{0, 0, -GRAVITY_NOMINAL};

    Quaterniond q = Quaterniond::FromTwoVectors(gravity, gravity_nominal);   // 得到当前重力值和世界坐标系真实重力值的旋转
    for (size_t i = 0; i < map->frame_num(); ++i) {
        Frame *frame = map->get_frame(i);
        PoseState imu_pose = frame->get_pose(frame->imu);           
        imu_pose.q = q * imu_pose.q;                                         // q_b_to_w * q_I_to_b = q_I_to_w
        imu_pose.p = scale * (q * imu_pose.p);                               // 实际位置尺度化
        frame->set_pose(frame->imu, imu_pose);                               // 利用IMU在世界坐标系的位姿来设置body系在世界坐标系的位姿
        if (apply_velocity) {
            frame->motion.v = q * velocities[i];                             // 设置IMU的速度
        } else {
            frame->motion.v.setZero();
        }
        frame->motion.bg = bg;                                               // 设置IMU的bg
        if (apply_ba) {                                                      // 设置IMU的ba
            frame->motion.ba = ba;
        } else {
            frame->motion.ba.setZero();
        }
    }
    size_t result_point_num = 0;
    for (size_t i = 0; i < map->track_num(); ++i) {
        if (map->get_track(i)->triangulate()) {
            result_point_num++;
        }
    }

    return result_point_num >= config->min_init_map_landmarks();            // 判断初始化最小的landmark是否满足
}
