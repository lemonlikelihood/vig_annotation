#pragma once

#include <slamtools/common.h>

 // 表明每一个参数在状态向量中的位置
enum ErrorStateLocation {
    ES_Q = 0,
    ES_P = 3,
    ES_V = 6,
    ES_BG = 9,
    ES_BA = 12,
    ES_SIZE = 15
};

// （位置信息）某一个参数是否固定不变，不参与优化
enum PoseFlag {
    PF_FIXED = 0,
    PF_SIZE
};

// 3D点是否固定，不参与优化
enum LandmarkFlag {
    LF_VALID = 0,
    LF_SIZE
};

// 外参数，传感器（body系）到相机的旋转
struct ExtrinsicParams {
    Eigen::Quaterniond q_cs;
    Eigen::Vector3d p_cs;
};

struct PoseState {                                                           // 位姿状态 q,p,flag
    PoseState() {
        q.setIdentity();
        p.setZero();
        flags.reset();
    }

    bool flag(PoseFlag f) const {
        return flags[f];
    }

    std::bitset<PF_SIZE>::reference flag(PoseFlag f) {
        return flags[f];
    }

    Eigen::Quaterniond q;
    Eigen::Vector3d p;

  private:
    std::bitset<PF_SIZE> flags;
};

struct MotionState {                                                // imu 运动 motion 状态, v,bg,ba
    MotionState() {
        v.setZero();
        bg.setZero();
        ba.setZero();
    }

    Eigen::Vector3d v;
    Eigen::Vector3d bg;
    Eigen::Vector3d ba;
};

struct LandmarkState {
    LandmarkState() {
        x.setZero();
        quality = 0;
        flags.reset();
    }

    Eigen::Vector3d x;                                                  // landmark 3D点 坐标
    double quality;                                                     // landmark 质量

    bool flag(LandmarkFlag f) const {                                   // 该3D点是否被三角化
        return flags[f];
    }

    std::bitset<LF_SIZE>::reference flag(LandmarkFlag f) {
        return flags[f];
    }

  private:
    std::bitset<LF_SIZE> flags;
};
