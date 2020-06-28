#pragma once

#include <slamtools/common.h>

// 从归一化平面到图像坐标系
inline Eigen::Vector2d apply_k(const Eigen::Vector2d &p, const Eigen::Matrix3d &K) {
    return {p(0) * K(0, 0) + K(0, 2), p(1) * K(1, 1) + K(1, 2)};
}

// 从图像坐标系到归一化平面
inline Eigen::Vector2d remove_k(const Eigen::Vector2d &p, const Eigen::Matrix3d &K) {
    return {(p(0) - K(0, 2)) / K(0, 0), (p(1) - K(1, 2)) / K(1, 1)};
}

Eigen::Matrix3d find_essential_matrix(const std::vector<Eigen::Vector2d> &points1, const std::vector<Eigen::Vector2d> &points2, double threshold = 1.0, double confidence = 0.999, size_t max_iteration = 1000, int seed = 0);
Eigen::Matrix3d find_homography_matrix(const std::vector<Eigen::Vector2d> &points1, const std::vector<Eigen::Vector2d> &points2, double threshold = 1.0, double confidence = 0.999, size_t max_iteration = 1000, int seed = 0);
Eigen::Matrix3d find_essential_matrix(const std::vector<Eigen::Vector2d> &points1, const std::vector<Eigen::Vector2d> &points2, std::vector<char> &inlier_mask, double threshold = 1.0, double confidence = 0.999, size_t max_iteration = 1000, int seed = 0);
Eigen::Matrix3d find_homography_matrix(const std::vector<Eigen::Vector2d> &points1, const std::vector<Eigen::Vector2d> &points2, std::vector<char> &inlier_mask, double threshold = 1.0, double confidence = 0.999, size_t max_iteration = 1000, int seed = 0);

Eigen::Vector4d triangulate_point(const Eigen::Matrix<double, 3, 4> &P1, const Eigen::Matrix<double, 3, 4> &P2, const Eigen::Vector2d &point1, const Eigen::Vector2d &point2);
Eigen::Vector4d triangulate_point(const std::vector<Eigen::Matrix<double, 3, 4>> &Ps, const std::vector<Eigen::Vector2d> &points);

bool triangulate_point_checked(const Eigen::Matrix<double, 3, 4> &P1, const Eigen::Matrix<double, 3, 4> &P2, const Eigen::Vector2d &point1, const Eigen::Vector2d &point2, Eigen::Vector3d &p);
bool triangulate_point_checked(const std::vector<Eigen::Matrix<double, 3, 4>> &Ps, const std::vector<Eigen::Vector2d> &points, Eigen::Vector3d &p);
bool triangulate_point_scored(const Eigen::Matrix<double, 3, 4> &P1, const Eigen::Matrix<double, 3, 4> &P2, const Eigen::Vector2d &point1, const Eigen::Vector2d &point2, Eigen::Vector3d &p, double &score);
bool triangulate_point_scored(const std::vector<Eigen::Matrix<double, 3, 4>> &Ps, const std::vector<Eigen::Vector2d> &points, Eigen::Vector3d &p, double &score);

size_t triangulate_from_rt(const std::vector<Eigen::Vector2d> &points1, const std::vector<Eigen::Vector2d> &points2, const Eigen::Matrix3d &R, const Eigen::Vector3d &T, std::vector<Eigen::Vector3d> &result_points, std::vector<char> &result_status);
size_t triangulate_from_rt(const std::vector<Eigen::Vector2d> &points1, const std::vector<Eigen::Vector2d> &points2, const std::vector<Eigen::Matrix3d> &Rs, const std::vector<Eigen::Vector3d> &Ts, std::vector<Eigen::Vector3d> &result_points, Eigen::Matrix3d &result_R, Eigen::Vector3d &result_T, std::vector<char> &result_status);

size_t triangulate_from_rt_scored(const std::vector<Eigen::Vector2d> &points1, const std::vector<Eigen::Vector2d> &points2, const Eigen::Matrix3d &R, const Eigen::Vector3d &T, std::vector<Eigen::Vector3d> &result_points, std::vector<char> &result_status, double &score);
size_t triangulate_from_rt_scored(const std::vector<Eigen::Vector2d> &points1, const std::vector<Eigen::Vector2d> &points2, const std::vector<Eigen::Matrix3d> &Rs, const std::vector<Eigen::Vector3d> &Ts, size_t count_threshold, std::vector<Eigen::Vector3d> &result_points, Eigen::Matrix3d &result_R, Eigen::Vector3d &result_T, std::vector<char> &result_status);

size_t triangulate_from_essential(const std::vector<Eigen::Vector2d> &points1, const std::vector<Eigen::Vector2d> &points2, const Eigen::Matrix3d &E, std::vector<Eigen::Vector3d> &result_points, std::vector<char> &result_status, Eigen::Matrix3d &R, Eigen::Vector3d &T);
size_t triangulate_from_homography(const std::vector<Eigen::Vector2d> &points1, const std::vector<Eigen::Vector2d> &points2, const Eigen::Matrix3d &H, std::vector<Eigen::Vector3d> &result_points, std::vector<char> &result_status, Eigen::Matrix3d &R, Eigen::Vector3d &T);

// 两帧图像三角化，利用叉乘原理，返回三维点的齐次坐标
inline Eigen::Vector4d triangulate_point(const Eigen::Matrix<double, 3, 4> &P1, const Eigen::Matrix<double, 3, 4> &P2, const Eigen::Vector2d &point1, const Eigen::Vector2d &point2) {
    Eigen::Matrix4d A;
    A.row(0) = point1(0) * P1.row(2) - P1.row(0);
    A.row(1) = point1(1) * P1.row(2) - P1.row(1);
    A.row(2) = point2(0) * P2.row(2) - P2.row(0);
    A.row(3) = point2(1) * P2.row(2) - P2.row(1);
    return A.jacobiSvd(Eigen::ComputeFullV).matrixV().col(3);
}

inline Eigen::Vector4d triangulate_point(const std::vector<Eigen::Matrix<double, 3, 4>> &Ps, const std::vector<Eigen::Vector2d> &points) {
    Eigen::Matrix<double, Eigen::Dynamic, 4> A(points.size() * 2, 4);
    for (size_t i = 0; i < points.size(); ++i) {
        A.row(i * 2 + 0) = points[i](0) * Ps[i].row(2) - Ps[i].row(0);
        A.row(i * 2 + 1) = points[i](1) * Ps[i].row(2) - Ps[i].row(1);
    }
    return A.jacobiSvd(Eigen::ComputeFullV).matrixV().col(3);
}

inline bool triangulate_point_checked(const Eigen::Matrix<double, 3, 4> &P1, const Eigen::Matrix<double, 3, 4> &P2, const Eigen::Vector2d &point1, const Eigen::Vector2d &point2, Eigen::Vector3d &p) {
    double score;
    return triangulate_point_scored(P1, P2, point1, point2, p, score);
}

inline bool triangulate_point_checked(const std::vector<Eigen::Matrix<double, 3, 4>> &Ps, const std::vector<Eigen::Vector2d> &points, Eigen::Vector3d &p) {
    double score;
    return triangulate_point_scored(Ps, points, p, score);
}


// 计算两帧图像之间某一个匹配点的三角化，返回得分和是否成功三角化
inline bool triangulate_point_scored(const Eigen::Matrix<double, 3, 4> &P1, const Eigen::Matrix<double, 3, 4> &P2, const Eigen::Vector2d &point1, const Eigen::Vector2d &point2, Eigen::Vector3d &p, double &score) {
    Eigen::Vector4d q = triangulate_point(P1, P2, point1, point2);   // 两帧图像三角化，利用叉乘原理，返回三维点的齐次坐标
    score = 0;

    Eigen::Vector3d q1 = P1 * q;               // q1 表示三维点在P1坐标系下的坐标
    Eigen::Vector3d q2 = P2 * q;               // q2 表示三维点在P2坐标系下的坐标

    if (q1[2] * q[3] > 0 && q2[2] * q[3] > 0) {          // 在同一侧
        if (q1[2] / q[3] < 100 && q2[2] / q[3] < 100) {  
            p = q.hnormalized();              // p 表示三维点在世界坐标系下的坐标
            score = 0.5 * ((q1.hnormalized() - point1).squaredNorm() + (q2.hnormalized() - point2).squaredNorm());
            return true;                      // 得分是重投影误差，得分越小越好
        }
    }

    return false;
}

// 利用多帧图片来进行三角化，p是三角化的结果，score是三角化的重投影误差
inline bool triangulate_point_scored(const std::vector<Eigen::Matrix<double, 3, 4>> &Ps, const std::vector<Eigen::Vector2d> &points, Eigen::Vector3d &p, double &score) {
    if (Ps.size() < 2) return false;
    Eigen::Vector4d q = triangulate_point(Ps, points);    // 每一帧观测数据都会提供两个约束方程，至少需要两个观测才能三角化
    score = 0;

    for (size_t i = 0; i < points.size(); ++i) {
        Eigen::Vector3d qi = Ps[i] * q;
        if (!(qi[2] * q[3] > 0)) {
            return false;
        }
        if (!(qi[2] / q[3] < 100)) {
            return false;
        }
        score += (qi.hnormalized() - points[i]).squaredNorm();
    }
    score /= points.size();
    p = q.hnormalized();
    return true;
}
