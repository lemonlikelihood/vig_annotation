#pragma once

#include <slamtools/common.h>

class Configurator {
  public:
    static std::shared_ptr<Configurator> default_config();      // configurator 默认构造函数
    virtual ~Configurator();
    virtual Eigen::Matrix3d camera_intrinsic() const;           // 相机内参
    virtual double keypoint_pixel_error() const;
    virtual Eigen::Quaterniond camera_to_center_rotation() const;
    virtual Eigen::Vector3d camera_to_center_translation() const;
    virtual Eigen::Quaterniond imu_to_center_rotation() const;
    virtual Eigen::Vector3d imu_to_center_translation() const;
    virtual Eigen::Matrix3d imu_gyro_white_noise() const;
    virtual Eigen::Matrix3d imu_accel_white_noise() const;
    virtual Eigen::Matrix3d imu_gyro_random_walk() const;
    virtual Eigen::Matrix3d imu_accel_random_walk() const;
    virtual int random() const;
    virtual size_t max_keypoint_detection() const;           // 每一帧图像需要检测最大关键点数目
    virtual size_t max_init_raw_frames() const;              // 初始化的时候需要的最大clones数量
    virtual size_t min_init_raw_frames() const;              // 初始化的时候需要的最小clones数量
    virtual size_t min_raw_matches() const;                  // 初始化的时候的最小匹配数
    virtual double min_raw_parallax() const;                 // 初始化的时候的最小视差
    virtual size_t min_raw_triangulation() const;          
    virtual size_t init_map_frames() const;
    virtual size_t min_init_map_landmarks() const;
    virtual bool init_refine_imu() const;
    virtual size_t solver_iteration_limit() const;
    virtual double solver_time_limit() const;
    virtual size_t tracking_window_size() const;
    virtual bool predict_keypoints() const;
};
