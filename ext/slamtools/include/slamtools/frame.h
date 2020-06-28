#pragma once

#include <slamtools/common.h>
#include <slamtools/state.h>
#include <slamtools/preintegrator.h>

class Track;
class SlidingWindow;
class Factor;
class Configurator;

struct create_if_empty_t {};
extern create_if_empty_t create_if_empty;

class Frame {
    friend class Track;
    friend class SlidingWindow;
    SlidingWindow *map;                                                                    // 每一帧图像所属的slidingWindow

  public:
    Frame();
    virtual ~Frame();

    std::unique_ptr<Frame> clone() const;                                                 // 返回唯一指针

    size_t keypoint_num() const {                                                         // 返回该帧图像的关键点个数
        return keypoints.size();
    }

    const Eigen::Vector2d &get_keypoint(size_t keypoint_id) const {                      // 获得序号为keypoint_id的关键点坐标（图像坐标系）
        return keypoints[keypoint_id];
    }

    Track *get_track(size_t keypoint_id) const {                                         // 获得该关键点所对应的track，不传引用不会创建新的
        return tracks[keypoint_id];
    }

    Track *get_track(size_t keypoint_id, const create_if_empty_t &);                     // 获得该关键点所对应的track,如果没有可以创建新的track

    Factor *get_reprojection_factor(size_t keypoint_id) {
        return reprojection_factors[keypoint_id].get();
    }

    Factor *get_preintegration_factor() {
        return preintegration_factor.get();
    }

    void detect_keypoints(Configurator *config);                                        // 按config要求来检测关键点
    void track_keypoints(Frame *next_frame, Configurator *config);                      // 按config要求来跟踪关键点
    void detect_segments(size_t max_segments = 0);                                      // 检测边缘信息

    PoseState get_pose(const ExtrinsicParams &sensor) const;                            // 按传感器类型返回传感器相对于世界坐标系的位姿
    void set_pose(const ExtrinsicParams &sensor, const PoseState &pose);                // 按传感器类型设置传感器相对于世界坐标系的位姿

    Eigen::Matrix3d K;                                                                               // 相机内参
    Eigen::Matrix2d sqrt_inv_cov;                                                                    
    std::shared_ptr<Image> image;                                                                    // 每一帧所对应的图像数据

    PoseState pose;                                                                                  // 位姿状态p,q,flag
    MotionState motion;                                                                              // 运动姿态ba,bg,v
    ExtrinsicParams camera;                                                                          // p_cs , q_cs
    ExtrinsicParams imu;                                                                             // p_is , q_is

    PreIntegrator preintegration;                                                                    // 每一帧图像都有一个预积分器

    Eigen::Vector3d external_gravity;                                                                // 外部重力
    std::vector<std::tuple<Eigen::Vector2d, Eigen::Vector2d>> segments;                              // 垂直边信息，以两个端点信息表示

  private:
    std::vector<Eigen::Vector2d> keypoints;                                                          // 关键点坐标，归一化坐标系，按关键点序号存储
    std::vector<Track *> tracks；               // 图像关键点所对应的tracks集合，每一个track都是map<frame,size_t>的组合，表示该关键点对应某一个关键帧的第几个关键点
    std::vector<std::unique_ptr<Factor>> reprojection_factors;                                       // 重投影因子
    std::unique_ptr<Factor> preintegration_factor;                                                   // 预积分因子
};

template <>
struct Comparator<Frame> {                                                                          // frame 按时间戳排序
    constexpr bool operator()(const Frame &frame_i, const Frame &frame_j) const {                   // 对图像作比较返回时间戳较小的
        return Comparator<Image *>()(frame_i.image.get(), frame_j.image.get());
    }
};
