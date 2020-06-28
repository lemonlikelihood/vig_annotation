#include <slamtools/frame.h>
#include <slamtools/track.h>
#include <slamtools/sliding_window.h>
#include <slamtools/factor.h>
#include <slamtools/stereo.h>
#include <slamtools/configurator.h>

using namespace Eigen;

create_if_empty_t create_if_empty{};

Frame::Frame() :
    map(nullptr) {
}

Frame::~Frame() = default;

std::unique_ptr<Frame> Frame::clone() const {
    std::unique_ptr<Frame> frame = std::make_unique<Frame>();
    frame->K = K;
    frame->sqrt_inv_cov = sqrt_inv_cov;
    frame->image = image;
    frame->pose = pose;
    frame->motion = motion;
    frame->camera = camera;
    frame->imu = imu;
    frame->preintegration = preintegration;
    frame->external_gravity = external_gravity; // maybe inappropriate?
    frame->keypoints = keypoints;
    frame->segments = segments;
    frame->tracks = std::vector<Track *>(keypoints.size(), nullptr);
    frame->reprojection_factors = std::vector<std::unique_ptr<Factor>>(keypoints.size());
    frame->map = nullptr;
    return frame;
}

// 返回当前帧中指定的序号所对应的track，如果track不存在，创建新的track
Track *Frame::get_track(size_t keypoint_id, const create_if_empty_t &) {
    if (tracks[keypoint_id] == nullptr) {
        assert(("for get_track(..., create_if_empty) to work, frame->map cannot be nullptr") && (map != nullptr));
        Track *track = map->create_track();
        track->add_keypoint(this, keypoint_id);
    }
    return tracks[keypoint_id];
}

void Frame::detect_keypoints(Configurator *config) {
    std::vector<Vector2d> pkeypoints(keypoints.size());
    for (size_t i = 0; i < keypoints.size(); ++i) {
        pkeypoints[i] = apply_k(keypoints[i], K);             // 从归一化平面到图像坐标系
    }
    image->detect_keypoints(pkeypoints, config->max_keypoint_detection());
    size_t old_keypoint_num = keypoints.size();
    keypoints.resize(pkeypoints.size());                     // 在进行detect检测特征点的时候关键点2D坐标已经有了，但是track信息全部置nullptr
    tracks.resize(pkeypoints.size(), nullptr);              
    reprojection_factors.resize(pkeypoints.size());
    for (size_t i = old_keypoint_num; i < pkeypoints.size(); ++i) {      // keypoints 前面old_keypoint_num个坐标点仍然在归一化平面坐标系
        keypoints[i] = remove_k(pkeypoints[i], K);          // 将新检测到的关键点从图像坐标系转换到归一化平面
    }                                        
}

void Frame::track_keypoints(Frame *next_frame, Configurator *config) {
    std::vector<Vector2d> curr_keypoints(keypoints.size());
    std::vector<Vector2d> next_keypoints;

    for (size_t i = 0; i < keypoints.size(); ++i) {
        curr_keypoints[i] = apply_k(keypoints[i], K);      // 从归一化平面到图像坐标系
    }

    if (config->predict_keypoints()) {                     // 利用IMU的q来提供一个初值进行预测
        Quaterniond delta_key_q = (camera.q_cs.conjugate() * imu.q_cs * next_frame->preintegration.delta.q * next_frame->imu.q_cs.conjugate() * next_frame->camera.q_cs).conjugate();
        next_keypoints.resize(curr_keypoints.size());
        for (size_t i = 0; i < keypoints.size(); ++i) {
            next_keypoints[i] = apply_k((delta_key_q * keypoints[i].homogeneous()).hnormalized(), next_frame->K);
        }
    }

    std::vector<char> status;
    image->track_keypoints(next_frame->image.get(), curr_keypoints, next_keypoints, status);      // opencv 光流法跟踪 ，跟踪的是图像坐标系的点，对当前帧跟踪的结果放入next_keypoints中

    for (size_t curr_keypoint_id = 0; curr_keypoint_id < curr_keypoints.size(); ++curr_keypoint_id) {
        if (status[curr_keypoint_id]) {                // 如果当前帧的某一个关键点被成功跟踪上，将跟踪上的关键点信息放入下一帧         
            size_t next_keypoint_id = next_frame->keypoints.size();   // 初始的时候这里size()为0
            next_frame->keypoints.emplace_back(remove_k(next_keypoints[curr_keypoint_id], next_frame->K)); // 归一化平面坐标
            next_frame->tracks.emplace_back(nullptr);
            next_frame->reprojection_factors.emplace_back(nullptr);
            get_track(curr_keypoint_id, create_if_empty)->add_keypoint(next_frame, next_keypoint_id);  // create_if_empty 保证如果是一个新的跟踪点，回生成一个新的track
        }
    }
}

void Frame::detect_segments(size_t max_segments) {
    image->detect_segments(segments, max_segments);
    for (size_t i = 0; i < segments.size(); ++i) {
        std::get<0>(segments[i]) = remove_k(std::get<0>(segments[i]), K);
        std::get<1>(segments[i]) = remove_k(std::get<1>(segments[i]), K);
    }
}

PoseState Frame::get_pose(const ExtrinsicParams &sensor) const {
    PoseState result;
    result.q = pose.q * sensor.q_cs;
    result.p = pose.p + pose.q * sensor.p_cs;
    return result;
}

void Frame::set_pose(const ExtrinsicParams &sensor, const PoseState &pose) {
    this->pose.q = pose.q * sensor.q_cs.conjugate();        // q_b_to_w = q_I_to_w * (q_I_to_b的转置)
    this->pose.p = pose.p - this->pose.q * sensor.p_cs;
}
