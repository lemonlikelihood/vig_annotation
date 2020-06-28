#include <slamtools/track.h>
#include <slamtools/frame.h>
#include <slamtools/sliding_window.h>
#include <slamtools/factor.h>
#include <slamtools/stereo.h>

using namespace Eigen;

Track::Track() = default;
Track::~Track() = default;

// 返回该track在指定frame中的2D坐标，keypoint_refs.at(frame)返回该3D在指定frame中对应的关键点序号，frame->get_keypoin（），输入序号返回2D点坐标
const Vector2d &Track::get_keypoint(Frame *frame) const {
    return frame->get_keypoint(keypoint_refs.at(frame));
}

// 加入一个新的观测
void Track::add_keypoint(Frame *frame, size_t keypoint_id) {
    frame->tracks[keypoint_id] = this;                 //   frame里面的track按该帧clone里面的关键点排序         
    frame->reprojection_factors[keypoint_id] = Factor::create_reprojection_error(frame, keypoint_id);
    keypoint_refs[frame] = keypoint_id;
}

// 删除一个观测，如果某一个观测为0，需要从地图中删除
void Track::remove_keypoint(Frame *frame, bool suicide_if_empty) {
    size_t keypoint_id = keypoint_refs.at(frame);
    frame->tracks[keypoint_id] = nullptr;
    frame->reprojection_factors[keypoint_id].reset();
    keypoint_refs.erase(frame);
    if (suicide_if_empty && keypoint_refs.size() == 0) {
        map->recycle_track(this);
    }
}


bool Track::triangulate() {
    std::vector<Matrix<double, 3, 4>> Ps;
    std::vector<Vector2d> ps;
    for (const auto &t : keypoint_map()) {               // keypoint_map() 该3D点对应的所有观测
        Matrix<double, 3, 4> P;
        Matrix3d R;
        Vector3d T;
        PoseState pose = t.first->get_pose(t.first->camera);   // t.first frame
        R = pose.q.conjugate().matrix();
        T = -(R * pose.p);
        P << R, T;
        Ps.push_back(P);                                 // 存的是每一帧的位姿
        ps.push_back(t.first->get_keypoint(t.second));   // 存的是每一帧的对应的2D坐标
    }
    Vector3d p;
    if (triangulate_point_checked(Ps, ps, p)) {          // 利用R，t 和 2D 坐标进行三角化，不会改变R，t，只计算3D点坐标
        landmark.x = p;
        landmark.flag(LF_VALID) = true;                  // 成功三角化true，固定不变，track设置landmark值
    } else {
        landmark.flag(LF_VALID) = false;                 // 否则false
    }
    return landmark.flag(LF_VALID);
}
