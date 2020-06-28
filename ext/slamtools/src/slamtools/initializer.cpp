#include <slamtools/initializer.h>
#include <slamtools/stereo.h>
#include <slamtools/lie_algebra.h>
#include <slamtools/frame.h>
#include <slamtools/track.h>
#include <slamtools/sliding_window.h>
#include <slamtools/pnp.h>
#include <slamtools/bundle_adjustor.h>
#include <slamtools/configurator.h>
#include <slamtools/homography.h>
#include <slamtools/essential.h>

using namespace Eigen;

Initializer::Initializer(std::shared_ptr<Configurator> config) :
    config(config) {
    raw = std::make_unique<SlidingWindow>();
}

Initializer::~Initializer() = default;

// append_frame 会进行IMU预积分和tracking,更新窗口内的frames和tracks信息
void Initializer::append_frame(std::unique_ptr<Frame> frame) {
    if (raw->frame_num() > 0) {                                     // 如果滑动窗口里有其他帧，先获取最后一帧，然后和当前帧做预积分
        Frame *last_frame = raw->get_frame(raw->frame_num() - 1);   // imu数据已经加载到当前帧的preintegration里面
        frame->preintegration.integrate(frame->image->t, last_frame->motion.bg, last_frame->motion.ba, true, false);
        last_frame->track_keypoints(frame.get(), config.get());    //  对上一帧图像进行track,track前next_frame没有关键点，track后成功的点会加入下一帧的关键点，然后重新检测一些新的关键点
    }
    frame->detect_keypoints(config.get());                         // 对于每一帧都要检测关键点，这里只是检测新的关键点，没有进行跟踪

    raw->put_frame(std::move(frame));                              // 将该帧放入滑动窗口
    while (raw->frame_num() > config->max_init_raw_frames()) {     // 如果滑动窗口中的帧数大于初始化指定帧数，删除最老帧
        raw->erase_frame(0);
    }
}

//  sfm 视觉初始化 保证滑动窗口中的帧数已经大于初始化要求的最小帧数，所有图像帧已经完成了IMU预积分，特征点检测跟踪，生成了track信息
std::unique_ptr<SlidingWindow> Initializer::init_sfm() const {
    // [1] find a proper pair for initialization
    Frame *init_frame_i = nullptr;
    Frame *init_frame_j = raw->get_frame(raw->frame_num() - 1);
    size_t init_frame_i_id = nil();                                // nil() 表示-1，未找到
    std::vector<Vector3d> init_points;                             // 初始化的3D点
    std::vector<std::pair<size_t, size_t>> init_matches;           // 两帧关键点匹配的index
    std::vector<char> init_point_status;                           // 匹配的状态
    Matrix3d init_R;                                               // 初始的R
    Vector3d init_T;                                               // 初始的t

    Frame *frame_j = init_frame_j;                                 // j表示最后一帧
    std::vector<Vector2d> frame_i_keypoints;                       // 第i帧的关键点
    std::vector<Vector2d> frame_j_keypoints;                       // 第j帧的关键点
    for (size_t frame_i_id = 0; frame_i_id + config->min_init_raw_frames() < raw->frame_num(); ++frame_i_id) {
        double total_parallax = 0;
        int common_track_num = 0;
        Frame *frame_i = raw->get_frame(frame_i_id);             
        frame_i_keypoints.clear();
        frame_j_keypoints.clear();
        init_matches.clear();
        for (size_t ki = 0; ki < frame_i->keypoint_num(); ++ki) {
            Track *track = frame_i->get_track(ki);                  // 获取第帧图像点的track信息
            if (!track) continue;                                   // 如果track为空，直接跳过
            size_t kj = track->get_keypoint_id(init_frame_j);       // 检测是否在第j帧成功检测到
            if (kj == nil()) continue;
            frame_i_keypoints.push_back(frame_i->get_keypoint(ki));  // 如果成功匹配，放入队列，归一化平面坐标
            frame_j_keypoints.push_back(frame_j->get_keypoint(kj));
            init_matches.emplace_back(ki, kj);
            total_parallax += (apply_k(frame_i->get_keypoint(ki), frame_i->K) - apply_k(frame_j->get_keypoint(kj), frame_j->K)).norm(); // 计算视差，图像平面上像素之间的距离
            common_track_num++;                                     // 成功匹配的个数
        }
        
        // 保证匹配数量足够大，匹配到的关键点视差比较大，便于三角化
        if (common_track_num < (int)config->min_raw_matches()) continue;      // 如果当前帧匹配个数小于数量阈值，跳到下一帧
        total_parallax /= std::max(common_track_num, 1);
        if (total_parallax < config->min_raw_parallax()) continue;            // 如果匹配的平均视差小于视差阈值，跳到下一帧

        std::vector<Matrix3d> Rs;
        std::vector<Vector3d> Ts;

        Matrix3d RH1, RH2;
        Vector3d TH1, TH2, nH1, nH2;
        Matrix3d H = find_homography_matrix(frame_i_keypoints, frame_j_keypoints, 0.7 / frame_i->K(0, 0), 0.999, 1000, config->random());
        if (!decompose_homography(H, RH1, RH2, TH1, TH2, nH1, nH2)) {          // 计算单应矩阵，如果是纯旋转，跳到下一帧
            continue; // is pure rotation
        }
        TH1 = TH1.normalized();
        TH2 = TH2.normalized();
        Rs.insert(Rs.end(), {RH1, RH1, RH2, RH2});
        Ts.insert(Ts.end(), {TH1, -TH1, TH2, -TH2});

        Matrix3d RE1, RE2;
        Vector3d TE;
        Matrix3d E = find_essential_matrix(frame_i_keypoints, frame_j_keypoints, 0.7 / frame_i->K(0, 0), 0.999, 1000, config->random());
        decompose_essential(E, RE1, RE2, TE);
        TE = TE.normalized();
        Rs.insert(Rs.end(), {RE1, RE1, RE2, RE2});
        Ts.insert(Ts.end(), {TE, -TE, TE, -TE});
        
        // 对单应矩阵和本质矩阵分解得到的8组R，t分别三角化求最好的三角化个数和最小的重投影误差，最小的重投影误差优先级别高
        size_t triangulated_num = triangulate_from_rt_scored(frame_i_keypoints, frame_j_keypoints, Rs, Ts, config->min_raw_triangulation(), init_points, init_R, init_T, init_point_status);

        if (triangulated_num < config->min_raw_triangulation()) {         // 如果三角化最好的个数不满足阈值，跳到下一帧
            continue;
        }

        init_frame_i = frame_i;                                          // 如果三角化成功，找到该帧作为初始帧
        init_frame_i_id = frame_i_id;
        break;
    }

    if (!init_frame_i) return nullptr;                                   // 如果没有找到初始帧，三角化失败，直接返回

    // [2] create sfm map

    // [2.1] enumerate keyframe ids                                     // 如果三角化成功，构建sfm
    std::vector<size_t> init_keyframe_ids;
    size_t init_map_frames = config->init_map_frames();                 // 8
    double keyframe_id_gap = (double)(raw->frame_num() - 1 - init_frame_i_id) / (double)(init_map_frames - 1);
    for (size_t i = 0; i < init_map_frames; ++i) {
        init_keyframe_ids.push_back((size_t)round(init_frame_i_id + keyframe_id_gap * i));
    }                                                                  // 从当前帧到最后一帧均匀找出用来初始化的帧

    // [2.2] make a clone of submap using keyframe ids
    std::unique_ptr<SlidingWindow> map = std::make_unique<SlidingWindow>();
    for (size_t i = 0; i < init_keyframe_ids.size(); ++i) {
        map->put_frame(raw->get_frame(init_keyframe_ids[i])->clone());   // 构建一个子地图和滑窗，这里只复制了关键点信息
    }
    for (size_t j = 1; j < init_keyframe_ids.size(); ++j) {              // 这里重新构建track信息和预积分信息
        Frame *old_frame_i = raw->get_frame(init_keyframe_ids[j - 1]);
        Frame *old_frame_j = raw->get_frame(init_keyframe_ids[j]);
        Frame *new_frame_i = map->get_frame(j - 1);
        Frame *new_frame_j = map->get_frame(j);
        for (size_t ki = 0; ki < old_frame_i->keypoint_num(); ++ki) {
            Track *track = old_frame_i->get_track(ki);
            if (track == nullptr) continue;
            size_t kj = track->get_keypoint_id(old_frame_j);
            if (kj == nil()) continue;
            new_frame_i->get_track(ki, create_if_empty)->add_keypoint(new_frame_j, kj);
        }
        new_frame_j->preintegration.data.clear();
        for (size_t f = init_keyframe_ids[j - 1]; f < init_keyframe_ids[j]; ++f) {
            Frame *old_frame = raw->get_frame(f + 1);
            std::vector<IMUData> &old_data = old_frame->preintegration.data;
            std::vector<IMUData> &new_data = new_frame_j->preintegration.data;
            new_data.insert(new_data.end(), old_data.begin(), old_data.end());
        }
    }

    Frame *new_init_frame_i = map->get_frame(0);                           // 新地图的第一帧
    Frame *new_init_frame_j = map->get_frame(map->frame_num() - 1);        // 新地图的最后一帧ß

    // [2.3] set init states
    PoseState pose;                                                        // 设置第一帧和最后一帧的camera位姿
    pose.q.setIdentity();
    pose.p.setZero();
    new_init_frame_i->set_pose(new_init_frame_i->camera, pose);
    pose.q = init_R.transpose();
    pose.p = -(init_R.transpose() * init_T);
    new_init_frame_j->set_pose(new_init_frame_j->camera, pose);

    for (size_t k = 0; k < init_points.size(); ++k) {                    // 对于成功三角化的点，设置第一帧作为参考帧，也就是作为世界坐标系，固定不变
        if (init_point_status[k] == 0) continue;
        Track *track = new_init_frame_i->get_track(init_matches[k].first);
        track->landmark.x = init_points[k];                             // 3D点信息与track绑定，就是与每一帧绑定
        track->landmark.flag(LF_VALID) = true;
    }

    // [2.4] solve other frames via pnp                                // pnp 通过前一帧的位姿和初始化的3D点来求解后一帧的位姿
    for (size_t j = 1; j + 1 < map->frame_num(); ++j) {                // 这个过程不会增加新的3D点，也不会改变初始化3D点的坐标
        Frame *frame_i = map->get_frame(j - 1);
        Frame *frame_j = map->get_frame(j);
        frame_j->set_pose(frame_j->camera, frame_i->get_pose(frame_i->camera)); // 后一帧的位姿初始值为前一帧
        visual_inertial_pnp(map.get(), frame_j, false);               // ceres求解位姿
    }

    // [2.5] triangulate more points
    for (size_t i = 0; i < map->track_num(); ++i) {                   // 对map中其他的track也进行三角化，增加更多的3D点
        Track *track = map->get_track(i);
        if (track->landmark.flag(LF_VALID)) continue;                 // 如果是初始三角化的点，不可以改变
        track->triangulate();                                         // 给track设置landmark值，并固定不变
    } 

    // [3] sfm

    // [3.1] bundle adjustment                                        // sfm固定第一帧的位姿，固定初始化的3D点
    map->get_frame(0)->pose.flag(PF_FIXED) = true;                    // 利用所有的观测和相机位姿最小化重投影误差估计，优化其中非固定的变量，包括相机位姿和3D点
    if (!BundleAdjustor().solve(map.get(), false, config->solver_iteration_limit() * 5, config->solver_time_limit())) {
        return nullptr;
    }

    // [3.2] cleanup invalid points                                   //  从map中清除没有成功三角化的track，或者三角化误差（平均重投影误差）大于1的track
    map->prune_tracks([](const Track *track) {                        
        return !track->landmark.flag(LF_VALID) || track->landmark.quality > 1.0;   
    });

    return map;
}
