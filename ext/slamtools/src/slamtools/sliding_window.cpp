#include <slamtools/sliding_window.h>
#include <cstdio>
#include <unordered_set>
#include <unordered_map>
#include <slamtools/frame.h>
#include <slamtools/track.h>
#include <slamtools/factor.h>

struct SlidingWindow::construct_by_map_t {};

SlidingWindow::SlidingWindow() = default;
SlidingWindow::~SlidingWindow() = default;

// 清空滑动窗口，删除所有的frames和3D点地图信息
void SlidingWindow::clear() {
    frames.clear();
    tracks.clear();
}

// 在map中加入一个新的地图帧，如果传入指定位置，需要重新维护预积分因子，每一帧图像的预积分因子和前一帧图像有关，加入一帧的时候并未检测是否达到窗口最大值
void SlidingWindow::put_frame(std::unique_ptr<Frame> frame, size_t pos) {
    frame->map = this;
    if (pos == nil()) {
        frames.emplace_back(std::move(frame));
        pos = frames.size() - 1;
    } else {
        frames.emplace(frames.begin() + pos, std::move(frame));
    }
    if (pos > 0) {
        Frame *frame_i = frames[pos - 1].get();
        Frame *frame_j = frames[pos].get();
        frame_j->preintegration_factor = Factor::create_preintegration_error(frame_i, frame_j);
    }
    if (pos < frames.size() - 1) {
        Frame *frame_i = frames[pos].get();
        Frame *frame_j = frames[pos + 1].get();
        frame_j->preintegration_factor = Factor::create_preintegration_error(frame_i, frame_j);
    }
}
 
// 从滑动窗口中删除某一帧clone，同时也要删除这一帧图像在map中的所有观测，并且还要重新维护预积分因子
void SlidingWindow::erase_frame(size_t id) {
    Frame *frame = frames[id].get();
    for (size_t i = 0; i < frame->keypoint_num(); ++i) {
        Track *track = frame->get_track(i);
        if (track != nullptr) {
            track->remove_keypoint(frame);
        }
    }
    frames.erase(frames.begin() + id);
    if (id > 0 && id < frames.size()) {
        Frame *frame_i = frames[id - 1].get();
        Frame *frame_j = frames[id].get();
        frame_j->preintegration_factor = Factor::create_preintegration_error(frame_i, frame_j);
    }
}

// 生成新的track
Track *SlidingWindow::create_track() {
    std::unique_ptr<Track> track = std::make_unique<Track>(construct_by_map_t());
    track->id_in_map = tracks.size();
    track->map = this;
    tracks.emplace_back(std::move(track));
    return tracks.back().get();
}

void SlidingWindow::erase_track(Track *track) {
    while (track->keypoint_map().size() > 0) {
        track->remove_keypoint(track->keypoint_map().begin()->first, false);
    }
    recycle_track(track);    // track 观测为0，从map中删除
}

// 传入一个筛选函数，剔除冗余的track
void SlidingWindow::prune_tracks(const std::function<bool(const Track *)> &condition) {
    std::vector<Track *> tracks_to_prune;
    for (size_t i = 0; i < track_num(); ++i) {
        Track *track = get_track(i);
        if (condition(track)) {
            tracks_to_prune.push_back(track);
        }
    }
    for (Track *t : tracks_to_prune) {
        erase_track(t);
    }
}

// vector,如果某一个关键点的观测为0，将该关键点从局部地图中删除，直接与vector中的最后一个交换位置，然后修改最后一个的id_in_map,并删除最后一个
void SlidingWindow::recycle_track(Track *track) {
    if (track->id_in_map != tracks.back()->id_in_map) {
        tracks[track->id_in_map].swap(tracks.back());
        tracks[track->id_in_map]->id_in_map = track->id_in_map;
    }
    tracks.pop_back();
}
