#pragma once

#include <slamtools/common.h>
#include <slamtools/sliding_window.h>
#include <slamtools/state.h>
#include <slamtools/frame.h>

class Frame;
class SlidingWindow;

class Track {
    friend class SlidingWindow;
    size_t id_in_map;                // track在slidingWindow中对应的序号，唯一标志一个3D点
    SlidingWindow *map;              // track所属于的滑动窗口
    Track();

  public:
    Track(const SlidingWindow::construct_by_map_t &) :
        Track() {
    }

    virtual ~Track();
    
    // 返回该3D点的所有观测
    const std::map<Frame *, size_t, Comparator<Frame *>> &keypoint_map() const {
        return keypoint_refs;
    }
    
    // 检测某一帧图像是否包含该3D点，也就是该关键帧是否在map中
    bool has_keypoint(Frame *frame) const {
        return keypoint_refs.count(frame) > 0;
    }

    // 返回该3D点在某一个关键帧的序号
    size_t get_keypoint_id(Frame *frame) const {
        if (has_keypoint(frame)) {
            return keypoint_refs.at(frame);
        } else {
            return nil();
        }
    }
    
    // 返回该3D点在某一帧中的2D坐标
    const Eigen::Vector2d &get_keypoint(Frame *frame) const;
    
    // 将该track 3D点新的观测<frame,size_t>加入track
    void add_keypoint(Frame *frame, size_t keypoint_id);

    // 将该track 3D点某一个观测frame从map中剔除
    void remove_keypoint(Frame *frame, bool suicide_if_empty = true);
   
    // 该3D点所有的观测是否被三角化
    bool triangulate();
    

    LandmarkState landmark;                                              // 该track所对应的3D点坐标

  private:
    std::map<Frame *, size_t, Comparator<Frame *>> keypoint_refs;        // track 表示某一个关键点是某一个关键帧的第几个关键点    
};

// This comparator is incorrect in theory, consider if a and b are modified...
// Hence, don't use this in production.
template <>
struct Comparator<Track> {
    bool operator()(const Track &a, const Track &b) const {
        auto ia = a.keypoint_map().cbegin();
        auto ib = b.keypoint_map().cbegin();
        while (ia != a.keypoint_map().cend() && ib != b.keypoint_map().cend()) {
            bool eab = Comparator<Frame *>()(ia->first, ib->first);
            bool eba = Comparator<Frame *>()(ib->first, ia->first);
            if (eab || eba) {
                return eab;
            } else if (ia->second != ib->second) {
                return ia->second < ib->second;
            } else {
                ia++;
                ib++;
            }
        }
        return ia != a.keypoint_map().cend();
    }
};
