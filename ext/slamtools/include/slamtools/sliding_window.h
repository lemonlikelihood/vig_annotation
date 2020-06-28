#pragma once

#include <slamtools/common.h>

class Frame;
class Track;
class Factor;

class SlidingWindow {
    friend class Track;
    struct construct_by_map_t;

  public:
    SlidingWindow();
    virtual ~SlidingWindow();

    void clear();
    
    // 返回窗口内总的clone帧数
    size_t frame_num() const {
        return frames.size();
    }
    
    // 从窗口帧列表中获取指定帧
    Frame *get_frame(size_t id) const {
        return frames[id].get();
    }
    
    // 将新的一帧放入窗口中
    void put_frame(std::unique_ptr<Frame> frame, size_t pos = nil());
    
    // 删除指定帧
    void erase_frame(size_t id);
    
    // 返回track的总数
    size_t track_num() const {
        return tracks.size();
    }
    
    // 返回指定的track
    Track *get_track(size_t id) const {
        return tracks[id].get();
    }
    
    // 创建一个新的track
    Track *create_track();

    // 删除指定的track
    void erase_track(Track *track);
    
    // 按输入条件删除冗余的track
    void prune_tracks(const std::function<bool(const Track *)> &condition);

  private:
    // vector,如果某一个关键点的观测为0，将该关键点从局部地图中删除，直接与vector中的最后一个交换位置，然后修改最后一个的id_in_map,并删除最后一个
    void recycle_track(Track *track);                           

    std::deque<std::unique_ptr<Frame>> frames;                 // 滑动窗口存储图像clone队列
    std::vector<std::unique_ptr<Track>> tracks;                // 滑动窗口中的track数组，track按某个关键点在slidingwindow中的序号排列，这个是总的局部地图信息
};
