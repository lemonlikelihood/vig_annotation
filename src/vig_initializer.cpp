/**************************************************************************
* VIG-Init
*
* Copyright SenseTime. All Rights Reserved.
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
**************************************************************************/

#include "vig_initializer.h"
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
#include "imu_initializer.h"

using namespace Eigen;

VigInitializer::VigInitializer(std::shared_ptr<Configurator> config) :
    Initializer(config) {
}

VigInitializer::~VigInitializer() = default;

void VigInitializer::append_frame(std::unique_ptr<Frame> frame) {
    frame->detect_segments();                               // 直线段检测
    Initializer::append_frame(std::move(frame));
}

std::unique_ptr<SlidingWindow> VigInitializer::init() const {
    if (raw->frame_num() < config->min_init_raw_frames()) return nullptr;    // 如果滑动窗口中帧数不够，不进行初始化，直接返回nullptr

    std::unique_ptr<SlidingWindow> map;

    if (!(map = init_sfm())) return nullptr;                                 // 进行sfm纯视觉初始化，map是raw的子窗口

    if (!init_imu(map.get())) return nullptr;                                // IMU视觉惯导联合初始化

    map->get_frame(0)->pose.flag(PF_FIXED) = true;                           // 初始化完成，固定第一帧的body系到世界坐标系的变换R，t
    // BundleAdjustor().solve(map.get(), true, config->solver_iteration_limit(), config->solver_time_limit());

    return map;                                                              // 返回成功初始化的子地图
}

bool VigInitializer::init_imu(SlidingWindow *map) const {
    return ImuInitializer(map, config).init();
}
