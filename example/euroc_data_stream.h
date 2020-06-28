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

#pragma once

#include <slamtools/common.h>
#include <slamtools/state.h>

class Configurator;

enum DataType {
    DT_IMU,
    DT_IMAGE,
    DT_END
};

struct EurocDataStream {
    EurocDataStream(const std::string &path);

    std::shared_ptr<Configurator> config() const;

    DataType next();
    IMUData read_imu();
    std::shared_ptr<Image> read_image();

    bool has_ground_truth() const;
    PoseState ground_truth_pose(double t) const;

  private:
    std::shared_ptr<Configurator> data_config;                          // 一些参数配置
    Eigen::Vector4d distortion_coeffs;                                  // 畸变参数
    std::deque<std::pair<double, std::string>> image_list;              // 时间戳和对应的图片文件名
    std::deque<IMUData> imu_list;                                       // imu数据deque
    std::map<double, PoseState> gtpose_list;                            // 时间戳和对应的gt位姿 map
