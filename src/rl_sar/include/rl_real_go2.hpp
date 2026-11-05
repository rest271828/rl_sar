/*
 * Copyright (c) 2024-2025 Ziqi Fan
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef RL_REAL_GO2_HPP
#define RL_REAL_GO2_HPP

// #define PLOT
// #define CSV_LOGGER
// #define USE_ROS

#include "rl_sdk.hpp"
#include "observation_buffer.hpp"
#include "inference_runtime.hpp"
#include "loop.hpp"
#include "fsm_go2.hpp"
#include "fsm_go2w.hpp"
#include "ep_student.hpp"

#include <unitree/robot/channel/channel_publisher.hpp>
#include <unitree/robot/channel/channel_subscriber.hpp>
#include <unitree/idl/go2/LowState_.hpp>
#include <unitree/idl/go2/LowCmd_.hpp>
#include <unitree/idl/go2/WirelessController_.hpp>
#include <unitree/common/time/time_tool.hpp>
#include <unitree/common/thread/thread.hpp>
#include <unitree/robot/b2/motion_switcher/motion_switcher_client.hpp>
#include <csignal>
#include <atomic>

#if defined(USE_ROS1) && defined(USE_ROS)
#include <ros/ros.h>
#include <geometry_msgs/Twist.h>
#elif defined(USE_ROS2) && defined(USE_ROS)
#include <rclcpp/rclcpp.hpp>
#include <geometry_msgs/msg/twist.hpp>
#endif

#include "matplotlibcpp.h"
namespace plt = matplotlibcpp;

#include <chrono>
#include <thread>
#include <cstring>
#include <ctime>
#include <cstdio>
#include <fstream>
#include <sys/stat.h>

using namespace unitree::common;
using namespace unitree::robot;
using namespace unitree::robot::b2;

// Non-blocking runtime logger (RL_SAR_LOG=1). Control path only enqueues; disk I/O is off-thread.
struct Go2LogSample
{
    uint64_t t_ns = 0;
    int motiontime = 0;
    char fsm[48] = {};
    float q[12] = {};
    float dq[12] = {};
    float q_cmd[12] = {};
    float kp[12] = {};
    float kd[12] = {};
    float gyro[3] = {};
    float quat[4] = {};
    float cmd_x = 0, cmd_y = 0, cmd_yaw = 0;
    int rl_init_done = 0;
    int ep_have_depth = 0;
    int ep_depth_fresh = 0;
    int config_is_ep = 0;
};

class Go2RuntimeLogger
{
public:
    void StartIfEnabled()
    {
        const char *en = std::getenv("RL_SAR_LOG");
        if (!(en && *en && std::string(en) != "0"))
        {
            return;
        }
        enabled_ = true;
        const auto now = std::chrono::system_clock::now();
        const std::time_t t = std::chrono::system_clock::to_time_t(now);
        std::tm tm{};
        localtime_r(&t, &tm);
        char stamp[32];
        std::strftime(stamp, sizeof(stamp), "%Y%m%d_%H%M%S", &tm);
        dir_ = std::string("logs/") + stamp + "_go2_test";
        ::mkdir("logs", 0755);
        ::mkdir(dir_.c_str(), 0755);

        {
            std::ofstream meta(dir_ + "/meta.txt");
            meta << "stamp=" << stamp << "\n";
            meta << "RL_SAR_GO2_CONFIG=" << (std::getenv("RL_SAR_GO2_CONFIG") ? std::getenv("RL_SAR_GO2_CONFIG") : "") << "\n";
            meta << "RL_SAR_EP_DEPTH_HOST=" << (std::getenv("RL_SAR_EP_DEPTH_HOST") ? std::getenv("RL_SAR_EP_DEPTH_HOST") : "") << "\n";
            meta << "note=nonblocking control.csv writer; drops when queue > 4000\n";
        }
        csv_.open(dir_ + "/control.csv", std::ios::out | std::ios::trunc);
        csv_ << "t_ns,motiontime,fsm,"
                "q0,q1,q2,q3,q4,q5,q6,q7,q8,q9,q10,q11,"
                "dq0,dq1,dq2,dq3,dq4,dq5,dq6,dq7,dq8,dq9,dq10,dq11,"
                "qcmd0,qcmd1,qcmd2,qcmd3,qcmd4,qcmd5,qcmd6,qcmd7,qcmd8,qcmd9,qcmd10,qcmd11,"
                "kp0,kp1,kp2,kp3,kp4,kp5,kp6,kp7,kp8,kp9,kp10,kp11,"
                "kd0,kd1,kd2,kd3,kd4,kd5,kd6,kd7,kd8,kd9,kd10,kd11,"
                "gyro_x,gyro_y,gyro_z,qw,qx,qy,qz,"
                "cmd_x,cmd_y,cmd_yaw,rl_init_done,ep_have_depth,ep_depth_fresh,config_is_ep\n";
        run_ = true;
        thr_ = std::thread([this]() { this->WriterLoop(); });
        std::cout << LOGGER::INFO << "[RL_SAR_LOG] writing " << dir_ << "/control.csv" << std::endl;
    }

    void Stop()
    {
        if (!enabled_)
        {
            return;
        }
        run_ = false;
        if (thr_.joinable())
        {
            thr_.join();
        }
        if (csv_.is_open())
        {
            csv_.flush();
            csv_.close();
        }
        std::cout << LOGGER::INFO << "[RL_SAR_LOG] stopped; dropped=" << dropped_.load()
                  << " dir=" << dir_ << std::endl;
        enabled_ = false;
    }

    // Control-thread safe: O(1) enqueue, never blocks on disk.
    void TryPush(const Go2LogSample &s)
    {
        if (!enabled_)
        {
            return;
        }
        if (queued_.load(std::memory_order_relaxed) > 4000)
        {
            dropped_.fetch_add(1, std::memory_order_relaxed);
            return;
        }
        queued_.fetch_add(1, std::memory_order_relaxed);
        queue_.push(s);
    }

    const std::string &dir() const { return dir_; }
    bool enabled() const { return enabled_; }

private:
    void WriterLoop()
    {
        Go2LogSample s;
        while (run_.load() || !queue_.empty())
        {
            if (queue_.try_pop(s))
            {
                queued_.fetch_sub(1, std::memory_order_relaxed);
                if (!csv_.is_open())
                {
                    continue;
                }
                csv_ << s.t_ns << ',' << s.motiontime << ',' << s.fsm;
                for (int i = 0; i < 12; ++i) csv_ << ',' << s.q[i];
                for (int i = 0; i < 12; ++i) csv_ << ',' << s.dq[i];
                for (int i = 0; i < 12; ++i) csv_ << ',' << s.q_cmd[i];
                for (int i = 0; i < 12; ++i) csv_ << ',' << s.kp[i];
                for (int i = 0; i < 12; ++i) csv_ << ',' << s.kd[i];
                csv_ << ',' << s.gyro[0] << ',' << s.gyro[1] << ',' << s.gyro[2]
                     << ',' << s.quat[0] << ',' << s.quat[1] << ',' << s.quat[2] << ',' << s.quat[3]
                     << ',' << s.cmd_x << ',' << s.cmd_y << ',' << s.cmd_yaw
                     << ',' << s.rl_init_done << ',' << s.ep_have_depth << ',' << s.ep_depth_fresh
                     << ',' << s.config_is_ep << '\n';
            }
            else
            {
                std::this_thread::sleep_for(std::chrono::milliseconds(2));
            }
        }
    }

    bool enabled_ = false;
    std::atomic<bool> run_{false};
    std::atomic<int> queued_{0};
    std::atomic<uint64_t> dropped_{0};
    tbb::concurrent_queue<Go2LogSample> queue_;
    std::thread thr_;
    std::ofstream csv_;
    std::string dir_;
};
#define TOPIC_LOWCMD "rt/lowcmd"
#define TOPIC_LOWSTATE "rt/lowstate"
#define TOPIC_JOYSTICK "rt/wirelesscontroller"
constexpr double PosStopF = (2.146E+9f);
constexpr double VelStopF = (16000.0f);

// union for joystick keys
typedef union
{
    struct
    {
        uint8_t R1 : 1;
        uint8_t L1 : 1;
        uint8_t start : 1;
        uint8_t select : 1;
        uint8_t R2 : 1;
        uint8_t L2 : 1;
        uint8_t F1 : 1;
        uint8_t F2 : 1;
        uint8_t A : 1;
        uint8_t B : 1;
        uint8_t X : 1;
        uint8_t Y : 1;
        uint8_t up : 1;
        uint8_t right : 1;
        uint8_t down : 1;
        uint8_t left : 1;
    } components;
    uint16_t value;
} xKeySwitchUnion;

class RL_Real : public RL
{
public:
    RL_Real(int argc, char **argv);
    ~RL_Real();

#if defined(USE_ROS2) && defined(USE_ROS)
    std::shared_ptr<rclcpp::Node> ros2_node;
#endif

private:
    // rl functions
    std::vector<float> Forward() override;
    void GetState(RobotState<float> *state) override;
    void SetCommand(const RobotCommand<float> *command) override;
    void RunModel();
    void RobotControl();

    // EP Student (go2_ep_student)
    void InitEpDepthBridge();
    void MaybeUpdateEpDepthFromBridge(const std::vector<float> &proprio53);
    bool EpTerrainIsFlat() const;
    ep_student::DepthEncoder ep_encoder_;
    ep_student::DepthTcpClient ep_depth_tcp_;
    ep_student::HistState ep_hist_;
    std::vector<float> ep_latent_;
    std::vector<float> ep_yaw_;
    std::vector<float> ep_last_action_;
    std::vector<float> ep_depth_proc_; // cached 58x87
    std::atomic<bool> ep_depth_fresh_{false};
    int ep_policy_step_ = 0;
    bool ep_have_depth_frame_ = false;
    uint64_t ep_last_depth_stamp_ns_ = 0;
    bool ep_offline_depth_ = false; // RL_SAR_EP_DEPTH_BIN loaded
    int ep_stale_encode_logs_ = 0;
    bool dry_run_ = false; // RL_SAR_DRY_RUN=1: no lowcmd / no motion switcher release
    Go2RuntimeLogger runtime_log_;

    // loop
    std::shared_ptr<LoopFunc> loop_keyboard;
    std::shared_ptr<LoopFunc> loop_control;
    std::shared_ptr<LoopFunc> loop_rl;
    std::shared_ptr<LoopFunc> loop_plot;

    // plot
    const int plot_size = 100;
    std::vector<int> plot_t;
    std::vector<std::vector<float>> plot_real_joint_pos, plot_target_joint_pos;
    void Plot();

    // unitree interface
    void InitLowCmd();
    int QueryMotionStatus();
    std::string QueryServiceName(std::string form, std::string name);
    uint32_t Crc32Core(uint32_t *ptr, uint32_t len);
    void LowStateMessageHandler(const void *messages);
    void JoystickHandler(const void *message);
    MotionSwitcherClient msc;
    unitree_go::msg::dds_::LowCmd_ unitree_low_command{};
    unitree_go::msg::dds_::LowState_ unitree_low_state{};
    unitree_go::msg::dds_::WirelessController_ joystick{};
    ChannelPublisherPtr<unitree_go::msg::dds_::LowCmd_> lowcmd_publisher;
    ChannelSubscriberPtr<unitree_go::msg::dds_::LowState_> lowstate_subscriber;
    ChannelSubscriberPtr<unitree_go::msg::dds_::WirelessController_> joystick_subscriber;
    xKeySwitchUnion unitree_joy;

    // others
    std::vector<float> mapped_joint_positions;
    std::vector<float> mapped_joint_velocities;

#if defined(USE_ROS1) && defined(USE_ROS)
    geometry_msgs::Twist cmd_vel;
    ros::Subscriber cmd_vel_subscriber;
    void CmdvelCallback(const geometry_msgs::Twist::ConstPtr &msg);
#elif defined(USE_ROS2) && defined(USE_ROS)
    geometry_msgs::msg::Twist cmd_vel;
    rclcpp::Subscription<geometry_msgs::msg::Twist>::SharedPtr cmd_vel_subscriber;
    void CmdvelCallback(const geometry_msgs::msg::Twist::SharedPtr msg);
#endif
};

#endif // RL_REAL_GO2_HPP
