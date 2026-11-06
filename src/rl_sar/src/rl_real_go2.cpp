/*
 * Copyright (c) 2024-2025 Ziqi Fan
 * SPDX-License-Identifier: Apache-2.0
 */

#include "rl_real_go2.hpp"

#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <mutex>
#include <chrono>
#include <string>

RL_Real::RL_Real(int argc, char **argv)
{
    bool wheel_mode = (argc > 2 && std::string(argv[2]) == "wheel");

    {
        const char *dry = std::getenv("RL_SAR_DRY_RUN");
        this->dry_run_ = (dry && *dry && std::string(dry) != "0");
        if (this->dry_run_)
        {
            std::cout << LOGGER::WARNING
                      << "[DRY_RUN] RL_SAR_DRY_RUN=1: no lowcmd publish, no MotionSwitcher ReleaseMode"
                      << std::endl;
        }
    }

#if defined(USE_ROS1) && defined(USE_ROS)
    ros::NodeHandle nh;
    this->cmd_vel_subscriber = nh.subscribe<geometry_msgs::Twist>("/cmd_vel", 10, &RL_Real::CmdvelCallback, this);
#elif defined(USE_ROS2) && defined(USE_ROS)
    ros2_node = std::make_shared<rclcpp::Node>("rl_real_node");
    this->cmd_vel_subscriber = ros2_node->create_subscription<geometry_msgs::msg::Twist>(
        "/cmd_vel", rclcpp::SystemDefaultsQoS(),
        [this] (const geometry_msgs::msg::Twist::SharedPtr msg) {this->CmdvelCallback(msg);}
    );
#endif

    // read params from yaml
    this->ang_vel_axis = "body";
    this->robot_name = wheel_mode ? "go2w" : "go2";
    this->ReadYaml(this->robot_name, "base.yaml");
    // Freeze GetUp/GetDown targets+gains before any policy InitRL merges EP/himloco yaml.
    this->CacheStandParamsFromBase();

    // auto load FSM by robot_name
    if (FSMManager::GetInstance().IsTypeSupported(this->robot_name))
    {
        auto fsm_ptr = FSMManager::GetInstance().CreateFSM(this->robot_name, this);
        if (fsm_ptr)
        {
            this->fsm = *fsm_ptr;
        }
    }
    else
    {
        std::cout << LOGGER::ERROR << "[FSM] No FSM registered for robot: " << this->robot_name << std::endl;
    }

    // init robot
    this->InitLowCmd();
    this->InitJointNum(this->params.Get<int>("num_of_dofs"));
    this->InitOutputs();
    this->InitControl();
    // create lowcmd publisher
    this->lowcmd_publisher.reset(new ChannelPublisher<unitree_go::msg::dds_::LowCmd_>(TOPIC_LOWCMD));
    this->lowcmd_publisher->InitChannel();
    // create lowstate subscriber
    this->lowstate_subscriber.reset(new ChannelSubscriber<unitree_go::msg::dds_::LowState_>(TOPIC_LOWSTATE));
    this->lowstate_subscriber->InitChannel(std::bind(&RL_Real::LowStateMessageHandler, this, std::placeholders::_1), 1);
    // create joystick subscriber
    this->joystick_subscriber.reset(new ChannelSubscriber<unitree_go::msg::dds_::WirelessController_>(TOPIC_JOYSTICK));
    this->joystick_subscriber->InitChannel(std::bind(&RL_Real::JoystickHandler, this, std::placeholders::_1), 1);
    // init MotionSwitcherClient
    this->msc.SetTimeout(10.0f);
    this->msc.Init();
    // Shut down motion control-related service
    if (!this->dry_run_)
    {
        while (this->QueryMotionStatus())
        {
            std::cout << "Try to deactivate the motion control-related service." << std::endl;
            int32_t ret = this->msc.ReleaseMode();
            if (ret == 0)
            {
                std::cout << "ReleaseMode succeeded." << std::endl;
            }
            else
            {
                std::cout << "ReleaseMode failed. Error code: " << ret << std::endl;
            }
            sleep(1);
        }
    }
    else
    {
        std::cout << LOGGER::WARNING << "[DRY_RUN] skipping MotionSwitcher ReleaseMode loop" << std::endl;
    }

    this->InitEpDepthBridge();
    this->runtime_log_.StartIfEnabled();

    // loop
    this->loop_keyboard = std::make_shared<LoopFunc>("loop_keyboard", 0.05, std::bind(&RL_Real::KeyboardInterface, this));
    this->loop_control = std::make_shared<LoopFunc>("loop_control", this->params.Get<float>("dt"), std::bind(&RL_Real::RobotControl, this));
    this->loop_rl = std::make_shared<LoopFunc>("loop_rl", this->params.Get<float>("dt") * this->params.Get<int>("decimation"), std::bind(&RL_Real::RunModel, this));
    this->loop_keyboard->start();
    this->loop_control->start();
    this->loop_rl->start();

#ifdef PLOT
    this->plot_t = std::vector<int>(this->plot_size, 0);
    this->plot_real_joint_pos.resize(this->params.Get<int>("num_of_dofs"));
    this->plot_target_joint_pos.resize(this->params.Get<int>("num_of_dofs"));
    for (auto &vector : this->plot_real_joint_pos) { vector = std::vector<float>(this->plot_size, 0); }
    for (auto &vector : this->plot_target_joint_pos) { vector = std::vector<float>(this->plot_size, 0); }
    this->loop_plot = std::make_shared<LoopFunc>("loop_plot", 0.002, std::bind(&RL_Real::Plot, this));
    this->loop_plot->start();
#endif
#ifdef CSV_LOGGER
    this->CSVInit(this->robot_name);
#endif
}

RL_Real::~RL_Real()
{
    this->ep_depth_tcp_.Stop();
    if (this->ep_encoder_preload_thr_.joinable())
    {
        this->ep_encoder_preload_thr_.join();
    }
    this->loop_keyboard->shutdown();
    this->loop_control->shutdown();
    this->loop_rl->shutdown();
#ifdef PLOT
    this->loop_plot->shutdown();
#endif
    this->runtime_log_.Stop();
    std::cout << LOGGER::INFO << "RL_Real exit" << std::endl;
}

void RL_Real::InitEpDepthBridge()
{
    this->ep_latent_.assign(ep_student::kLatent, 0.0f);
    this->ep_yaw_.assign(2, 0.0f);
    this->ep_depth_proc_.assign(ep_student::kOutH * ep_student::kOutW, -0.5f);

    // Optional one-shot depth dump for offline verify: RL_SAR_EP_DEPTH_BIN=/path/to/WxH_uint16.mm
    // paired with RL_SAR_EP_DEPTH_WH=848x480
    const char *bin = std::getenv("RL_SAR_EP_DEPTH_BIN");
    if (bin && *bin)
    {
        int w = 848, h = 480;
        if (const char *wh = std::getenv("RL_SAR_EP_DEPTH_WH"))
        {
            int rw = 0, rh = 0;
            if (std::sscanf(wh, "%dx%d", &rw, &rh) == 2 && rw > 0 && rh > 0)
            {
                w = rw;
                h = rh;
            }
        }
        std::ifstream ifs(bin, std::ios::binary);
        if (ifs)
        {
            std::vector<uint16_t> mm(static_cast<size_t>(w) * static_cast<size_t>(h));
            ifs.read(reinterpret_cast<char *>(mm.data()), static_cast<std::streamsize>(mm.size() * sizeof(uint16_t)));
            if (ifs.gcount() == static_cast<std::streamsize>(mm.size() * sizeof(uint16_t)))
            {
                ep_student::PreprocessDepthUint16Mm(mm.data(), h, w, this->ep_depth_proc_);
                this->ep_have_depth_frame_ = true;
                this->ep_depth_fresh_ = true;
                this->ep_offline_depth_ = true;
                this->ep_last_depth_stamp_ns_ = 1; // synthetic non-zero stamp
                this->policy_output_allowed.store(true);
                std::cout << LOGGER::INFO << "[EP_DEPTH] loaded offline bin " << bin
                          << " (" << w << "x" << h << ")" << std::endl;
            }
            else
            {
                std::cout << LOGGER::ERROR << "[EP_DEPTH] short read from " << bin << std::endl;
            }
        }
        else
        {
            std::cout << LOGGER::ERROR << "[EP_DEPTH] cannot open RL_SAR_EP_DEPTH_BIN=" << bin << std::endl;
        }
    }

    // Jetson TCP bridge (preferred when laptop cannot join Jetson ROS2/CycloneDDS domain)
    const char *host = std::getenv("RL_SAR_EP_DEPTH_HOST");
    const char *port_s = std::getenv("RL_SAR_EP_DEPTH_PORT");
    if (host && *host)
    {
        int port = 9876;
        if (port_s && *port_s)
        {
            port = std::atoi(port_s);
        }
        this->ep_depth_tcp_.Start(host, port);
        std::cout << LOGGER::INFO << "[EP_DEPTH] TCP client -> " << host << ":" << port
                  << " (Jetson should run scripts/jetson_depth_tcp_bridge.py)" << std::endl;
    }
    else
    {
        std::cout << LOGGER::WARNING
                  << "[EP_DEPTH] no RL_SAR_EP_DEPTH_HOST; latent stays 0 until TCP/bin provides depth"
                  << std::endl;
    }
    if (this->IsEpStudentConfig() && !this->ep_encoder_preload_thr_.joinable())
    {
        this->ep_encoder_preload_thr_ = std::thread([this]() { this->EnsureEpEncoderLoaded(); });
    }
}

namespace
{
constexpr uint64_t kEpFreshNs = 200ull * 1000000ull;       // 200ms
constexpr uint64_t kEpAllowFirstNs = 300ull * 1000000ull;  // 300ms
constexpr uint64_t kEpHoldGiveUpNs = 2000ull * 1000000ull; // 2s
constexpr uint64_t kEpWarnPeriodNs = 1000000000ull;        // 1Hz
constexpr uint64_t kEpStaleNs = 1000ull * 1000000ull;      // 1s
} // namespace

uint64_t RL_Real::EpMonoNs()
{
    using namespace std::chrono;
    return static_cast<uint64_t>(duration_cast<nanoseconds>(steady_clock::now().time_since_epoch()).count());
}

bool RL_Real::IsEpStudentConfig() const
{
    if (this->config_name == "go2_ep_student")
    {
        return true;
    }
    const char *env_cfg = std::getenv("RL_SAR_GO2_CONFIG");
    return env_cfg && *env_cfg && std::string(env_cfg) == "go2_ep_student";
}

std::string RL_Real::EpVisionWeightPath() const
{
    std::string cfg = this->config_name;
    if (cfg.empty() || cfg == "himloco")
    {
        const char *env_cfg = std::getenv("RL_SAR_GO2_CONFIG");
        if (env_cfg && *env_cfg)
        {
            cfg = env_cfg;
        }
    }
    if (cfg.empty())
    {
        cfg = "go2_ep_student";
    }
    // EPWT .bin only — LibTorch cannot reliably unpickle torch.save(.pt) dicts.
    // IMPORTANT: YamlParams::Get defaults missing keys to T() (empty string), and
    // encoder preload runs before InitRL merges go2_ep_student/config.yaml. An empty
    // name becomes POLICY/.../go2_ep_student/ + SiblingBinPath → ".../.bin".
    std::string weight_name = "vision_weight.bin";
    if (this->params.Has("vision_weight_name"))
    {
        const std::string from_yaml =
            this->params.Get<std::string>("vision_weight_name", weight_name);
        if (!from_yaml.empty())
        {
            weight_name = from_yaml;
        }
    }
    // Basename only — reject accidental empty / "." / ".bin" stems from bad joins.
    const auto slash = weight_name.find_last_of('/');
    if (slash != std::string::npos)
    {
        weight_name = weight_name.substr(slash + 1);
    }
    if (weight_name.empty() || weight_name == "." || weight_name == ".bin" ||
        weight_name == ".pt")
    {
        weight_name = "vision_weight.bin";
    }
    return std::string(POLICY_DIR) + "/" + this->robot_name + "/" + cfg + "/" + weight_name;
}

void RL_Real::EnsureEpEncoderLoaded()
{
    if (this->ep_encoder_.ok)
    {
        return;
    }
    // Mutex (not call_once): early preload can fail when yaml name is still missing;
    // allow a later retry after InitRL merges config.yaml.
    static std::mutex load_mu;
    std::lock_guard<std::mutex> lock(load_mu);
    if (this->ep_encoder_.ok)
    {
        return;
    }
    const std::string path = this->EpVisionWeightPath();
    try
    {
        this->ep_encoder_.EnsureLoaded(path);
        std::cout << LOGGER::INFO << "[EP_DEPTH] loaded vision encoder from " << path
                  << std::endl;
    }
    catch (const std::exception &e)
    {
        std::cerr << "[EP_DEPTH] encoder preload failed (" << path << "): " << e.what()
                  << std::endl;
    }
}

void RL_Real::NoteDepthFrame(uint64_t stamp_ns, uint64_t recv_ns, uint64_t seq)
{
    this->ep_have_depth_frame_ = true;
    this->ep_last_seen_seq_ = seq;
    this->ep_last_frame_recv_ns_ = recv_ns;
    this->ep_last_depth_stamp_ns_ = stamp_ns;
    uint64_t age = 0;
    if (recv_ns == 0)
    {
        this->ep_depth_fresh_ = false;
        return;
    }
    const uint64_t now = EpMonoNs();
    age = now >= recv_ns ? now - recv_ns : 0;
    this->ep_depth_fresh_ = (age < kEpFreshNs);
}

bool RL_Real::DepthRecvAgeNs(uint64_t &age_ns) const
{
    if (!this->ep_have_depth_frame_ || this->ep_last_frame_recv_ns_ == 0)
    {
        return false;
    }
    const uint64_t now = EpMonoNs();
    age_ns = now >= this->ep_last_frame_recv_ns_ ? now - this->ep_last_frame_recv_ns_ : 0;
    return true;
}

void RL_Real::MaybeEncodeIfLoaded(const std::vector<float> &proprio53)
{
    if (!this->ep_encoder_.ok || !this->ep_have_depth_frame_)
    {
        return;
    }
    try
    {
        this->ep_encoder_.Encode(this->ep_depth_proc_, proprio53, this->ep_latent_, this->ep_yaw_);
    }
    catch (const std::exception &e)
    {
        static bool warned = false;
        if (!warned)
        {
            warned = true;
            std::cerr << "[EP_DEPTH] encoder unavailable, latent stays 0: " << e.what() << std::endl;
        }
    }
}

void RL_Real::PollEpDepthWarmup()
{
    if (!this->IsEpStudentConfig() || this->ep_offline_depth_)
    {
        return;
    }
    int w = 0, h = 0;
    uint64_t stamp = 0, recv_ns = 0, seq = 0;
    if (!this->ep_depth_tcp_.PeekLatestMeta(w, h, stamp, recv_ns, seq))
    {
        uint64_t age = 0;
        this->ep_depth_fresh_ = this->DepthRecvAgeNs(age) && age < kEpFreshNs;
        return;
    }
    this->NoteDepthFrame(stamp, recv_ns, seq);
    if (seq == this->ep_last_encoded_seq_)
    {
        return;
    }
    if (!this->ep_encoder_.ok)
    {
        return; // never EnsureLoaded on the 200Hz GetUp path
    }
    std::vector<uint16_t> mm;
    if (!this->ep_depth_tcp_.TryGetLatestMm(mm, w, h, stamp, recv_ns, seq))
    {
        return;
    }
    ep_student::PreprocessDepthUint16Mm(mm.data(), h, w, this->ep_depth_proc_);
    std::vector<float> proprio(ep_student::kNProprio, 0.0f);
    this->MaybeEncodeIfLoaded(proprio);
    this->ep_last_encoded_seq_ = seq;
}

void RL_Real::OnEpLocomotionEnter()
{
    this->ep_rl_enter_mono_ns_ = EpMonoNs();
    this->ep_last_wait_warn_ns_ = 0;
    if (!this->ep_offline_depth_)
    {
        this->policy_output_allowed.store(false);
    }
    else
    {
        this->policy_output_allowed.store(true);
    }
    this->PollEpDepthWarmup();
}

bool RL_Real::TryArmEpFirstForward()
{
    if (this->ep_offline_depth_)
    {
        this->policy_output_allowed.store(true);
        return true;
    }
    if (this->policy_output_allowed.load())
    {
        return true;
    }
    this->PollEpDepthWarmup();
    uint64_t age = 0;
    const bool have = this->DepthRecvAgeNs(age);
    if (have && age < kEpAllowFirstNs)
    {
        this->policy_output_allowed.store(true);
        std::cout << LOGGER::INFO << "[EP_DEPTH] first Forward armed recv_age_ms="
                  << (age / 1000000ull) << " stamp_ns=" << this->ep_last_depth_stamp_ns_
                  << " seq=" << this->ep_last_seen_seq_ << std::endl;
        return true;
    }
    const uint64_t now = EpMonoNs();
    if (this->ep_last_wait_warn_ns_ == 0 || now - this->ep_last_wait_warn_ns_ >= kEpWarnPeriodNs)
    {
        this->ep_last_wait_warn_ns_ = now;
        const uint64_t since_enter =
            this->ep_rl_enter_mono_ns_ == 0 ? 0 : now - this->ep_rl_enter_mono_ns_;
        if (!have)
        {
            std::cout << LOGGER::WARNING
                      << (since_enter > kEpHoldGiveUpNs
                              ? "[EP_DEPTH] waiting for fresh depth before first Forward; holding stand PD (no frame for >2s)"
                              : "[EP_DEPTH] waiting for fresh depth before first Forward; holding stand PD")
                      << std::endl;
        }
        else
        {
            std::cout << LOGGER::WARNING
                      << "[EP_DEPTH] waiting for fresh depth before first Forward; holding stand PD (frame too old)"
                      << std::endl;
        }
    }
    return false;
}

bool RL_Real::EpTerrainIsFlat() const
{
    const char *terr = std::getenv("RL_SAR_EP_TERRAIN");
    if (terr && *terr)
    {
        const std::string t(terr);
        if (t == "parkour")
        {
            return false;
        }
        if (t == "flat")
        {
            return true;
        }
    }
    // Real robot default: parkour one-hot (obs[11]=1), same as step scenes in sim.
    return false;
}

void RL_Real::MaybeUpdateEpDepthFromBridge(const std::vector<float> &proprio53)
{
    std::vector<uint16_t> mm;
    int w = 0, h = 0;
    uint64_t stamp = 0, recv_ns = 0, seq = 0;
    const bool got = this->ep_depth_tcp_.TryGetLatestMm(mm, w, h, stamp, recv_ns, seq);
    if (got)
    {
        this->NoteDepthFrame(stamp, recv_ns, seq);
        if (seq != this->ep_last_encoded_seq_)
        {
            ep_student::PreprocessDepthUint16Mm(mm.data(), h, w, this->ep_depth_proc_);
        }
    }
    else if (this->ep_offline_depth_)
    {
        this->ep_depth_fresh_ = true;
    }
    else
    {
        uint64_t age = 0;
        this->ep_depth_fresh_ = this->DepthRecvAgeNs(age) && age < kEpFreshNs;
    }

    if (!this->ep_have_depth_frame_ && !this->ep_offline_depth_)
    {
        return;
    }

    uint64_t age = 0;
    if (!this->ep_offline_depth_ && this->DepthRecvAgeNs(age) && age > kEpStaleNs)
    {
        const uint64_t now = EpMonoNs();
        if (this->ep_last_stale_warn_ns_ == 0 || now - this->ep_last_stale_warn_ns_ >= kEpWarnPeriodNs)
        {
            this->ep_last_stale_warn_ns_ = now;
            std::cout << LOGGER::WARNING << "[EP_DEPTH] encoding with STALE depth cache"
                      << " last_stamp_ns=" << this->ep_last_depth_stamp_ns_
                      << " recv_age_ms=" << (age / 1000000ull) << std::endl;
        }
    }

    if (!this->ep_offline_depth_ && got && seq == this->ep_last_encoded_seq_)
    {
        return;
    }

    try
    {
        this->EnsureEpEncoderLoaded();
        this->ep_encoder_.Encode(this->ep_depth_proc_, proprio53, this->ep_latent_, this->ep_yaw_);
        if (got)
        {
            this->ep_last_encoded_seq_ = seq;
        }
    }
    catch (const std::exception &e)
    {
        static bool warned = false;
        if (!warned)
        {
            warned = true;
            std::cerr << "[EP_DEPTH] encoder unavailable, latent stays 0: " << e.what() << std::endl;
        }
        if ((int)this->ep_latent_.size() != ep_student::kLatent)
        {
            this->ep_latent_.assign(ep_student::kLatent, 0.0f);
        }
        if ((int)this->ep_yaw_.size() != 2)
        {
            this->ep_yaw_.assign(2, 0.0f);
        }
    }
}

void RL_Real::GetState(RobotState<float> *state)
{
    if (this->unitree_joy.components.A) this->control.SetGamepad(Input::Gamepad::A);
    if (this->unitree_joy.components.B) this->control.SetGamepad(Input::Gamepad::B);
    if (this->unitree_joy.components.X) this->control.SetGamepad(Input::Gamepad::X);
    if (this->unitree_joy.components.Y) this->control.SetGamepad(Input::Gamepad::Y);
    if (this->unitree_joy.components.L1) this->control.SetGamepad(Input::Gamepad::LB);
    if (this->unitree_joy.components.R1) this->control.SetGamepad(Input::Gamepad::RB);
    if (this->unitree_joy.components.F1) this->control.SetGamepad(Input::Gamepad::LStick);
    if (this->unitree_joy.components.F2) this->control.SetGamepad(Input::Gamepad::RStick);
    if (this->unitree_joy.components.up) this->control.SetGamepad(Input::Gamepad::DPadUp);
    if (this->unitree_joy.components.down) this->control.SetGamepad(Input::Gamepad::DPadDown);
    if (this->unitree_joy.components.left) this->control.SetGamepad(Input::Gamepad::DPadLeft);
    if (this->unitree_joy.components.right) this->control.SetGamepad(Input::Gamepad::DPadRight);
    if (this->unitree_joy.components.L1 && this->unitree_joy.components.A) this->control.SetGamepad(Input::Gamepad::LB_A);
    if (this->unitree_joy.components.L1 && this->unitree_joy.components.B) this->control.SetGamepad(Input::Gamepad::LB_B);
    if (this->unitree_joy.components.L1 && this->unitree_joy.components.X) this->control.SetGamepad(Input::Gamepad::LB_X);
    if (this->unitree_joy.components.L1 && this->unitree_joy.components.Y) this->control.SetGamepad(Input::Gamepad::LB_Y);
    if (this->unitree_joy.components.L1 && this->unitree_joy.components.F1) this->control.SetGamepad(Input::Gamepad::LB_LStick);
    if (this->unitree_joy.components.L1 && this->unitree_joy.components.F2) this->control.SetGamepad(Input::Gamepad::LB_RStick);
    if (this->unitree_joy.components.L1 && this->unitree_joy.components.up) this->control.SetGamepad(Input::Gamepad::LB_DPadUp);
    if (this->unitree_joy.components.L1 && this->unitree_joy.components.down) this->control.SetGamepad(Input::Gamepad::LB_DPadDown);
    if (this->unitree_joy.components.L1 && this->unitree_joy.components.left) this->control.SetGamepad(Input::Gamepad::LB_DPadLeft);
    if (this->unitree_joy.components.L1 && this->unitree_joy.components.right) this->control.SetGamepad(Input::Gamepad::LB_DPadRight);
    if (this->unitree_joy.components.R1 && this->unitree_joy.components.A) this->control.SetGamepad(Input::Gamepad::RB_A);
    if (this->unitree_joy.components.R1 && this->unitree_joy.components.B) this->control.SetGamepad(Input::Gamepad::RB_B);
    if (this->unitree_joy.components.R1 && this->unitree_joy.components.X) this->control.SetGamepad(Input::Gamepad::RB_X);
    if (this->unitree_joy.components.R1 && this->unitree_joy.components.Y) this->control.SetGamepad(Input::Gamepad::RB_Y);
    if (this->unitree_joy.components.R1 && this->unitree_joy.components.F1) this->control.SetGamepad(Input::Gamepad::RB_LStick);
    if (this->unitree_joy.components.R1 && this->unitree_joy.components.F2) this->control.SetGamepad(Input::Gamepad::RB_RStick);
    if (this->unitree_joy.components.R1 && this->unitree_joy.components.up) this->control.SetGamepad(Input::Gamepad::RB_DPadUp);
    if (this->unitree_joy.components.R1 && this->unitree_joy.components.down) this->control.SetGamepad(Input::Gamepad::RB_DPadDown);
    if (this->unitree_joy.components.R1 && this->unitree_joy.components.left) this->control.SetGamepad(Input::Gamepad::RB_DPadLeft);
    if (this->unitree_joy.components.R1 && this->unitree_joy.components.right) this->control.SetGamepad(Input::Gamepad::RB_DPadRight);
    if (this->unitree_joy.components.L1 && this->unitree_joy.components.R1) this->control.SetGamepad(Input::Gamepad::LB_RB);

    this->control.x = this->joystick.ly();
    this->control.y = -this->joystick.lx();
    this->control.yaw = -this->joystick.rx();

    state->imu.quaternion[0] = this->unitree_low_state.imu_state().quaternion()[0]; // w
    state->imu.quaternion[1] = this->unitree_low_state.imu_state().quaternion()[1]; // x
    state->imu.quaternion[2] = this->unitree_low_state.imu_state().quaternion()[2]; // y
    state->imu.quaternion[3] = this->unitree_low_state.imu_state().quaternion()[3]; // z

    for (int i = 0; i < 3; ++i)
    {
        state->imu.gyroscope[i] = this->unitree_low_state.imu_state().gyroscope()[i];
    }
    for (int i = 0; i < this->params.Get<int>("num_of_dofs"); ++i)
    {
        state->motor_state.q[i] = this->unitree_low_state.motor_state()[this->params.Get<std::vector<int>>("joint_mapping")[i]].q();
        state->motor_state.dq[i] = this->unitree_low_state.motor_state()[this->params.Get<std::vector<int>>("joint_mapping")[i]].dq();
        state->motor_state.tau_est[i] = this->unitree_low_state.motor_state()[this->params.Get<std::vector<int>>("joint_mapping")[i]].tau_est();
    }
}

void RL_Real::SetCommand(const RobotCommand<float> *command)
{
    if (this->dry_run_)
    {
        return;
    }

    unitree_go::msg::dds_::LowCmd_ dds_low_command;
    dds_low_command.head()[0] = 0xFE;
    dds_low_command.head()[1] = 0xEF;
    dds_low_command.level_flag() = 0xFF;
    dds_low_command.gpio() = 0;

    for (int i = 0; i < 20; ++i)
    {
        dds_low_command.motor_cmd()[i].mode() = 0x01;
        dds_low_command.motor_cmd()[i].q() = PosStopF;
        dds_low_command.motor_cmd()[i].kp() = 0;
        dds_low_command.motor_cmd()[i].dq() = VelStopF;
        dds_low_command.motor_cmd()[i].kd() = 0;
        dds_low_command.motor_cmd()[i].tau() = 0;
    }

    for (int i = 0; i < this->params.Get<int>("num_of_dofs"); ++i)
    {
        dds_low_command.motor_cmd()[this->params.Get<std::vector<int>>("joint_mapping")[i]].mode() = 0x01;
        dds_low_command.motor_cmd()[this->params.Get<std::vector<int>>("joint_mapping")[i]].q() = command->motor_command.q[i];
        dds_low_command.motor_cmd()[this->params.Get<std::vector<int>>("joint_mapping")[i]].dq() = command->motor_command.dq[i];
        dds_low_command.motor_cmd()[this->params.Get<std::vector<int>>("joint_mapping")[i]].kp() = command->motor_command.kp[i];
        dds_low_command.motor_cmd()[this->params.Get<std::vector<int>>("joint_mapping")[i]].kd() = command->motor_command.kd[i];
        dds_low_command.motor_cmd()[this->params.Get<std::vector<int>>("joint_mapping")[i]].tau() = command->motor_command.tau[i];
    }

    dds_low_command.crc() = Crc32Core((uint32_t *)&dds_low_command, (sizeof(unitree_go::msg::dds_::LowCmd_) >> 2) - 1);
    lowcmd_publisher->Write(dds_low_command);

#ifdef PLOT
    this->unitree_low_command = dds_low_command;
#endif
}

void RL_Real::RobotControl()
{
    this->GetState(&this->robot_state);

    this->StateController(&this->robot_state, &this->robot_command);

    if (this->runtime_log_.enabled())
    {
        Go2LogSample s;
        s.t_ns = static_cast<uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::steady_clock::now().time_since_epoch())
                .count());
        s.motiontime = this->motiontime;
        const std::string fsm_name =
            (this->fsm.current_state_) ? this->fsm.current_state_->GetStateName() : "none";
        std::snprintf(s.fsm, sizeof(s.fsm), "%s", fsm_name.c_str());
        const int n = std::min(12, this->params.Get<int>("num_of_dofs"));
        for (int i = 0; i < n; ++i)
        {
            s.q[i] = this->robot_state.motor_state.q[i];
            s.dq[i] = this->robot_state.motor_state.dq[i];
            s.q_cmd[i] = this->robot_command.motor_command.q[i];
            s.kp[i] = this->robot_command.motor_command.kp[i];
            s.kd[i] = this->robot_command.motor_command.kd[i];
        }
        for (int i = 0; i < 3; ++i)
        {
            s.gyro[i] = this->robot_state.imu.gyroscope[i];
        }
        for (int i = 0; i < 4; ++i)
        {
            s.quat[i] = this->robot_state.imu.quaternion[i];
        }
        s.cmd_x = this->control.x;
        s.cmd_y = this->control.y;
        s.cmd_yaw = this->control.yaw;
        s.rl_init_done = this->rl_init_done ? 1 : 0;
        s.ep_have_depth = this->ep_have_depth_frame_ ? 1 : 0;
        s.ep_depth_fresh = this->ep_depth_fresh_.load() ? 1 : 0;
        s.config_is_ep = (this->config_name == "go2_ep_student") ? 1 : 0;
        this->runtime_log_.TryPush(s);
    }

    this->control.ClearInput();

    this->SetCommand(&this->robot_command);
}

void RL_Real::RunModel()
{
    if (this->rl_init_done)
    {
        this->episode_length_buf += 1;
        this->obs.ang_vel = this->robot_state.imu.gyroscope;
        this->obs.commands = {this->control.x, this->control.y, this->control.yaw};
#if !defined(USE_CMAKE) && defined(USE_ROS)
        if (this->control.navigation_mode)
        {
            this->obs.commands = {(float)this->cmd_vel.linear.x, (float)this->cmd_vel.linear.y, (float)this->cmd_vel.angular.z};

        }
#endif
        this->obs.base_quat = this->robot_state.imu.quaternion;
        this->obs.dof_pos = this->robot_state.motor_state.q;
        this->obs.dof_vel = this->robot_state.motor_state.dq;

        this->obs.actions = this->Forward();
        // EP: warm encoder / pull TCP while gated, but do not enqueue policy targets
        // until a recently received depth frame is available (recv_age < 300ms).
        if (this->config_name == "go2_ep_student" && !this->policy_output_allowed.load())
        {
            return;
        }
        this->ComputeOutput(this->obs.actions, this->output_dof_pos, this->output_dof_vel, this->output_dof_tau);

        if (!this->output_dof_pos.empty())
        {
            output_dof_pos_queue.push(this->output_dof_pos);
        }
        if (!this->output_dof_vel.empty())
        {
            output_dof_vel_queue.push(this->output_dof_vel);
        }
        if (!this->output_dof_tau.empty())
        {
            output_dof_tau_queue.push(this->output_dof_tau);
        }

#ifdef CSV_LOGGER
        std::vector<float> tau_est = this->robot_state.motor_state.tau_est;
        this->CSVLogger(this->output_dof_tau, tau_est, this->obs.dof_pos, this->output_dof_pos, this->obs.dof_vel);
#endif
    }
}

std::vector<float> RL_Real::Forward()
{
    std::unique_lock<std::mutex> lock(this->model_mutex, std::try_to_lock);

    // If model is being reinitialized, return previous actions to avoid blocking
    if (!lock.owns_lock())
    {
        std::cout << LOGGER::WARNING << "Model is being reinitialized, using previous actions" << std::endl;
        return this->obs.actions;
    }

    std::vector<float> actions;

    // Extreme Parkour Student: (obs[753], depth_latent[32]) -> action[12]
    if (this->config_name == "go2_ep_student")
    {
        int update_interval = ep_student::kUpdateInterval;
        try
        {
            update_interval = this->params.Get<int>("ep_update_interval");
        }
        catch (...)
        {
        }
        if (update_interval <= 0)
        {
            update_interval = ep_student::kUpdateInterval;
        }

        if (this->episode_length_buf <= 1)
        {
            this->ep_hist_.ready = false;
            this->ep_policy_step_ = 0;
            this->ep_stale_encode_logs_ = 0;
            // Keep GetUp-warmed encoder/latent/recv_ns. Do not snapshot stamp and wait
            // for it to advance — that deadlocks when ROS header.stamp is frozen.
            if (!this->ep_offline_depth_)
            {
                this->policy_output_allowed.store(false);
            }
            else
            {
                this->policy_output_allowed.store(true);
            }
        }

        ep_student::ProprioInput pin;
        pin.ang_vel = this->obs.ang_vel;
        pin.base_quat = this->obs.base_quat;
        pin.cmd_vx = this->control.x;
        pin.terrain_flat = this->EpTerrainIsFlat();
        pin.dof_pos = this->obs.dof_pos;
        pin.dof_vel = this->obs.dof_vel;
        pin.last_action = this->ep_last_action_;
        try
        {
            pin.default_pos = this->params.Get<std::vector<float>>("default_dof_pos");
        }
        catch (...)
        {
        }
        try
        {
            pin.scales.ang_vel = this->params.Get<float>("ang_vel_scale");
        }
        catch (...)
        {
        }
        try
        {
            pin.scales.dof_pos = this->params.Get<float>("dof_pos_scale");
        }
        catch (...)
        {
        }
        try
        {
            pin.scales.dof_vel = this->params.Get<float>("dof_vel_scale");
        }
        catch (...)
        {
        }
        try
        {
            pin.scales.clip_obs = this->params.Get<float>("clip_obs");
        }
        catch (...)
        {
        }

        std::vector<float> ep_obs = ep_student::BuildObservation(pin, this->ep_hist_);
        // Pull depth every step until armed so GetUp→RL does not wait on update_interval.
        const bool need_depth_gate = !this->policy_output_allowed.load();
        if (need_depth_gate || (this->ep_policy_step_ % update_interval == 0))
        {
            std::vector<float> proprio(ep_obs.begin(), ep_obs.begin() + ep_student::kNProprio);
            proprio[6] = 0.0f;
            proprio[7] = 0.0f;
            this->MaybeUpdateEpDepthFromBridge(proprio);
        }
        else
        {
            uint64_t age = 0;
            this->ep_depth_fresh_ = this->DepthRecvAgeNs(age) && age < kEpFreshNs;
        }
        if (this->ep_yaw_.size() >= 2)
        {
            ep_obs[6] = 1.5f * this->ep_yaw_[0];
            ep_obs[7] = 1.5f * this->ep_yaw_[1];
        }
        if ((int)this->ep_latent_.size() != ep_student::kLatent)
        {
            this->ep_latent_.assign(ep_student::kLatent, 0.0f);
        }
        if (!this->TryArmEpFirstForward())
        {
            // Hold: no policy forward yet (FSM keeps stand PD).
            actions.assign(12, 0.0f);
            this->ep_policy_step_ += 1;
            return actions;
        }
        actions = this->model->forward({ep_obs, this->ep_latent_});
        this->ep_policy_step_ += 1;
        if ((int)actions.size() >= 12)
        {
            this->ep_last_action_.assign(actions.begin(), actions.begin() + 12);
        }
    }
    else
    {
        std::vector<float> clamped_obs = this->ComputeObservation();
        if (!this->params.Get<std::vector<int>>("observations_history").empty())
        {
            this->history_obs_buf.insert(clamped_obs);
            this->history_obs = this->history_obs_buf.get_obs_vec(this->params.Get<std::vector<int>>("observations_history"));
            actions = this->model->forward({this->history_obs});
        }
        else
        {
            actions = this->model->forward({clamped_obs});
        }
    }

    if (!this->params.Get<std::vector<float>>("clip_actions_upper").empty() && !this->params.Get<std::vector<float>>("clip_actions_lower").empty())
    {
        return clamp(actions, this->params.Get<std::vector<float>>("clip_actions_lower"), this->params.Get<std::vector<float>>("clip_actions_upper"));
    }
    else
    {
        return actions;
    }
}

void RL_Real::Plot()
{
    this->plot_t.erase(this->plot_t.begin());
    this->plot_t.push_back(this->motiontime);
    plt::cla();
    plt::clf();
    for (int i = 0; i < this->params.Get<int>("num_of_dofs"); ++i)
    {
        this->plot_real_joint_pos[i].erase(this->plot_real_joint_pos[i].begin());
        this->plot_target_joint_pos[i].erase(this->plot_target_joint_pos[i].begin());
        this->plot_real_joint_pos[i].push_back(this->unitree_low_state.motor_state()[i].q());
        this->plot_target_joint_pos[i].push_back(this->unitree_low_command.motor_cmd()[i].q());
        plt::subplot(this->params.Get<int>("num_of_dofs"), 1, i + 1);
        plt::named_plot("_real_joint_pos", this->plot_t, this->plot_real_joint_pos[i], "r");
        plt::named_plot("_target_joint_pos", this->plot_t, this->plot_target_joint_pos[i], "b");
        plt::xlim(this->plot_t.front(), this->plot_t.back());
    }
    // plt::legend();
    plt::pause(0.0001);
}

uint32_t RL_Real::Crc32Core(uint32_t *ptr, uint32_t len)
{
    unsigned int xbit = 0;
    unsigned int data = 0;
    unsigned int CRC32 = 0xFFFFFFFF;
    const unsigned int dwPolynomial = 0x04c11db7;

    for (unsigned int i = 0; i < len; ++i)
    {
        xbit = 1 << 31;
        data = ptr[i];
        for (unsigned int bits = 0; bits < 32; bits++)
        {
            if (CRC32 & 0x80000000)
            {
                CRC32 <<= 1;
                CRC32 ^= dwPolynomial;
            }
            else
            {
                CRC32 <<= 1;
            }

            if (data & xbit)
            {
                CRC32 ^= dwPolynomial;
            }
            xbit >>= 1;
        }
    }

    return CRC32;
}

void RL_Real::InitLowCmd()
{
    this->unitree_low_command.head()[0] = 0xFE;
    this->unitree_low_command.head()[1] = 0xEF;
    this->unitree_low_command.level_flag() = 0xFF;
    this->unitree_low_command.gpio() = 0;

    for (int i = 0; i < 20; ++i)
    {
        this->unitree_low_command.motor_cmd()[i].mode() = (0x01); // motor switch to servo (PMSM) mode
        this->unitree_low_command.motor_cmd()[i].q() = (PosStopF);
        this->unitree_low_command.motor_cmd()[i].kp() = (0);
        this->unitree_low_command.motor_cmd()[i].dq() = (VelStopF);
        this->unitree_low_command.motor_cmd()[i].kd() = (0);
        this->unitree_low_command.motor_cmd()[i].tau() = (0);
    }
}

int RL_Real::QueryMotionStatus()
{
    std::string robotForm, motionName;
    int motionStatus;
    int32_t ret = this->msc.CheckMode(robotForm, motionName);
    if (ret == 0)
    {
        std::cout << "CheckMode succeeded." << std::endl;
    }
    else
    {
        std::cout << "CheckMode failed. Error code: " << ret << std::endl;
    }
    if (motionName.empty())
    {
        std::cout << "The motion control-related service is deactivated." << std::endl;
        motionStatus = 0;
    }
    else
    {
        std::string serviceName = QueryServiceName(robotForm, motionName);
        std::cout << "Service: " << serviceName << " is activate" << std::endl;
        motionStatus = 1;
    }
    return motionStatus;
}

std::string RL_Real::QueryServiceName(std::string form, std::string name)
{
    if (form == "0")
    {
        if (name == "normal" )   return "sport_mode";
        if (name == "ai" )       return "ai_sport";
        if (name == "advanced" ) return "advanced_sport";
    }
    else
    {
        if (name == "ai-w" )     return "wheeled_sport(go2W)";
        if (name == "normal-w" ) return "wheeled_sport(b2W)";
    }
    return "";
}

void RL_Real::LowStateMessageHandler(const void *message)
{
    this->unitree_low_state = *(unitree_go::msg::dds_::LowState_ *)message;
}

void RL_Real::JoystickHandler(const void *message)
{
    joystick = *(unitree_go::msg::dds_::WirelessController_ *)message;
    this->unitree_joy.value = joystick.keys();
}

#if !defined(USE_CMAKE) && defined(USE_ROS)
void RL_Real::CmdvelCallback(
#if defined(USE_ROS1) && defined(USE_ROS)
    const geometry_msgs::Twist::ConstPtr &msg
#elif defined(USE_ROS2) && defined(USE_ROS)
    const geometry_msgs::msg::Twist::SharedPtr msg
#endif
)
{
    this->cmd_vel = *msg;
}
#endif

#if defined(USE_ROS1) && defined(USE_ROS)
void signalHandler(int signum)
{
    ros::shutdown();
    exit(0);
}
#elif defined(USE_CMAKE) || !defined(USE_ROS)
// Signal handler for CMAKE mode
volatile sig_atomic_t g_shutdown_requested = 0;
void signalHandler(int signum)
{
    std::cout << LOGGER::INFO << "Received signal " << signum << ", shutting down..." << std::endl;
    g_shutdown_requested = 1;
}
#endif

int main(int argc, char **argv)
{
    if (argc < 2)
    {
        std::cout << LOGGER::ERROR << "Usage: " << argv[0] << " networkInterface [wheel]" << std::endl;
        throw std::runtime_error("Invalid arguments");
    }
    ChannelFactory::Instance()->Init(0, argv[1]);

#if defined(USE_ROS1) && defined(USE_ROS)
    signal(SIGINT, signalHandler);
    ros::init(argc, argv, "rl_sar");
    RL_Real rl_sar(argc, argv);
    ros::spin();
#elif defined(USE_ROS2) && defined(USE_ROS)
    rclcpp::init(argc, argv);
    auto rl_sar = std::make_shared<RL_Real>(argc, argv);
    rclcpp::spin(rl_sar->ros2_node);
    rclcpp::shutdown();
#elif defined(USE_CMAKE) || !defined(USE_ROS)
    signal(SIGINT, signalHandler);
    RL_Real rl_sar(argc, argv);
    // Offline smoke: RL_SAR_EP_SMOKE_SECONDS=N exits after N seconds without needing Ctrl-C.
    int smoke_s = 0;
    if (const char *smoke = std::getenv("RL_SAR_EP_SMOKE_SECONDS"))
    {
        smoke_s = std::atoi(smoke);
    }
    if (smoke_s > 0)
    {
        for (int i = 0; i < smoke_s && !g_shutdown_requested; ++i)
        {
            sleep(1);
        }
        g_shutdown_requested = 1;
        std::cout << LOGGER::INFO << "[SMOKE] auto-exit after " << smoke_s << "s" << std::endl;
    }
    while (!g_shutdown_requested) { sleep(1); }
    std::cout << LOGGER::INFO << "Exiting..." << std::endl;
#endif

    return 0;
}
