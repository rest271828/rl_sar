/*
 * Copyright (c) 2024-2025 Ziqi Fan
 * SPDX-License-Identifier: Apache-2.0
 */

#include "rl_sim_mujoco.hpp"
#include "vector_math.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>

RL_Sim* RL_Sim::instance = nullptr;

namespace
{
constexpr double kPitchDeg = 22.9;
constexpr double kCamBody[3] = {0.24, -0.0175, 0.12};
constexpr double kHfovDeg = 87.0;
constexpr double kFar = 2.0;

struct PlaneRays
{
    std::vector<double> dirs_cam;
    std::vector<double> plane_scale;
    bool ready = false;
};

PlaneRays &Rays()
{
    static PlaneRays rays;
    if (rays.ready)
    {
        return rays;
    }
    const int n = ep_student::kDepthH * ep_student::kDepthW;
    rays.dirs_cam.resize(n * 3);
    rays.plane_scale.resize(n);
    const double hfov = kHfovDeg * 3.14159265358979323846 / 180.0;
    const double fx = 0.5 * ep_student::kDepthW / std::tan(0.5 * hfov);
    const double fy = fx;
    const double cx = 0.5 * (ep_student::kDepthW - 1);
    const double cy = 0.5 * (ep_student::kDepthH - 1);
    for (int v = 0; v < ep_student::kDepthH; ++v)
    {
        for (int u = 0; u < ep_student::kDepthW; ++u)
        {
            const int i = v * ep_student::kDepthW + u;
            double x = (u - cx) / fx;
            double y = -(v - cy) / fy;
            double z = -1.0;
            const double nrm = std::sqrt(x * x + y * y + z * z);
            x /= nrm;
            y /= nrm;
            z /= nrm;
            rays.dirs_cam[3 * i + 0] = x;
            rays.dirs_cam[3 * i + 1] = y;
            rays.dirs_cam[3 * i + 2] = z;
            rays.plane_scale[i] = -z;
        }
    }
    rays.ready = true;
    return rays;
}
} // namespace

RL_Sim::RL_Sim(int argc, char **argv)
{
    // Set static instance pointer early for signal handler
    instance = this;

    if (argc < 3)
    {
        std::cout << LOGGER::ERROR << "Usage: " << argv[0] << " robot_name scene_name" << std::endl;
        throw std::runtime_error("Invalid arguments");
    }
    else
    {
        this->robot_name = argv[1];
        this->scene_name = argv[2];
    }

    this->ang_vel_axis = "body";

    // now launch mujoco
    std::cout << LOGGER::INFO << "[MuJoCo] Launching..." << std::endl;

    // display an error if running on macOS under Rosetta 2
#if defined(__APPLE__) && defined(__AVX__)
    if (rosetta_error_msg)
    {
        DisplayErrorDialogBox("Rosetta 2 is not supported", rosetta_error_msg);
        std::exit(1);
    }
#endif

    // print version, check compatibility
    std::cout << LOGGER::INFO << "[MuJoCo] Version: " << mj_versionString() << std::endl;
    if (mjVERSION_HEADER != mj_version())
    {
        mju_error("Headers and library have different versions");
    }

    // scan for libraries in the plugin directory to load additional plugins
    scanPluginLibraries();

    mjvCamera cam;
    mjv_defaultCamera(&cam);

    mjvOption opt;
    mjv_defaultOption(&opt);

    mjvPerturb pert;
    mjv_defaultPerturb(&pert);

    // simulate object encapsulates the UI
    sim = std::make_unique<mj::Simulate>(
        std::make_unique<mj::GlfwAdapter>(),
        &cam, &opt, &pert, /* is_passive = */ false);

    // Prefer repo scenes/go2_ep_step40 for EP step40 XMLs; fall back to rl_sar_zoo mjcf.
    std::string filename = std::string(CMAKE_CURRENT_SOURCE_DIR) + "/../rl_sar_zoo/" + this->robot_name + "_description/mjcf/" + this->scene_name + ".xml";
    {
        const std::string scenes_ep = std::string(CMAKE_CURRENT_SOURCE_DIR) + "/../../scenes/go2_ep_step40/" + this->scene_name + ".xml";
        if (std::filesystem::exists(scenes_ep))
        {
            filename = scenes_ep;
        }
    }
    std::cout << LOGGER::INFO << "[MuJoCo] scene xml: " << filename << std::endl;

    // start physics thread
    std::thread physicsthreadhandle(&PhysicsThread, sim.get(), filename.c_str());
    physicsthreadhandle.detach();

    while (1)
    {
        if (d)
        {
            std::cout << LOGGER::INFO << "[MuJoCo] Data prepared" << std::endl;
            break;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
    }

    this->mj_model = m;
    this->mj_data = d;
    this->SetupSysJoystick("/dev/input/js0", 16); // 16 bits joystick

    // read params from yaml
    this->ReadYaml(this->robot_name, "base.yaml");
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
    this->InitJointNum(this->params.Get<int>("num_of_dofs"));
    this->InitOutputs();
    this->InitControl();

    // loop
    this->loop_control = std::make_shared<LoopFunc>("loop_control", this->params.Get<float>("dt"), std::bind(&RL_Sim::RobotControl, this));
    this->loop_rl = std::make_shared<LoopFunc>("loop_rl", this->params.Get<float>("dt") * this->params.Get<int>("decimation"), std::bind(&RL_Sim::RunModel, this));
    this->loop_control->start();
    this->loop_rl->start();

    // keyboard
    this->loop_keyboard = std::make_shared<LoopFunc>("loop_keyboard", 0.05, std::bind(&RL_Sim::KeyboardInterface, this));
    this->loop_keyboard->start();

    // joystick
    this->loop_joystick = std::make_shared<LoopFunc>("loop_joystick", 0.01, std::bind(&RL_Sim::GetSysJoystick, this));
    this->loop_joystick->start();

#ifdef PLOT
    this->plot_t = std::vector<int>(this->plot_size, 0);
    this->plot_real_joint_pos.resize(this->params.Get<int>("num_of_dofs"));
    this->plot_target_joint_pos.resize(this->params.Get<int>("num_of_dofs"));
    for (auto &vector : this->plot_real_joint_pos) { vector = std::vector<float>(this->plot_size, 0); }
    for (auto &vector : this->plot_target_joint_pos) { vector = std::vector<float>(this->plot_size, 0); }
    this->loop_plot = std::make_shared<LoopFunc>("loop_plot", 0.001, std::bind(&RL_Sim::Plot, this));
    this->loop_plot->start();
#endif
#ifdef CSV_LOGGER
    this->CSVInit(this->robot_name);
#endif

    std::cout << LOGGER::INFO << "RL_Sim start" << std::endl;

    // start simulation UI loop (blocking call)
    sim->RenderLoop();
}

RL_Sim::~RL_Sim()
{
    // Clear static instance pointer
    instance = nullptr;

    this->loop_keyboard->shutdown();
    this->loop_joystick->shutdown();
    this->loop_control->shutdown();
    this->loop_rl->shutdown();
#ifdef PLOT
    this->loop_plot->shutdown();
#endif
    std::cout << LOGGER::INFO << "RL_Sim exit" << std::endl;
}

void RL_Sim::GetState(RobotState<float> *state)
{
    if (mj_data)
    {
        state->imu.quaternion[0] = mj_data->sensordata[3 * this->params.Get<int>("num_of_dofs") + 0];
        state->imu.quaternion[1] = mj_data->sensordata[3 * this->params.Get<int>("num_of_dofs") + 1];
        state->imu.quaternion[2] = mj_data->sensordata[3 * this->params.Get<int>("num_of_dofs") + 2];
        state->imu.quaternion[3] = mj_data->sensordata[3 * this->params.Get<int>("num_of_dofs") + 3];

        state->imu.gyroscope[0] = mj_data->sensordata[3 * this->params.Get<int>("num_of_dofs") + 4];
        state->imu.gyroscope[1] = mj_data->sensordata[3 * this->params.Get<int>("num_of_dofs") + 5];
        state->imu.gyroscope[2] = mj_data->sensordata[3 * this->params.Get<int>("num_of_dofs") + 6];

        for (int i = 0; i < this->params.Get<int>("num_of_dofs"); ++i)
        {
            state->motor_state.q[i] = mj_data->sensordata[this->params.Get<std::vector<int>>("joint_mapping")[i]];
            state->motor_state.dq[i] = mj_data->sensordata[this->params.Get<std::vector<int>>("joint_mapping")[i] + this->params.Get<int>("num_of_dofs")];
            state->motor_state.tau_est[i] = mj_data->sensordata[this->params.Get<std::vector<int>>("joint_mapping")[i] + 2 * this->params.Get<int>("num_of_dofs")];
        }
    }
}

void RL_Sim::SetCommand(const RobotCommand<float> *command)
{
    if (mj_data)
    {
        for (int i = 0; i < this->params.Get<int>("num_of_dofs"); ++i)
        {
            mj_data->ctrl[this->params.Get<std::vector<int>>("joint_mapping")[i]] =
                command->motor_command.tau[i] +
                command->motor_command.kp[i] * (command->motor_command.q[i] - mj_data->sensordata[this->params.Get<std::vector<int>>("joint_mapping")[i]]) +
                command->motor_command.kd[i] * (command->motor_command.dq[i] - mj_data->sensordata[this->params.Get<std::vector<int>>("joint_mapping")[i] + this->params.Get<int>("num_of_dofs")]);
        }
    }
}

void RL_Sim::RobotControl()
{
    // Lock the sim mutex once for the entire control cycle to prevent race conditions
    const std::lock_guard<std::recursive_mutex> lock(sim->mtx);

    this->GetState(&this->robot_state);

    this->StateController(&this->robot_state, &this->robot_command);

    if (this->control.current_keyboard == Input::Keyboard::R || this->control.current_gamepad == Input::Gamepad::RB_Y)
    {
        if (this->mj_model && this->mj_data)
        {
            mj_resetData(this->mj_model, this->mj_data);
            mj_forward(this->mj_model, this->mj_data);
        }
    }
    if (this->control.current_keyboard == Input::Keyboard::Enter || this->control.current_gamepad == Input::Gamepad::RB_X)
    {
        if (simulation_running)
        {
            sim->run = 0;
            std::cout << std::endl << LOGGER::INFO << "Simulation Stop" << std::endl;
        }
        else
        {
            sim->run = 1;
            std::cout << std::endl << LOGGER::INFO << "Simulation Start" << std::endl;
        }
        simulation_running = !simulation_running;
    }

    this->control.ClearInput();

    this->SetCommand(&this->robot_command);
}

void RL_Sim::SetupSysJoystick(const std::string& device, int bits)
{
    this->sys_js = std::make_unique<Joystick>(device);
    if (!this->sys_js->isFound())
    {
        std::cout << LOGGER::ERROR << "Joystick [" << device << "] open failed." << std::endl;
        // exit(1);
    }

    this->sys_js_max_value = (1 << (bits - 1));
}

void RL_Sim::GetSysJoystick()
{
    // Clear all button event states
    for (int i = 0; i < 20; ++i)
    {
        this->sys_js_button[i].on_press = false;
        this->sys_js_button[i].on_release = false;
    }

    // Check if joystick is valid before using
    if (!this->sys_js)
    {
        return;
    }

    while (this->sys_js->sample(&this->sys_js_event))
    {
        if (this->sys_js_event.isButton())
        {
            this->sys_js_button[this->sys_js_event.number].update(this->sys_js_event.value);
        }
        else if (this->sys_js_event.isAxis())
        {
            double normalized = double(this->sys_js_event.value) / this->sys_js_max_value;
            if (std::abs(normalized) < this->axis_deadzone)
            {
                this->sys_js_axis[this->sys_js_event.number] = 0;
            }
            else
            {
                this->sys_js_axis[this->sys_js_event.number] = this->sys_js_event.value;
            }
        }
    }

    if (this->sys_js_button[0].on_press) this->control.SetGamepad(Input::Gamepad::A);
    if (this->sys_js_button[1].on_press) this->control.SetGamepad(Input::Gamepad::B);
    if (this->sys_js_button[2].on_press) this->control.SetGamepad(Input::Gamepad::X);
    if (this->sys_js_button[3].on_press) this->control.SetGamepad(Input::Gamepad::Y);
    if (this->sys_js_button[4].on_press) this->control.SetGamepad(Input::Gamepad::LB);
    if (this->sys_js_button[5].on_press) this->control.SetGamepad(Input::Gamepad::RB);
    if (this->sys_js_button[9].on_press) this->control.SetGamepad(Input::Gamepad::LStick);
    if (this->sys_js_button[10].on_press) this->control.SetGamepad(Input::Gamepad::RStick);
    if (this->sys_js_axis[7] < 0) this->control.SetGamepad(Input::Gamepad::DPadUp);
    if (this->sys_js_axis[7] > 0) this->control.SetGamepad(Input::Gamepad::DPadDown);
    if (this->sys_js_axis[6] > 0) this->control.SetGamepad(Input::Gamepad::DPadLeft);
    if (this->sys_js_axis[6] < 0) this->control.SetGamepad(Input::Gamepad::DPadRight);
    if (this->sys_js_button[4].pressed && this->sys_js_button[0].on_press) this->control.SetGamepad(Input::Gamepad::LB_A);
    if (this->sys_js_button[4].pressed && this->sys_js_button[1].on_press) this->control.SetGamepad(Input::Gamepad::LB_B);
    if (this->sys_js_button[4].pressed && this->sys_js_button[2].on_press) this->control.SetGamepad(Input::Gamepad::LB_X);
    if (this->sys_js_button[4].pressed && this->sys_js_button[3].on_press) this->control.SetGamepad(Input::Gamepad::LB_Y);
    if (this->sys_js_button[4].pressed && this->sys_js_button[9].on_press) this->control.SetGamepad(Input::Gamepad::LB_LStick);
    if (this->sys_js_button[4].pressed && this->sys_js_button[10].on_press) this->control.SetGamepad(Input::Gamepad::LB_RStick);
    if (this->sys_js_button[4].pressed && this->sys_js_axis[7] < 0) this->control.SetGamepad(Input::Gamepad::LB_DPadUp);
    if (this->sys_js_button[4].pressed && this->sys_js_axis[7] > 0) this->control.SetGamepad(Input::Gamepad::LB_DPadDown);
    if (this->sys_js_button[4].pressed && this->sys_js_axis[6] > 0) this->control.SetGamepad(Input::Gamepad::LB_DPadRight);
    if (this->sys_js_button[4].pressed && this->sys_js_axis[6] < 0) this->control.SetGamepad(Input::Gamepad::LB_DPadLeft);
    if (this->sys_js_button[5].pressed && this->sys_js_button[0].on_press) this->control.SetGamepad(Input::Gamepad::RB_A);
    if (this->sys_js_button[5].pressed && this->sys_js_button[1].on_press) this->control.SetGamepad(Input::Gamepad::RB_B);
    if (this->sys_js_button[5].pressed && this->sys_js_button[2].on_press) this->control.SetGamepad(Input::Gamepad::RB_X);
    if (this->sys_js_button[5].pressed && this->sys_js_button[3].on_press) this->control.SetGamepad(Input::Gamepad::RB_Y);
    if (this->sys_js_button[5].pressed && this->sys_js_button[9].on_press) this->control.SetGamepad(Input::Gamepad::RB_LStick);
    if (this->sys_js_button[5].pressed && this->sys_js_button[10].on_press) this->control.SetGamepad(Input::Gamepad::RB_RStick);
    if (this->sys_js_button[5].pressed && this->sys_js_axis[7] < 0) this->control.SetGamepad(Input::Gamepad::RB_DPadUp);
    if (this->sys_js_button[5].pressed && this->sys_js_axis[7] > 0) this->control.SetGamepad(Input::Gamepad::RB_DPadDown);
    if (this->sys_js_button[5].pressed && this->sys_js_axis[6] > 0) this->control.SetGamepad(Input::Gamepad::RB_DPadRight);
    if (this->sys_js_button[5].pressed && this->sys_js_axis[6] < 0) this->control.SetGamepad(Input::Gamepad::RB_DPadLeft);
    if (this->sys_js_button[4].pressed && this->sys_js_button[5].on_press) this->control.SetGamepad(Input::Gamepad::LB_RB);

    float ly = -float(this->sys_js_axis[1]) / float(this->sys_js_max_value);
    float lx = -float(this->sys_js_axis[0]) / float(this->sys_js_max_value);
    float rx = -float(this->sys_js_axis[3]) / float(this->sys_js_max_value);

    bool has_input = (ly != 0.0f || lx != 0.0f || rx != 0.0f);

    if (has_input)
    {
        this->control.x = ly;
        this->control.y = lx;
        this->control.yaw = rx;
        this->sys_js_active = true;
    }
    else if (this->sys_js_active)
    {
        this->control.x = 0.0f;
        this->control.y = 0.0f;
        this->control.yaw = 0.0f;
        this->sys_js_active = false;
    }
}

void RL_Sim::RunModel()
{
    if (this->rl_init_done && simulation_running)
    {
        this->episode_length_buf += 1;
        this->obs.ang_vel = this->robot_state.imu.gyroscope;
        // Headless smokes: lock commands so Xvfb phantom keystrokes cannot ramp vy/yaw.
        // RL_SAR_AUTO_RL=1 ⇒ default cmd (0,0,0) unless RL_SAR_CMD_* set.
        const char* auto_rl = std::getenv("RL_SAR_AUTO_RL");
        const bool lock_cmd = (auto_rl && *auto_rl && std::string(auto_rl) != "0")
            || std::getenv("RL_SAR_CMD_X") || std::getenv("RL_SAR_CMD_Y") || std::getenv("RL_SAR_CMD_YAW");
        if (lock_cmd)
        {
            const char* cx = std::getenv("RL_SAR_CMD_X");
            const char* cy = std::getenv("RL_SAR_CMD_Y");
            const char* cyaw = std::getenv("RL_SAR_CMD_YAW");
            this->control.x = cx ? std::strtof(cx, nullptr) : 0.0f;
            this->control.y = cy ? std::strtof(cy, nullptr) : 0.0f;
            this->control.yaw = cyaw ? std::strtof(cyaw, nullptr) : 0.0f;
            this->control.SetKeyboard(Input::Keyboard::None);
        }
        this->obs.commands = {this->control.x, this->control.y, this->control.yaw};
        //not currently available for non-ros mujoco version
        // if (this->control.navigation_mode)
        // {
        //     this->obs.commands = {(float)this->cmd_vel.linear.x, (float)this->cmd_vel.linear.y, (float)this->cmd_vel.angular.z};
        // }
        this->obs.base_quat = this->robot_state.imu.quaternion;
        this->obs.dof_pos = this->robot_state.motor_state.q;
        this->obs.dof_vel = this->robot_state.motor_state.dq;

        this->obs.actions = this->Forward();
        if (this->config_name == "go2_ep_student" && mj_data && mj_model && (this->episode_length_buf % 50 == 0))
        {
            // Prefer torso/base body height; freejoint qpos[2] can be misleading if model differs.
            float z = (float)mj_data->qpos[2];
            int bid = mj_name2id(mj_model, mjOBJ_BODY, "base");
            if (bid < 0) bid = mj_name2id(mj_model, mjOBJ_BODY, "base_link");
            if (bid < 0) bid = mj_name2id(mj_model, mjOBJ_BODY, "trunk");
            if (bid >= 0) z = (float)mj_data->xpos[3 * bid + 2];
            float amax = 0.0f;
            for (float a : this->obs.actions) amax = std::max(amax, std::fabs(a));
            // stderr avoids FSM "\r" line-clear races on stdout
            std::cerr << "[EP_MUJOCO] t=" << this->episode_length_buf
                      << " z=" << z << " cmd_x=" << this->control.x
                      << " amax=" << amax << std::endl;
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

        // this->TorqueProtect(this->output_dof_tau);
        // this->AttitudeProtect(this->robot_state.imu.quaternion, 75.0f, 75.0f);

#ifdef CSV_LOGGER
        std::vector<float> tau_est(this->params.Get<int>("num_of_dofs"), 0.0f);
        for (int i = 0; i < this->params.Get<int>("num_of_dofs"); ++i)
        {
            tau_est[i] = this->joint_efforts[this->params.Get<std::vector<std::string>>("joint_controller_names")[i]];
        }
        this->CSVLogger(this->output_dof_tau, tau_est, this->obs.dof_pos, this->output_dof_pos, this->obs.dof_vel);
#endif
    }
}

bool RL_Sim::EpSceneIsStep() const
{
    return this->scene_name.find("step40") != std::string::npos;
}

bool RL_Sim::EpTerrainIsFlat() const
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
    return !this->EpSceneIsStep();
}

void RL_Sim::ResetEpDepthState()
{
    this->ep_latent_.assign(EP_LATENT, 0.0f);
    this->ep_yaw_.assign(2, 0.0f);
    this->ep_depth_proc_.assign(ep_student::kOutH * ep_student::kOutW, -0.5f);
    this->ep_encoder_.Reset();
}

void RL_Sim::EnsureEpDepthEncoder()
{
    std::string weight_name = "vision_weight.bin";
    try
    {
        weight_name = this->params.Get<std::string>("vision_weight_name");
    }
    catch (...)
    {
    }
    const std::string path = std::string(POLICY_DIR) + "/" + this->robot_name + "/" + this->config_name + "/" + weight_name;
    this->ep_encoder_.EnsureLoaded(path);
}

void RL_Sim::UpdateEpDepthLatent(const std::vector<float> &proprio53)
{
    if (!this->mj_model || !this->mj_data || (int)proprio53.size() < EP_N_PROPRIO)
    {
        return;
    }
    try
    {
        this->EnsureEpDepthEncoder();
    }
    catch (const std::exception &e)
    {
        static bool warned = false;
        if (!warned)
        {
            warned = true;
            std::cerr << "[EP_DEPTH] encoder unavailable, latent stays 0: " << e.what() << std::endl;
        }
        if ((int)this->ep_latent_.size() != EP_LATENT)
        {
            this->ep_latent_.assign(EP_LATENT, 0.0f);
        }
        if ((int)this->ep_yaw_.size() != 2)
        {
            this->ep_yaw_.assign(2, 0.0f);
        }
        return;
    }

    int bid = mj_name2id(this->mj_model, mjOBJ_BODY, "base");
    if (bid < 0)
        bid = mj_name2id(this->mj_model, mjOBJ_BODY, "base_link");
    if (bid < 0)
    {
        return;
    }
    const mjtNum *xpos = this->mj_data->xpos + 3 * bid;
    const mjtNum *xmat = this->mj_data->xmat + 9 * bid;

    const double pitch = kPitchDeg * 3.14159265358979323846 / 180.0;
    const double forward[3] = {std::cos(pitch), 0.0, -std::sin(pitch)};
    const double right[3] = {0.0, -1.0, 0.0};
    const double z_cam[3] = {-forward[0], -forward[1], -forward[2]};
    double up[3] = {
        z_cam[1] * right[2] - z_cam[2] * right[1],
        z_cam[2] * right[0] - z_cam[0] * right[2],
        z_cam[0] * right[1] - z_cam[1] * right[0],
    };
    const double up_n = std::sqrt(up[0] * up[0] + up[1] * up[1] + up[2] * up[2]);
    up[0] /= up_n;
    up[1] /= up_n;
    up[2] /= up_n;
    const double r_cb[9] = {
        right[0], up[0], z_cam[0],
        right[1], up[1], z_cam[1],
        right[2], up[2], z_cam[2],
    };
    double r_cw[9];
    for (int i = 0; i < 3; ++i)
    {
        for (int j = 0; j < 3; ++j)
        {
            r_cw[3 * i + j] = 0.0;
            for (int k = 0; k < 3; ++k)
            {
                r_cw[3 * i + j] += xmat[3 * i + k] * r_cb[3 * k + j];
            }
        }
    }
    double cam_body_w[3];
    for (int i = 0; i < 3; ++i)
    {
        cam_body_w[i] = xmat[3 * i + 0] * kCamBody[0] + xmat[3 * i + 1] * kCamBody[1] + xmat[3 * i + 2] * kCamBody[2];
    }
    const mjtNum pnt[3] = {xpos[0] + cam_body_w[0], xpos[1] + cam_body_w[1], xpos[2] + cam_body_w[2]};

    PlaneRays &rays = Rays();
    const int nray = ep_student::kDepthH * ep_student::kDepthW;
    std::vector<mjtNum> dirs_w(nray * 3);
    for (int i = 0; i < nray; ++i)
    {
        const double x = rays.dirs_cam[3 * i + 0];
        const double y = rays.dirs_cam[3 * i + 1];
        const double z = rays.dirs_cam[3 * i + 2];
        dirs_w[3 * i + 0] = r_cw[0] * x + r_cw[1] * y + r_cw[2] * z;
        dirs_w[3 * i + 1] = r_cw[3] * x + r_cw[4] * y + r_cw[5] * z;
        dirs_w[3 * i + 2] = r_cw[6] * x + r_cw[7] * y + r_cw[8] * z;
    }
    std::vector<int> geomid(nray, -1);
    std::vector<mjtNum> dist(nray, -1.0);
    mjtByte geomgroup[6] = {1, 1, 0, 0, 1, 1};
    mj_multiRay(
        this->mj_model, this->mj_data, pnt, dirs_w.data(), geomgroup, 1, bid,
        geomid.data(), dist.data(), nray, kFar * 2.0);

    std::vector<float> meters(nray, static_cast<float>(kFar));
    for (int i = 0; i < nray; ++i)
    {
        if (dist[i] >= 0.0)
        {
            meters[i] = static_cast<float>(std::min(dist[i] * rays.plane_scale[i], kFar));
        }
        if (meters[i] < 0.0f)
        {
            meters[i] = 0.0f;
        }
    }
    // Already on the 60x106 training grid: crop + bicubic via shared preprocess.
    ep_student::PreprocessDepthMeters(meters.data(), ep_student::kDepthH, ep_student::kDepthW, this->ep_depth_proc_);
    this->ep_encoder_.Encode(this->ep_depth_proc_, proprio53, this->ep_latent_, this->ep_yaw_);
}

void RL_Sim::MaybeDumpEpAction(const std::vector<float> &actions)
{
    const char *path = std::getenv("RL_SAR_EP_ACTION_DUMP");
    if (!path || !*path)
    {
        return;
    }
    if (!this->ep_action_dump_open_)
    {
        this->ep_action_dump_.open(path, std::ios::out);
        this->ep_action_dump_open_ = this->ep_action_dump_.is_open();
        if (!this->ep_action_dump_open_)
        {
            std::cerr << "[EP_MUJOCO] cannot open RL_SAR_EP_ACTION_DUMP=" << path << std::endl;
            return;
        }
        this->ep_action_dump_ << "policy_step,z,cmd_x,amax";
        for (int i = 0; i < 12; ++i)
            this->ep_action_dump_ << ",a" << i;
        this->ep_action_dump_ << "\n";
        std::cout << LOGGER::INFO << "[EP_MUJOCO] action dump -> " << path << std::endl;
    }
    float z = 0.0f;
    if (this->mj_data && this->mj_model)
    {
        z = (float)this->mj_data->qpos[2];
        int bid = mj_name2id(this->mj_model, mjOBJ_BODY, "base");
        if (bid < 0)
            bid = mj_name2id(this->mj_model, mjOBJ_BODY, "base_link");
        if (bid >= 0)
            z = (float)this->mj_data->xpos[3 * bid + 2];
    }
    float amax = 0.0f;
    for (float a : actions)
        amax = std::max(amax, std::fabs(a));
    this->ep_action_dump_ << this->ep_policy_step_ << "," << z << "," << this->control.x << "," << amax;
    for (int i = 0; i < 12 && i < (int)actions.size(); ++i)
        this->ep_action_dump_ << "," << actions[i];
    this->ep_action_dump_ << "\n";
}

std::vector<float> RL_Sim::Forward()
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
    // Plane depth updates encoder every 5 policy steps; other steps reuse latent.
    if (this->config_name == "go2_ep_student")
    {
        if (this->episode_length_buf <= 1)
        {
            this->ep_hist_ready_ = false;
            this->ep_policy_step_ = 0;
            this->ResetEpDepthState();
        }
        std::vector<float> ep_obs = this->BuildEpObservation();
        if (this->ep_policy_step_ % EP_UPDATE_INTERVAL == 0)
        {
            std::vector<float> proprio(ep_obs.begin(), ep_obs.begin() + EP_N_PROPRIO);
            proprio[6] = 0.0f;
            proprio[7] = 0.0f;
            this->UpdateEpDepthLatent(proprio);
        }
        if (this->ep_yaw_.size() >= 2)
        {
            ep_obs[6] = 1.5f * this->ep_yaw_[0];
            ep_obs[7] = 1.5f * this->ep_yaw_[1];
        }
        if ((int)this->ep_latent_.size() != EP_LATENT)
        {
            this->ep_latent_.assign(EP_LATENT, 0.0f);
        }
        actions = this->model->forward({ep_obs, this->ep_latent_});
        this->MaybeDumpEpAction(actions);
        this->ep_policy_step_ += 1;
        if ((int)actions.size() >= 12)
        {
            this->ep_last_action_.assign(actions.begin(), actions.begin() + 12);
        }
    }
    else
    {
        std::vector<float> clamped_obs = this->ComputeObservation();
        if (this->params.Get<std::vector<int>>("observations_history").size() != 0)
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


std::vector<float> RL_Sim::BuildEpObservation()
{
    // EP proprio (53): ang_vel(3), imu roll/pitch(2), yaw slots(3), cmd slots(3),
    // terrain one-hots(2), dof_pos(12), dof_vel(12), last_action(12), contact(4)
    // Then scan(132)=0, priv_explicit(9)=0 (JIT estimator overwrites), priv_latent(29)=0,
    // history(10*53). dof_* already in policy order via joint_mapping.
    std::vector<float> proprio(EP_N_PROPRIO, 0.0f);

    float ang_scale = 0.25f;
    float dof_pos_scale = 1.0f;
    float dof_vel_scale = 0.05f;
    try { ang_scale = this->params.Get<float>("ang_vel_scale"); } catch (...) {}
    try { dof_pos_scale = this->params.Get<float>("dof_pos_scale"); } catch (...) {}
    try { dof_vel_scale = this->params.Get<float>("dof_vel_scale"); } catch (...) {}

    if (this->obs.ang_vel.size() >= 3)
    {
        proprio[0] = this->obs.ang_vel[0] * ang_scale;
        proprio[1] = this->obs.ang_vel[1] * ang_scale;
        proprio[2] = this->obs.ang_vel[2] * ang_scale;
    }

    // roll / pitch from base quaternion (wxyz)
    if (this->obs.base_quat.size() >= 4)
    {
        std::vector<float> euler = QuaternionToEuler(this->obs.base_quat);
        proprio[3] = euler[0]; // roll
        proprio[4] = euler[1]; // pitch
    }

    // indices 5,6,7 stay 0 here. Forward writes yaw*1.5 into 6:8 after depth update.
    // commands: 8,9 masked 0; 10 = vx
    proprio[10] = this->control.x;
    // step40 scenes default to non-flat one-hot; RL_SAR_EP_TERRAIN=parkour|flat overrides.
    const bool flat = this->EpTerrainIsFlat();
    proprio[11] = flat ? 0.0f : 1.0f;
    proprio[12] = flat ? 1.0f : 0.0f;

    auto default_pos = this->params.Get<std::vector<float>>("default_dof_pos");
    if (this->ep_last_action_.size() != 12)
        this->ep_last_action_.assign(12, 0.0f);

    for (int i = 0; i < 12; ++i)
    {
        float q0 = (i < (int)default_pos.size()) ? default_pos[i] : 0.0f;
        float q = (i < (int)this->obs.dof_pos.size()) ? this->obs.dof_pos[i] : q0;
        float dq = (i < (int)this->obs.dof_vel.size()) ? this->obs.dof_vel[i] : 0.0f;
        proprio[13 + i] = (q - q0) * dof_pos_scale;
        proprio[25 + i] = dq * dof_vel_scale;
        proprio[37 + i] = this->ep_last_action_[i];
    }
    // contact unknown in stock MuJoCo → 0 (= contact_filt.float()-0.5 mid); use -0.5 (no contact)
    for (int i = 0; i < 4; ++i)
        proprio[49 + i] = -0.5f;

    // history: mask yaw slots like EP (indices 6:8 = 0) before pushing
    std::vector<float> hist_frame = proprio;
    hist_frame[6] = 0.0f;
    hist_frame[7] = 0.0f;

    if (!this->ep_hist_ready_ || (int)this->ep_proprio_hist_.size() != EP_HIST_LEN * EP_N_PROPRIO)
    {
        this->ep_proprio_hist_.assign(EP_HIST_LEN * EP_N_PROPRIO, 0.0f);
        for (int h = 0; h < EP_HIST_LEN; ++h)
            std::copy(hist_frame.begin(), hist_frame.end(), this->ep_proprio_hist_.begin() + h * EP_N_PROPRIO);
        this->ep_hist_ready_ = true;
    }
    else
    {
        // drop oldest, append newest
        std::copy(this->ep_proprio_hist_.begin() + EP_N_PROPRIO, this->ep_proprio_hist_.end(), this->ep_proprio_hist_.begin());
        std::copy(hist_frame.begin(), hist_frame.end(), this->ep_proprio_hist_.end() - EP_N_PROPRIO);
    }

    std::vector<float> ep_obs(EP_N_OBS, 0.0f);
    // current proprio (with live yaw slots — zeros here)
    std::copy(proprio.begin(), proprio.end(), ep_obs.begin());
    // scan zeros already
    // priv_explicit / priv_latent zeros — JIT estimator fills priv_explicit
    const int hist_off = EP_N_PROPRIO + EP_N_SCAN + EP_N_PRIV_E + EP_N_PRIV_L;
    std::copy(this->ep_proprio_hist_.begin(), this->ep_proprio_hist_.end(), ep_obs.begin() + hist_off);

    // clip_obs
    float clip = 100.0f;
    try { clip = this->params.Get<float>("clip_obs"); } catch (...) {}
    for (float &v : ep_obs)
    {
        if (v > clip) v = clip;
        if (v < -clip) v = -clip;
    }
    return ep_obs;
}

void RL_Sim::Plot()
{
    this->plot_t.erase(this->plot_t.begin());
    this->plot_t.push_back(this->motiontime);
    plt::cla();
    plt::clf();
    for (int i = 0; i < this->params.Get<int>("num_of_dofs"); ++i)
    {
        this->plot_real_joint_pos[i].erase(this->plot_real_joint_pos[i].begin());
        this->plot_target_joint_pos[i].erase(this->plot_target_joint_pos[i].begin());
        this->plot_real_joint_pos[i].push_back(mj_data->sensordata[i]);
        // this->plot_target_joint_pos[i].push_back();  // TODO
        plt::subplot(this->params.Get<int>("num_of_dofs"), 1, i + 1);
        plt::named_plot("_real_joint_pos", this->plot_t, this->plot_real_joint_pos[i], "r");
        plt::named_plot("_target_joint_pos", this->plot_t, this->plot_target_joint_pos[i], "b");
        plt::xlim(this->plot_t.front(), this->plot_t.back());
    }
    // plt::legend();
    plt::pause(0.01);
}

// Signal handler for Ctrl+C
void signalHandler(int signum)
{
    std::cout << LOGGER::INFO << "Received signal " << signum << ", exiting..." << std::endl;
    if (RL_Sim::instance && RL_Sim::instance->sim)
    {
        RL_Sim::instance->sim->exitrequest.store(1);
    }
}

int main(int argc, char **argv)
{
    signal(SIGINT, signalHandler);
    RL_Sim rl_sar(argc, argv);
    return 0;
}
