/*
 * Offline unit check: load vision_weight + base_jit, preprocess synthetic depth,
 * run dual-input Forward once. Does NOT talk to the robot.
 *
 * Usage:
 *   ./cmake_build/bin/ep_depth_smoke
 *   RL_SAR_EP_DEPTH_BIN=/tmp/d.bin RL_SAR_EP_DEPTH_WH=848x480 ./cmake_build/bin/ep_depth_smoke
 */

#include "ep_student.hpp"
#include "inference_runtime.hpp"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

int main(int argc, char **argv)
{
    (void)argc;
    (void)argv;

    const std::string policy_dir = std::string(POLICY_DIR) + "/go2/go2_ep_student";
    const std::string vision_path = policy_dir + "/vision_weight.pt";
    const std::string jit_path = policy_dir + "/base_jit.pt";

    std::vector<float> depth_proc;
    if (const char *bin = std::getenv("RL_SAR_EP_DEPTH_BIN"))
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
        if (!ifs)
        {
            std::cerr << "SMOKE_FAIL cannot open " << bin << std::endl;
            return 2;
        }
        std::vector<uint16_t> mm(static_cast<size_t>(w) * static_cast<size_t>(h));
        ifs.read(reinterpret_cast<char *>(mm.data()),
                 static_cast<std::streamsize>(mm.size() * sizeof(uint16_t)));
        if (!ifs)
        {
            std::cerr << "SMOKE_FAIL short read " << bin << std::endl;
            return 2;
        }
        // Fill mid-range millimeters (~1.0 m) where zeros appear for a denser test.
        for (auto &v : mm)
        {
            if (v == 0)
            {
                v = 1000;
            }
        }
        ep_student::PreprocessDepthUint16Mm(mm.data(), h, w, depth_proc);
        std::cout << "SMOKE depth from bin " << bin << " " << w << "x" << h << std::endl;
    }
    else
    {
        // Synthetic 848x480 ramp in millimeters.
        const int w = 848, h = 480;
        std::vector<uint16_t> mm(static_cast<size_t>(w) * static_cast<size_t>(h));
        for (int y = 0; y < h; ++y)
        {
            for (int x = 0; x < w; ++x)
            {
                mm[y * w + x] = static_cast<uint16_t>(500 + (x % 1500));
            }
        }
        ep_student::PreprocessDepthUint16Mm(mm.data(), h, w, depth_proc);
        std::cout << "SMOKE synthetic depth 848x480" << std::endl;
    }

    if ((int)depth_proc.size() != ep_student::kOutH * ep_student::kOutW)
    {
        std::cerr << "SMOKE_FAIL bad preprocess size " << depth_proc.size() << std::endl;
        return 3;
    }

    ep_student::DepthEncoder enc;
    try
    {
        enc.EnsureLoaded(vision_path);
    }
    catch (const std::exception &e)
    {
        std::cerr << "SMOKE_FAIL encoder load: " << e.what() << std::endl;
        return 4;
    }

    std::vector<float> proprio(ep_student::kNProprio, 0.0f);
    proprio[10] = 0.5f; // vx
    proprio[11] = 1.0f; // parkour
    std::vector<float> latent, yaw;
    try
    {
        enc.Encode(depth_proc, proprio, latent, yaw);
    }
    catch (const std::exception &e)
    {
        std::cerr << "SMOKE_FAIL encode: " << e.what() << std::endl;
        return 5;
    }
    if ((int)latent.size() != ep_student::kLatent || (int)yaw.size() != 2)
    {
        std::cerr << "SMOKE_FAIL latent/yaw size" << std::endl;
        return 5;
    }

    auto model = InferenceRuntime::ModelFactory::load_model(jit_path);
    if (!model)
    {
        std::cerr << "SMOKE_FAIL load jit " << jit_path << std::endl;
        return 6;
    }

    ep_student::ProprioInput pin;
    pin.ang_vel = {0, 0, 0};
    pin.base_quat = {1, 0, 0, 0};
    pin.cmd_vx = 0.5f;
    pin.terrain_flat = false;
    pin.dof_pos.assign(12, 0.0f);
    pin.dof_vel.assign(12, 0.0f);
    pin.last_action.assign(12, 0.0f);
    pin.default_pos = {-0.1f, 0.7f, -1.5f, 0.1f, 0.7f, -1.5f, -0.1f, 1.0f, -1.5f, 0.1f, 1.0f, -1.5f};
    ep_student::HistState hist;
    std::vector<float> obs = ep_student::BuildObservation(pin, hist);
    obs[6] = 1.5f * yaw[0];
    obs[7] = 1.5f * yaw[1];
    if ((int)obs.size() != ep_student::kNObs)
    {
        std::cerr << "SMOKE_FAIL obs dim " << obs.size() << std::endl;
        return 7;
    }

    std::vector<float> actions;
    try
    {
        actions = model->forward({obs, latent});
    }
    catch (const std::exception &e)
    {
        std::cerr << "SMOKE_FAIL jit forward: " << e.what() << std::endl;
        return 8;
    }
    if ((int)actions.size() < 12)
    {
        std::cerr << "SMOKE_FAIL action size " << actions.size() << std::endl;
        return 8;
    }

    float lat_abs = 0.0f;
    for (float v : latent)
    {
        lat_abs += std::fabs(v);
    }
    std::cout << "SMOKE_OK_EP_DEPTH obs=" << obs.size()
              << " latent_l1=" << lat_abs
              << " yaw=(" << yaw[0] << "," << yaw[1] << ")"
              << " action0=" << actions[0] << std::endl;
    return 0;
}
