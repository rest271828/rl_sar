/*
 * Offline unit check: load vision_weight + base_jit, preprocess depth,
 * run dual-input Forward once. Does NOT talk to the robot.
 *
 * Usage:
 *   ./cmake_build/bin/ep_depth_smoke
 *   RL_SAR_EP_DEPTH_BIN=/path/to.mm RL_SAR_EP_DEPTH_WH=848x480 ./cmake_build/bin/ep_depth_smoke
 *
 * Valid% gate (sim/offline AB; default min_valid=0.2):
 *   RL_SAR_EP_DEPTH_GATE=fill|hold|refuse   (default fill)
 *   RL_SAR_EP_DEPTH_MIN_VALID=0.2
 *   RL_SAR_EP_DEPTH_HOLD_BIN=/path/to_prev_good.mm   (required for hold when current is bad)
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

namespace
{

bool LoadMm(const char *path, int w, int h, std::vector<uint16_t> &mm)
{
    std::ifstream ifs(path, std::ios::binary);
    if (!ifs)
    {
        std::cerr << "SMOKE_FAIL cannot open " << path << std::endl;
        return false;
    }
    mm.assign(static_cast<size_t>(w) * static_cast<size_t>(h), 0);
    ifs.read(reinterpret_cast<char *>(mm.data()),
             static_cast<std::streamsize>(mm.size() * sizeof(uint16_t)));
    if (!ifs)
    {
        std::cerr << "SMOKE_FAIL short read " << path << std::endl;
        return false;
    }
    return true;
}

float EnvFloat(const char *key, float def)
{
    if (const char *s = std::getenv(key))
    {
        char *end = nullptr;
        const float v = std::strtof(s, &end);
        if (end != s)
        {
            return v;
        }
    }
    return def;
}

} // namespace

int main(int argc, char **argv)
{
    (void)argc;
    (void)argv;

    const std::string policy_dir = std::string(POLICY_DIR) + "/go2/go2_ep_student";
    const std::string vision_path = policy_dir + "/vision_weight.bin";
    const std::string jit_path = policy_dir + "/base_jit.pt";

    std::vector<float> depth_proc;
    std::string gate = "fill";
    if (const char *g = std::getenv("RL_SAR_EP_DEPTH_GATE"))
    {
        gate = g;
    }
    const float min_valid = EnvFloat("RL_SAR_EP_DEPTH_MIN_VALID", ep_student::kDefaultMinValidFrac);
    float valid_frac = 1.0f;
    bool gated_refuse = false;
    std::string gate_tag = "FILL";

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
        std::vector<uint16_t> mm;
        if (!LoadMm(bin, w, h, mm))
        {
            return 2;
        }
        valid_frac = ep_student::DepthValidFractionUint16Mm(mm.data(), h, w);
        const bool ok = ep_student::DepthFrameAcceptableUint16Mm(mm.data(), h, w, min_valid);
        std::cout << "SMOKE depth from bin " << bin << " " << w << "x" << h
                  << " valid_frac=" << valid_frac << " min_valid=" << min_valid
                  << " gate=" << gate << std::endl;

        if (!ok && gate == "refuse")
        {
            gated_refuse = true;
            gate_tag = "REFUSE";
            depth_proc.assign(ep_student::kOutH * ep_student::kOutW, -0.5f);
            std::cout << "SMOKE GATE_REFUSE skip encoder update (latent stays zero)" << std::endl;
        }
        else if (!ok && gate == "hold")
        {
            const char *hold_bin = std::getenv("RL_SAR_EP_DEPTH_HOLD_BIN");
            if (!hold_bin)
            {
                std::cerr << "SMOKE_FAIL hold gate needs RL_SAR_EP_DEPTH_HOLD_BIN" << std::endl;
                return 2;
            }
            std::vector<uint16_t> hold;
            if (!LoadMm(hold_bin, w, h, hold))
            {
                return 2;
            }
            // Blend: keep current valid pixels, fill holes from last-good frame.
            for (size_t i = 0; i < mm.size(); ++i)
            {
                if (mm[i] == 0)
                {
                    mm[i] = hold[i];
                }
            }
            gate_tag = "HOLD";
            ep_student::PreprocessDepthUint16Mm(mm.data(), h, w, depth_proc);
            std::cout << "SMOKE GATE_HOLD using " << hold_bin
                      << " blended_valid="
                      << ep_student::DepthValidFractionUint16Mm(mm.data(), h, w) << std::endl;
        }
        else
        {
            gate_tag = ok ? "ACCEPT" : "FILL_BAD";
            ep_student::PreprocessDepthUint16Mm(mm.data(), h, w, depth_proc);
        }
    }
    else
    {
        // Synthetic 848x480 ramp in millimeters (dense → always acceptable).
        const int w = 848, h = 480;
        std::vector<uint16_t> mm(static_cast<size_t>(w) * static_cast<size_t>(h));
        for (int y = 0; y < h; ++y)
        {
            for (int x = 0; x < w; ++x)
            {
                mm[y * w + x] = static_cast<uint16_t>(500 + (x % 1500));
            }
        }
        valid_frac = ep_student::DepthValidFractionUint16Mm(mm.data(), h, w);
        ep_student::PreprocessDepthUint16Mm(mm.data(), h, w, depth_proc);
        gate_tag = "SYNTH";
        std::cout << "SMOKE synthetic depth 848x480 valid_frac=" << valid_frac << std::endl;
    }

    if ((int)depth_proc.size() != ep_student::kOutH * ep_student::kOutW)
    {
        std::cerr << "SMOKE_FAIL bad preprocess size " << depth_proc.size() << std::endl;
        return 3;
    }

    std::vector<float> latent(ep_student::kLatent, 0.0f);
    std::vector<float> yaw(2, 0.0f);

    if (!gated_refuse)
    {
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

    float lat_l2 = 0.0f;
    float amax = 0.0f;
    for (float v : latent)
    {
        lat_l2 += v * v;
    }
    lat_l2 = std::sqrt(lat_l2);
    for (int i = 0; i < 12; ++i)
    {
        amax = std::max(amax, std::fabs(actions[i]));
    }
    std::cout << "SMOKE_OK_EP_DEPTH gate_tag=" << gate_tag
              << " valid_frac=" << valid_frac
              << " encoder_updated=" << (gated_refuse ? 0 : 1)
              << " latent_l2=" << lat_l2
              << " amax=" << amax
              << " yaw=(" << yaw[0] << "," << yaw[1] << ")"
              << " action0=" << actions[0] << std::endl;
    return 0;
}
