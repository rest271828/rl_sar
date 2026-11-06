/*
 * Shared Extreme Parkour Student helpers for MuJoCo sim and real Go2.
 * Dual-input contract: obs[753] + depth_latent[32] -> action[12].
 */

#ifndef EP_STUDENT_HPP
#define EP_STUDENT_HPP

#include <cstdint>
#include <mutex>
#include <string>
#include <vector>

namespace ep_student
{

constexpr int kNProprio = 53;
constexpr int kNScan = 132;
constexpr int kNPrivE = 9;
constexpr int kNPrivL = 29;
constexpr int kHistLen = 10;
constexpr int kNObs = kNProprio + kNScan + kNPrivE + kNPrivL + kHistLen * kNProprio; // 753
constexpr int kUpdateInterval = 5;
constexpr int kLatent = 32;
constexpr int kDepthH = 60;
constexpr int kDepthW = 106;
constexpr int kCropH = 58;
constexpr int kCropW = 98;
constexpr int kOutH = 58;
constexpr int kOutW = 87;
constexpr float kFarMeters = 2.0f;

struct ObsScales
{
    float ang_vel = 0.25f;
    float dof_pos = 1.0f;
    float dof_vel = 0.05f;
    float clip_obs = 100.0f;
};

struct ProprioInput
{
    std::vector<float> ang_vel;   // 3
    std::vector<float> base_quat; // wxyz, 4
    float cmd_vx = 0.0f;
    bool terrain_flat = false;
    std::vector<float> dof_pos;      // 12 policy order
    std::vector<float> dof_vel;      // 12
    std::vector<float> last_action;  // 12
    std::vector<float> default_pos;  // 12
    ObsScales scales;
};

// History state for BuildObservation.
struct HistState
{
    std::vector<float> proprio_hist; // hist_len * n_proprio
    bool ready = false;
};

struct DepthEncoder
{
    bool ok = false;
#ifdef USE_TORCH
    // Implemented in .cpp; opaque storage kept as vectors of floats is not enough —
    // torch::Tensor lives in the .cpp translation unit only via Impl.
#endif
    struct Impl;
    Impl *impl = nullptr;

    DepthEncoder();
    ~DepthEncoder();
    DepthEncoder(const DepthEncoder &) = delete;
    DepthEncoder &operator=(const DepthEncoder &) = delete;

    void Reset();
    void EnsureLoaded(const std::string &vision_weight_path);
    // depth_out_58x87: already preprocessed (meters/far - 0.5), length OutH*OutW.
    // proprio53: yaw slots 6:8 should be zeroed by caller before encode.
    // Writes latent[32] and yaw[2].
    void Encode(const std::vector<float> &depth_58x87,
                const std::vector<float> &proprio53,
                std::vector<float> &latent_out,
                std::vector<float> &yaw_out);
};

// Resize/crop RealSense (or any) metric depth toward the EP 58x87 training contract:
// resize -> 106x60, crop 58x98 (skip 4 left cols), bicubic -> 58x87, then d/far - 0.5.
void PreprocessDepthMeters(const float *src_meters, int src_h, int src_w,
                           std::vector<float> &out_58x87);

// Convert 16UC1 millimeters (scale 0.001) with clip [0, far].
void PreprocessDepthUint16Mm(const uint16_t *src_mm, int src_h, int src_w,
                             std::vector<float> &out_58x87, float far_m = kFarMeters);

std::vector<float> BuildObservation(const ProprioInput &in, HistState &hist);

// TCP depth frame protocol (Jetson bridge -> laptop):
// magic "EPDF" (4) | ver u32le=1 | w u32le | h u32le | stamp_ns u64le | payload u16le[w*h] millimeters
constexpr uint32_t kDepthProtoMagic = 0x46445045u; // 'EPDF' little-endian
constexpr uint32_t kDepthProtoVersion = 1;

class DepthTcpClient
{
public:
    DepthTcpClient();
    ~DepthTcpClient();

    // Non-blocking connect attempt; reconnects in background thread when host/port set.
    void Start(const std::string &host, int port);
    void Stop();

    // PEEK: copy latest frame without consuming seq. stamp_ns is the effective
    // (never-decreasing) stamp. Returns false if no frame has been accepted yet.
    bool TryGetLatestMm(std::vector<uint16_t> &out_mm, int &w, int &h, uint64_t &stamp_ns) const;
    bool TryGetLatestMm(std::vector<uint16_t> &out_mm, int &w, int &h,
                       uint64_t &stamp_ns, uint64_t &recv_ns, uint64_t &seq) const;
    bool PeekLatestMeta(int &w, int &h, uint64_t &stamp_ns, uint64_t &recv_ns, uint64_t &seq) const;
    uint64_t LatestRecvAgeNs() const; // ~0ull if no frame
    uint64_t Seq() const;
    bool Connected() const;

private:
    struct Impl;
    Impl *impl_;
};

} // namespace ep_student

#endif // EP_STUDENT_HPP
