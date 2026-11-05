/*
 * Shared Extreme Parkour Student helpers.
 */

#include "ep_student.hpp"

#include <algorithm>
#include <arpa/inet.h>
#include <atomic>
#include <chrono>
#include <cstring>
#include <fstream>
#include <iostream>
#include <netinet/in.h>
#include <stdexcept>
#include <thread>
#include <unistd.h>
#include <unordered_map>
#include "vector_math.hpp"

#ifdef USE_TORCH
#include <torch/torch.h>
#include <torch/csrc/jit/serialization/import_read.h>
#include <caffe2/serialize/inline_container.h>
#endif

namespace ep_student
{
namespace
{

float Cubic(float x)
{
    const float a = -0.75f;
    x = std::fabs(x);
    if (x <= 1.f)
    {
        return ((a + 2.f) * x - (a + 3.f)) * x * x + 1.f;
    }
    if (x < 2.f)
    {
        return ((a * x - 5.f * a) * x + 8.f * a) * x - 4.f * a;
    }
    return 0.f;
}

void ResizeBicubic(const std::vector<float> &src, int src_h, int src_w,
                   std::vector<float> &dst, int dst_h, int dst_w)
{
    dst.assign(dst_h * dst_w, 0.f);
    const float scale_x = static_cast<float>(src_w) / static_cast<float>(dst_w);
    const float scale_y = static_cast<float>(src_h) / static_cast<float>(dst_h);
    auto at = [&](int y, int x) -> float {
        y = std::max(0, std::min(src_h - 1, y));
        x = std::max(0, std::min(src_w - 1, x));
        return src[y * src_w + x];
    };
    for (int dy = 0; dy < dst_h; ++dy)
    {
        const float sy = (dy + 0.5f) * scale_y - 0.5f;
        const int iy = static_cast<int>(std::floor(sy));
        const float fy = sy - iy;
        const float wy[4] = {Cubic(fy + 1.f), Cubic(fy), Cubic(1.f - fy), Cubic(2.f - fy)};
        for (int dx = 0; dx < dst_w; ++dx)
        {
            const float sx = (dx + 0.5f) * scale_x - 0.5f;
            const int ix = static_cast<int>(std::floor(sx));
            const float fx = sx - ix;
            const float wx[4] = {Cubic(fx + 1.f), Cubic(fx), Cubic(1.f - fx), Cubic(2.f - fx)};
            float acc = 0.f;
            for (int ky = 0; ky < 4; ++ky)
            {
                for (int kx = 0; kx < 4; ++kx)
                {
                    acc += wy[ky] * wx[kx] * at(iy + ky - 1, ix + kx - 1);
                }
            }
            dst[dy * dst_w + dx] = acc;
        }
    }
}

void ResizeNearest(const std::vector<float> &src, int src_h, int src_w,
                   std::vector<float> &dst, int dst_h, int dst_w)
{
    dst.assign(dst_h * dst_w, 0.f);
    for (int dy = 0; dy < dst_h; ++dy)
    {
        const int sy = std::min(src_h - 1, (dy * src_h) / dst_h);
        for (int dx = 0; dx < dst_w; ++dx)
        {
            const int sx = std::min(src_w - 1, (dx * src_w) / dst_w);
            dst[dy * dst_w + dx] = src[sy * src_w + sx];
        }
    }
}

uint32_t ReadU32LE(const uint8_t *p)
{
    return static_cast<uint32_t>(p[0]) |
           (static_cast<uint32_t>(p[1]) << 8) |
           (static_cast<uint32_t>(p[2]) << 16) |
           (static_cast<uint32_t>(p[3]) << 24);
}

uint64_t ReadU64LE(const uint8_t *p)
{
    return static_cast<uint64_t>(ReadU32LE(p)) |
           (static_cast<uint64_t>(ReadU32LE(p + 4)) << 32);
}

bool RecvAll(int fd, void *buf, size_t n)
{
    auto *p = static_cast<uint8_t *>(buf);
    size_t got = 0;
    while (got < n)
    {
        const ssize_t r = ::recv(fd, p + got, n - got, 0);
        if (r <= 0)
        {
            return false;
        }
        got += static_cast<size_t>(r);
    }
    return true;
}

#ifdef USE_TORCH
void CollectTensors(const c10::IValue &value, std::unordered_map<std::string, torch::Tensor> &out)
{
    if (!value.isGenericDict())
    {
        return;
    }
    for (const auto &item : value.toGenericDict())
    {
        if (!item.key().isString())
        {
            continue;
        }
        const std::string key = item.key().toStringRef();
        if (item.value().isTensor())
        {
            out[key] = item.value().toTensor().to(torch::kFloat32).contiguous();
        }
        else
        {
            CollectTensors(item.value(), out);
        }
    }
}

const torch::Tensor &Need(const std::unordered_map<std::string, torch::Tensor> &w, const std::string &key)
{
    auto it = w.find(key);
    if (it == w.end())
    {
        throw std::runtime_error("missing vision weight " + key);
    }
    return it->second;
}
#endif

} // namespace

struct DepthEncoder::Impl
{
#ifdef USE_TORCH
    torch::Tensor conv1_w, conv1_b, conv2_w, conv2_b;
    torch::Tensor fc1_w, fc1_b, fc2_w, fc2_b;
    torch::Tensor comb1_w, comb1_b, comb2_w, comb2_b;
    torch::Tensor gru_w_ih, gru_w_hh, gru_b_ih, gru_b_hh;
    torch::Tensor out_w, out_b;
    torch::Tensor hidden;
#endif
};

DepthEncoder::DepthEncoder() : impl(new Impl) {}
DepthEncoder::~DepthEncoder() { delete impl; }

void DepthEncoder::Reset()
{
#ifdef USE_TORCH
    if (impl && impl->hidden.defined())
    {
        impl->hidden.zero_();
    }
#endif
}

namespace
{
std::string SiblingBinPath(const std::string &path)
{
    const std::string suf = ".pt";
    if (path.size() > suf.size() && path.compare(path.size() - suf.size(), suf.size(), suf) == 0)
    {
        return path.substr(0, path.size() - suf.size()) + ".bin";
    }
    if (path.size() > 4 && path.compare(path.size() - 4, 4, ".bin") == 0)
    {
        return path;
    }
    return path + ".bin";
}

bool FileReadable(const std::string &path)
{
    std::ifstream ifs(path, std::ios::binary);
    return static_cast<bool>(ifs);
}

std::unordered_map<std::string, torch::Tensor> LoadEpwtBin(const std::string &path)
{
    std::ifstream ifs(path, std::ios::binary);
    if (!ifs)
    {
        throw std::runtime_error("cannot open " + path);
    }
    char magic[4];
    ifs.read(magic, 4);
    if (ifs.gcount() != 4 || std::strncmp(magic, "EPWT", 4) != 0)
    {
        throw std::runtime_error("bad EPWT magic in " + path);
    }
    uint32_t ver = 0, n = 0;
    ifs.read(reinterpret_cast<char *>(&ver), 4);
    ifs.read(reinterpret_cast<char *>(&n), 4);
    if (ver != 1 || n == 0 || n > 256)
    {
        throw std::runtime_error("bad EPWT header in " + path);
    }
    std::unordered_map<std::string, torch::Tensor> out;
    for (uint32_t i = 0; i < n; ++i)
    {
        uint32_t name_len = 0;
        ifs.read(reinterpret_cast<char *>(&name_len), 4);
        if (name_len == 0 || name_len > 512)
        {
            throw std::runtime_error("bad tensor name length");
        }
        std::string name(name_len, '\0');
        ifs.read(name.data(), name_len);
        uint32_t ndim = 0;
        ifs.read(reinterpret_cast<char *>(&ndim), 4);
        if (ndim == 0 || ndim > 8)
        {
            throw std::runtime_error("bad ndim for " + name);
        }
        std::vector<int64_t> dims(ndim);
        int64_t numel = 1;
        for (uint32_t d = 0; d < ndim; ++d)
        {
            uint32_t dim = 0;
            ifs.read(reinterpret_cast<char *>(&dim), 4);
            dims[d] = static_cast<int64_t>(dim);
            numel *= dims[d];
        }
        std::vector<float> buf(static_cast<size_t>(numel));
        ifs.read(reinterpret_cast<char *>(buf.data()),
                 static_cast<std::streamsize>(numel * sizeof(float)));
        if (!ifs)
        {
            throw std::runtime_error("short read for " + name);
        }
        out[name] = torch::from_blob(buf.data(), dims, torch::kFloat32).clone();
    }
    return out;
}
} // namespace

void DepthEncoder::EnsureLoaded(const std::string &vision_weight_path)
{
#ifdef USE_TORCH
    if (ok)
    {
        return;
    }
    // Prefer C++-friendly EPWT binary (scripts/export_vision_weight_bin.py).
    // Raw torch.save(.pt) dicts are not reliably loadable via LibTorch unpickler.
    std::string load_path = vision_weight_path;
    const std::string bin_path = SiblingBinPath(vision_weight_path);
    std::unordered_map<std::string, torch::Tensor> weights;
    if (FileReadable(bin_path))
    {
        weights = LoadEpwtBin(bin_path);
        load_path = bin_path;
    }
    else if (FileReadable(vision_weight_path))
    {
        // Best-effort torch zip archive (works for some JIT/module dumps).
        caffe2::serialize::PyTorchStreamReader reader(vision_weight_path);
        std::string archive;
        bool has_root_data_pkl = false;
        for (const auto &rec : reader.getAllRecords())
        {
            if (rec == "data.pkl")
            {
                has_root_data_pkl = true;
            }
            const std::string suffix = "/data.pkl";
            if (rec.size() > suffix.size() &&
                rec.compare(rec.size() - suffix.size(), suffix.size(), suffix) == 0)
            {
                archive = rec.substr(0, rec.size() - suffix.size());
                break;
            }
        }
        if (archive.empty() && has_root_data_pkl)
        {
            archive = "data";
        }
        if (archive.empty())
        {
            throw std::runtime_error(
                "no loadable vision weights; run scripts/export_vision_weight_bin.py to create " +
                bin_path);
        }
        c10::IValue loaded = torch::jit::readArchiveAndTensors(
            archive, "", "data/", c10::nullopt, c10::nullopt, c10::Device(c10::kCPU), reader);
        CollectTensors(loaded, weights);
        if (weights.count("depth_encoder_state_dict") == 0 &&
            weights.count("base_backbone.image_compression.0.weight") == 0)
        {
            // Nested dict under depth_encoder_state_dict may appear as empty after unpickle.
            throw std::runtime_error(
                "torch archive did not yield encoder tensors; export " + bin_path +
                " with scripts/export_vision_weight_bin.py");
        }
    }
    else
    {
        throw std::runtime_error("vision weight not found: " + vision_weight_path);
    }

    auto take = [&](const std::string &key) { return Need(weights, key); };
    impl->conv1_w = take("base_backbone.image_compression.0.weight");
    impl->conv1_b = take("base_backbone.image_compression.0.bias");
    impl->conv2_w = take("base_backbone.image_compression.3.weight");
    impl->conv2_b = take("base_backbone.image_compression.3.bias");
    impl->fc1_w = take("base_backbone.image_compression.6.weight");
    impl->fc1_b = take("base_backbone.image_compression.6.bias");
    impl->fc2_w = take("base_backbone.image_compression.8.weight");
    impl->fc2_b = take("base_backbone.image_compression.8.bias");
    impl->comb1_w = take("combination_mlp.0.weight");
    impl->comb1_b = take("combination_mlp.0.bias");
    impl->comb2_w = take("combination_mlp.2.weight");
    impl->comb2_b = take("combination_mlp.2.bias");
    impl->gru_w_ih = take("rnn.weight_ih_l0");
    impl->gru_w_hh = take("rnn.weight_hh_l0");
    impl->gru_b_ih = take("rnn.bias_ih_l0");
    impl->gru_b_hh = take("rnn.bias_hh_l0");
    impl->out_w = take("output_mlp.0.weight");
    impl->out_b = take("output_mlp.0.bias");
    impl->hidden = torch::zeros({1, 1, 512}, torch::kFloat32);
    ok = true;
    std::cout << "[EP_DEPTH] loaded vision encoder from " << load_path << std::endl;
#else
    (void)vision_weight_path;
    throw std::runtime_error("libtorch is not enabled");
#endif
}

void DepthEncoder::Encode(const std::vector<float> &depth_58x87,
                          const std::vector<float> &proprio53,
                          std::vector<float> &latent_out,
                          std::vector<float> &yaw_out)
{
#ifdef USE_TORCH
    if (!ok)
    {
        throw std::runtime_error("DepthEncoder not loaded");
    }
    if ((int)depth_58x87.size() != kOutH * kOutW)
    {
        throw std::runtime_error("depth size must be 58*87");
    }
    if ((int)proprio53.size() < kNProprio)
    {
        throw std::runtime_error("proprio size must be >= 53");
    }

    torch::NoGradGuard guard;
    auto img = torch::from_blob(const_cast<float *>(depth_58x87.data()),
                                {1, 1, kOutH, kOutW}, torch::kFloat32)
                   .clone();
    auto x = at::elu(at::max_pool2d(at::conv2d(img, impl->conv1_w, impl->conv1_b), {2, 2}, {2, 2}));
    x = at::elu(at::conv2d(x, impl->conv2_w, impl->conv2_b));
    x = x.flatten(1);
    x = at::elu(at::linear(x, impl->fc1_w, impl->fc1_b));
    x = at::elu(at::linear(x, impl->fc2_w, impl->fc2_b));
    auto proprio = torch::from_blob(const_cast<float *>(proprio53.data()), {1, kNProprio}, torch::kFloat32).clone();
    proprio.narrow(1, 6, 2).zero_();
    auto cat = at::cat({x, proprio}, 1);
    auto comb = at::elu(at::linear(cat, impl->comb1_w, impl->comb1_b));
    comb = at::linear(comb, impl->comb2_w, impl->comb2_b);
    if (!impl->hidden.defined())
    {
        impl->hidden = torch::zeros({1, 1, 512}, torch::kFloat32);
    }
    auto gru_out = at::gru(
        comb.view({1, 1, 32}), impl->hidden,
        {impl->gru_w_ih, impl->gru_w_hh, impl->gru_b_ih, impl->gru_b_hh},
        true, 1, 0.0, false, false, true);
    impl->hidden = std::get<1>(gru_out);
    auto y = at::tanh(at::linear(std::get<0>(gru_out).view({1, 512}), impl->out_w, impl->out_b));
    auto acc = y.view({34}).contiguous();
    const float *p = acc.data_ptr<float>();
    latent_out.assign(p, p + kLatent);
    yaw_out.assign(p + kLatent, p + kLatent + 2);
#else
    (void)depth_58x87;
    (void)proprio53;
    latent_out.assign(kLatent, 0.0f);
    yaw_out.assign(2, 0.0f);
    throw std::runtime_error("libtorch is not enabled");
#endif
}

void PreprocessDepthMeters(const float *src_meters, int src_h, int src_w,
                           std::vector<float> &out_58x87)
{
    if (!src_meters || src_h <= 0 || src_w <= 0)
    {
        out_58x87.assign(kOutH * kOutW, -0.5f);
        return;
    }
    std::vector<float> src(src_meters, src_meters + src_h * src_w);
    for (float &d : src)
    {
        if (!(d > 0.0f))
        {
            d = kFarMeters;
        }
        if (d > kFarMeters)
        {
            d = kFarMeters;
        }
        if (d < 0.0f)
        {
            d = 0.0f;
        }
    }

    std::vector<float> resized_106;
    // First hop to training ray grid size (106x60). Use nearest for speed on large RealSense frames;
    // crop/resize to 58x87 still uses bicubic like MuJoCo path.
    ResizeNearest(src, src_h, src_w, resized_106, kDepthH, kDepthW);

    std::vector<float> cropped(kCropH * kCropW);
    for (int v = 0; v < kCropH; ++v)
    {
        for (int u = 0; u < kCropW; ++u)
        {
            cropped[v * kCropW + u] = resized_106[v * kDepthW + (u + 4)];
        }
    }
    ResizeBicubic(cropped, kCropH, kCropW, out_58x87, kOutH, kOutW);
    for (float &d : out_58x87)
    {
        d = d / kFarMeters - 0.5f;
    }
}

void PreprocessDepthUint16Mm(const uint16_t *src_mm, int src_h, int src_w,
                             std::vector<float> &out_58x87, float far_m)
{
    std::vector<float> meters(src_h * src_w);
    const float scale = 0.001f;
    for (int i = 0; i < src_h * src_w; ++i)
    {
        float m = static_cast<float>(src_mm[i]) * scale;
        if (!(m > 0.0f))
        {
            m = far_m;
        }
        if (m > far_m)
        {
            m = far_m;
        }
        meters[i] = m;
    }
    // Temporarily override far via local far_m: PreprocessDepthMeters uses kFarMeters.
    // If far_m differs, scale after; for EP far is always 2.0.
    (void)far_m;
    PreprocessDepthMeters(meters.data(), src_h, src_w, out_58x87);
}

std::vector<float> BuildObservation(const ProprioInput &in, HistState &hist)
{
    std::vector<float> proprio(kNProprio, 0.0f);

    if (in.ang_vel.size() >= 3)
    {
        proprio[0] = in.ang_vel[0] * in.scales.ang_vel;
        proprio[1] = in.ang_vel[1] * in.scales.ang_vel;
        proprio[2] = in.ang_vel[2] * in.scales.ang_vel;
    }
    if (in.base_quat.size() >= 4)
    {
        std::vector<float> euler = QuaternionToEuler(in.base_quat);
        proprio[3] = euler[0];
        proprio[4] = euler[1];
    }
    proprio[10] = in.cmd_vx;
    proprio[11] = in.terrain_flat ? 0.0f : 1.0f;
    proprio[12] = in.terrain_flat ? 1.0f : 0.0f;

    std::vector<float> last = in.last_action;
    if ((int)last.size() != 12)
    {
        last.assign(12, 0.0f);
    }
    for (int i = 0; i < 12; ++i)
    {
        float q0 = (i < (int)in.default_pos.size()) ? in.default_pos[i] : 0.0f;
        float q = (i < (int)in.dof_pos.size()) ? in.dof_pos[i] : q0;
        float dq = (i < (int)in.dof_vel.size()) ? in.dof_vel[i] : 0.0f;
        proprio[13 + i] = (q - q0) * in.scales.dof_pos;
        proprio[25 + i] = dq * in.scales.dof_vel;
        proprio[37 + i] = last[i];
    }
    for (int i = 0; i < 4; ++i)
    {
        proprio[49 + i] = -0.5f;
    }

    std::vector<float> hist_frame = proprio;
    hist_frame[6] = 0.0f;
    hist_frame[7] = 0.0f;

    if (!hist.ready || (int)hist.proprio_hist.size() != kHistLen * kNProprio)
    {
        hist.proprio_hist.assign(kHistLen * kNProprio, 0.0f);
        for (int h = 0; h < kHistLen; ++h)
        {
            std::copy(hist_frame.begin(), hist_frame.end(), hist.proprio_hist.begin() + h * kNProprio);
        }
        hist.ready = true;
    }
    else
    {
        std::copy(hist.proprio_hist.begin() + kNProprio, hist.proprio_hist.end(), hist.proprio_hist.begin());
        std::copy(hist_frame.begin(), hist_frame.end(), hist.proprio_hist.end() - kNProprio);
    }

    std::vector<float> ep_obs(kNObs, 0.0f);
    std::copy(proprio.begin(), proprio.end(), ep_obs.begin());
    const int hist_off = kNProprio + kNScan + kNPrivE + kNPrivL;
    std::copy(hist.proprio_hist.begin(), hist.proprio_hist.end(), ep_obs.begin() + hist_off);

    const float clip = in.scales.clip_obs;
    for (float &v : ep_obs)
    {
        if (v > clip)
            v = clip;
        if (v < -clip)
            v = -clip;
    }
    return ep_obs;
}

namespace
{
uint64_t MonoNs()
{
    using namespace std::chrono;
    return static_cast<uint64_t>(duration_cast<nanoseconds>(steady_clock::now().time_since_epoch()).count());
}
} // namespace

struct DepthTcpClient::Impl
{
    std::string host;
    int port = 0;
    std::atomic<bool> run{false};
    std::atomic<bool> connected{false};
    std::thread thr;
    mutable std::mutex mu;
    std::vector<uint16_t> latest;
    int w = 0;
    int h = 0;
    uint64_t wire_stamp_ns = 0;
    uint64_t effective_stamp_ns = 0;
    uint64_t recv_ns = 0;
    uint64_t seq = 0;
    uint64_t last_wire_stamp = 0;
    uint64_t last_effective_stamp = 0;
    uint64_t last_recv_ns = 0;
    int log_accepts_left = 8;
};

DepthTcpClient::DepthTcpClient() : impl_(new Impl) {}
DepthTcpClient::~DepthTcpClient()
{
    Stop();
    delete impl_;
}

bool DepthTcpClient::Connected() const { return impl_->connected.load(); }

uint64_t DepthTcpClient::Seq() const
{
    std::lock_guard<std::mutex> lock(impl_->mu);
    return impl_->seq;
}

void DepthTcpClient::Start(const std::string &host, int port)
{
    Stop();
    impl_->host = host;
    impl_->port = port;
    impl_->run = true;
    impl_->thr = std::thread([this]() {
        while (impl_->run.load())
        {
            int fd = ::socket(AF_INET, SOCK_STREAM, 0);
            if (fd < 0)
            {
                std::this_thread::sleep_for(std::chrono::seconds(1));
                continue;
            }
            sockaddr_in addr{};
            addr.sin_family = AF_INET;
            addr.sin_port = htons(static_cast<uint16_t>(impl_->port));
            if (::inet_pton(AF_INET, impl_->host.c_str(), &addr.sin_addr) != 1)
            {
                ::close(fd);
                std::cerr << "[EP_DEPTH_TCP] bad host " << impl_->host << std::endl;
                std::this_thread::sleep_for(std::chrono::seconds(2));
                continue;
            }
            if (::connect(fd, reinterpret_cast<sockaddr *>(&addr), sizeof(addr)) != 0)
            {
                ::close(fd);
                impl_->connected = false;
                std::this_thread::sleep_for(std::chrono::seconds(1));
                continue;
            }
            impl_->connected = true;
            {
                std::lock_guard<std::mutex> lock(impl_->mu);
                impl_->last_wire_stamp = 0;
                impl_->last_effective_stamp = 0;
                impl_->last_recv_ns = 0;
            }
            std::cout << "[EP_DEPTH_TCP] connected to " << impl_->host << ":" << impl_->port << std::endl;

            while (impl_->run.load())
            {
                uint8_t hdr[24];
                if (!RecvAll(fd, hdr, sizeof(hdr)))
                {
                    break;
                }
                const uint32_t magic = ReadU32LE(hdr + 0);
                const uint32_t ver = ReadU32LE(hdr + 4);
                const uint32_t w = ReadU32LE(hdr + 8);
                const uint32_t h = ReadU32LE(hdr + 12);
                const uint64_t wire = ReadU64LE(hdr + 16);
                if (magic != kDepthProtoMagic || ver != kDepthProtoVersion || w == 0 || h == 0 ||
                    w > 4096 || h > 4096)
                {
                    std::cerr << "[EP_DEPTH_TCP] bad header magic/ver/size" << std::endl;
                    break;
                }
                const size_t nbytes = static_cast<size_t>(w) * static_cast<size_t>(h) * sizeof(uint16_t);
                std::vector<uint16_t> frame(static_cast<size_t>(w) * static_cast<size_t>(h));
                if (!RecvAll(fd, frame.data(), nbytes))
                {
                    break;
                }
                const uint64_t now = MonoNs();
                {
                    std::lock_guard<std::mutex> lock(impl_->mu);
                    const bool stamp_up = wire > impl_->last_wire_stamp;
                    const bool recv_advanced = now > impl_->last_recv_ns;
                    // Accept if stamp increased OR this is a new TCP packet with newer recv time
                    // (covers frozen ROS stamps). Stamp rollback = bridge restart; also accept.
                    if (!stamp_up && !recv_advanced)
                    {
                        continue;
                    }
                    uint64_t effective = wire;
                    if (!stamp_up)
                    {
                        effective = now;
                    }
                    if (effective <= impl_->last_effective_stamp)
                    {
                        effective = impl_->last_effective_stamp + 1;
                    }
                    impl_->latest.swap(frame);
                    impl_->w = static_cast<int>(w);
                    impl_->h = static_cast<int>(h);
                    impl_->wire_stamp_ns = wire;
                    impl_->effective_stamp_ns = effective;
                    impl_->recv_ns = now;
                    impl_->last_wire_stamp = wire;
                    impl_->last_effective_stamp = effective;
                    impl_->last_recv_ns = now;
                    impl_->seq += 1;
                    const bool frozen = !stamp_up;
                    if (impl_->log_accepts_left > 0 || (impl_->seq % 30 == 0))
                    {
                        if (impl_->log_accepts_left > 0)
                        {
                            --impl_->log_accepts_left;
                        }
                        std::cerr << "[EP_DEPTH] tcp frame " << w << "x" << h
                                  << " stamp_ns=" << effective
                                  << " wire_stamp_ns=" << wire
                                  << " seq=" << impl_->seq
                                  << " frozen_ros=" << (frozen ? 1 : 0) << std::endl;
                    }
                }
            }
            ::close(fd);
            impl_->connected = false;
            std::cerr << "[EP_DEPTH_TCP] disconnected; retrying..." << std::endl;
            std::this_thread::sleep_for(std::chrono::seconds(1));
        }
    });
}

void DepthTcpClient::Stop()
{
    impl_->run = false;
    if (impl_->thr.joinable())
    {
        impl_->thr.join();
    }
    impl_->connected = false;
}

bool DepthTcpClient::TryGetLatestMm(std::vector<uint16_t> &out_mm, int &w, int &h, uint64_t &stamp_ns) const
{
    uint64_t recv_ns = 0;
    uint64_t seq = 0;
    return TryGetLatestMm(out_mm, w, h, stamp_ns, recv_ns, seq);
}

bool DepthTcpClient::TryGetLatestMm(std::vector<uint16_t> &out_mm, int &w, int &h,
                                   uint64_t &stamp_ns, uint64_t &recv_ns, uint64_t &seq) const
{
    std::lock_guard<std::mutex> lock(impl_->mu);
    if (impl_->seq == 0 || impl_->latest.empty())
    {
        return false;
    }
    out_mm = impl_->latest;
    w = impl_->w;
    h = impl_->h;
    stamp_ns = impl_->effective_stamp_ns;
    recv_ns = impl_->recv_ns;
    seq = impl_->seq;
    return true;
}

bool DepthTcpClient::PeekLatestMeta(int &w, int &h, uint64_t &stamp_ns, uint64_t &recv_ns, uint64_t &seq) const
{
    std::lock_guard<std::mutex> lock(impl_->mu);
    if (impl_->seq == 0 || impl_->latest.empty())
    {
        return false;
    }
    w = impl_->w;
    h = impl_->h;
    stamp_ns = impl_->effective_stamp_ns;
    recv_ns = impl_->recv_ns;
    seq = impl_->seq;
    return true;
}

uint64_t DepthTcpClient::LatestRecvAgeNs() const
{
    std::lock_guard<std::mutex> lock(impl_->mu);
    if (impl_->seq == 0 || impl_->recv_ns == 0)
    {
        return ~0ull;
    }
    const uint64_t now = MonoNs();
    return now >= impl_->recv_ns ? now - impl_->recv_ns : 0;
}

} // namespace ep_student
