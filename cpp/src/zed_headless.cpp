/**
 * zed_headless.cpp
 *
 * Headless ZED camera capture/streaming for Linux systems without an X server.
 *
 * Features
 * --------
 *   --shm      Write frames to POSIX shared memory (one segment per source)
 *   --rtsp     Stream via RTSP using GStreamer gst-rtsp-server
 *   --record   Record to video files using OpenCV VideoWriter
 *
 * A UTC timestamp is always burned into every frame before it is dispatched.
 *
 * Build dependencies
 * ------------------
 *   ZED SDK >= 3,  OpenCV >= 4,  GStreamer >= 1.16 with gst-rtsp-server,  CUDA
 *
 * Tested on Manjaro Linux (headless, no X server required).
 */

// ---- ZED SDK ----------------------------------------------------------------
#include <sl/Camera.hpp>

// ---- OpenCV -----------------------------------------------------------------
#include <opencv2/opencv.hpp>

// ---- GStreamer ---------------------------------------------------------------
#include <gst/gst.h>
#include <gst/app/gstappsrc.h>
#include <gst/rtsp-server/rtsp-server.h>

// ---- POSIX shared memory / IPC ----------------------------------------------
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <pthread.h>
#include <unistd.h>

// ---- Standard library -------------------------------------------------------
#include <getopt.h>
#include <csignal>
#include <atomic>
#include <thread>
#include <mutex>
#include <memory>
#include <stdexcept>
#include <vector>
#include <map>
#include <string>
#include <sstream>
#include <iostream>
#include <iomanip>
#include <cstring>
#include <ctime>
#include <chrono>

// =============================================================================
// Signal handling
// =============================================================================

static std::atomic<bool> g_running{true};

static void signal_handler(int) { g_running = false; }

// =============================================================================
// Source type
// =============================================================================

enum class Source { LEFT, RIGHT, DEPTH, STEREO, DISPARITY };

static const std::map<std::string, Source> kSourceMap = {
    {"left",      Source::LEFT},
    {"right",     Source::RIGHT},
    {"depth",     Source::DEPTH},
    {"stereo",    Source::STEREO},
    {"disparity", Source::DISPARITY},
};

static std::string source_name(Source s) {
    for (auto& kv : kSourceMap)
        if (kv.second == s) return kv.first;
    return "unknown";
}

// =============================================================================
// Configuration
// =============================================================================

struct Config {
    // Output modes
    bool enable_shm    = false;
    bool enable_rtsp   = false;
    bool enable_record = false;

    std::vector<Source> sources = {Source::LEFT};

    // Camera
    sl::RESOLUTION resolution = sl::RESOLUTION::HD720;
    int fps = 30;

    // RTSP
    int         rtsp_port    = 8554;
    std::string rtsp_codec   = "h264";  // h264 | h265 | mjpeg
    int         rtsp_bitrate = 4000;    // kbps (h264/h265 only)
    bool        rtsp_hw_enc  = false;   // try NVIDIA hardware encoder

    // Shared memory
    std::string shm_prefix = "/zed";    // /zed_left, /zed_right, …

    // Recording
    std::string record_dir    = "./recordings";
    std::string record_format = "mp4";  // mp4 | avi | mkv

    // Depth
    float depth_max_m = 10.0f;          // max metres for colour normalisation
};

// =============================================================================
// Utilities
// =============================================================================

static int64_t now_ns() {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
               std::chrono::system_clock::now().time_since_epoch()).count();
}

static std::string format_timestamp(int64_t ns) {
    int64_t sec  = ns / 1000000000LL;
    int64_t msec = (ns % 1000000000LL) / 1000000LL;
    std::time_t t = static_cast<std::time_t>(sec);
    std::tm *tm  = std::gmtime(&t);
    char buf[32];
    std::strftime(buf, sizeof(buf), "%Y-%m-%dT%H:%M:%S", tm);
    std::ostringstream oss;
    oss << buf << '.' << std::setw(3) << std::setfill('0') << msec << 'Z';
    return oss.str();
}

// Burn a white UTC timestamp string into the bottom-left corner of the frame.
static void draw_timestamp(cv::Mat& frame, int64_t ns) {
    const std::string ts = format_timestamp(ns);
    const int   font  = cv::FONT_HERSHEY_SIMPLEX;
    const double sc   = 0.55;
    const int   th    = 1;
    cv::Point pos(8, frame.rows - 8);
    // black shadow for legibility on any background
    cv::putText(frame, ts, {pos.x + 1, pos.y + 1},
                font, sc, {0, 0, 0}, th + 1, cv::LINE_AA);
    // white foreground
    cv::putText(frame, ts, pos,
                font, sc, {255, 255, 255}, th, cv::LINE_AA);
}

// Recursively create directories (POSIX, no std::filesystem needed).
static void makedirs(const std::string& path) {
    for (size_t i = 1; i <= path.size(); ++i) {
        if (i == path.size() || path[i] == '/') {
            std::string sub = path.substr(0, i);
            mkdir(sub.c_str(), 0755);   // silently ignores EEXIST
        }
    }
}

// =============================================================================
// ZED → OpenCV helpers
// =============================================================================

static cv::Mat sl2cv(sl::Mat& m) {
    int t;
    switch (m.getDataType()) {
        case sl::MAT_TYPE::U8_C1:  t = CV_8UC1;  break;
        case sl::MAT_TYPE::U8_C3:  t = CV_8UC3;  break;
        case sl::MAT_TYPE::U8_C4:  t = CV_8UC4;  break;
        case sl::MAT_TYPE::F32_C1: t = CV_32FC1; break;
        default:                   t = CV_8UC4;  break;
    }
    return cv::Mat(m.getHeight(), m.getWidth(), t,
                   m.getPtr<sl::uchar1>(sl::MEM::CPU),
                   m.getStepBytes(sl::MEM::CPU));
}

/**
 * Retrieve one frame for the given source and convert it to a BGR8 cv::Mat.
 * Also fills ts_out with a UTC nanosecond timestamp captured just before retrieval.
 * Returns false if retrieval fails.
 */
static bool capture_frame(sl::Camera& cam, Source src, const Config& cfg,
                           cv::Mat& bgr_out, int64_t& ts_out) {
    ts_out = now_ns();
    sl::Mat tmp;

    switch (src) {
        case Source::LEFT:
        case Source::RIGHT: {
            sl::VIEW view = (src == Source::LEFT) ? sl::VIEW::LEFT : sl::VIEW::RIGHT;
            if (cam.retrieveImage(tmp, view, sl::MEM::CPU) != sl::ERROR_CODE::SUCCESS)
                return false;
            cv::cvtColor(sl2cv(tmp), bgr_out, cv::COLOR_BGRA2BGR);
            break;
        }

        case Source::DEPTH: {
            // Float depth map → normalise to [0,255] → TURBO colourmap
            sl::Mat df;
            if (cam.retrieveMeasure(df, sl::MEASURE::DEPTH, sl::MEM::CPU) != sl::ERROR_CODE::SUCCESS)
                return false;
            cv::Mat d32 = sl2cv(df).clone();    // clone so we own the data
            cv::patchNaNs(d32, 0.0f);
            cv::min(d32, cfg.depth_max_m, d32); // clamp Inf / far values
            cv::Mat d8;
            d32.convertTo(d8, CV_8UC1, 255.0f / cfg.depth_max_m);
            cv::applyColorMap(d8, bgr_out, cv::COLORMAP_TURBO);
            break;
        }

        case Source::STEREO: {
            if (cam.retrieveImage(tmp, sl::VIEW::SIDE_BY_SIDE, sl::MEM::CPU) != sl::ERROR_CODE::SUCCESS)
                return false;
            cv::cvtColor(sl2cv(tmp), bgr_out, cv::COLOR_BGRA2BGR);
            break;
        }

        case Source::DISPARITY: {
            // ZED's coloured disparity visualisation (VIEW::DEPTH is the 8-bit coloured view)
            if (cam.retrieveImage(tmp, sl::VIEW::DEPTH, sl::MEM::CPU) != sl::ERROR_CODE::SUCCESS)
                return false;
            cv::cvtColor(sl2cv(tmp), bgr_out, cv::COLOR_BGRA2BGR);
            break;
        }
    }
    return true;
}

// =============================================================================
// Shared memory writer
// =============================================================================
//
// Memory layout of each shared segment:
//
//   [ ShmHeader ]                      (fixed size, includes process-shared mutex + condvar)
//   [ uint8_t[] ]                      (raw BGR8 frame: width × height × 3 bytes)
//
// Readers open the same segment with shm_open(O_RDONLY) and lock the mutex to
// safely copy the frame, or use frame_count to detect new frames.
//
// POSIX shared memory name: <prefix>_<source>   e.g. /zed_left, /zed_depth

struct ShmHeader {
    uint32_t        magic;          // 0x5A454443  ("ZEDC")
    uint32_t        version;        // 1
    int64_t         timestamp_ns;   // UTC nanoseconds since epoch
    uint64_t        frame_count;    // monotonically increasing
    int32_t         width;
    int32_t         height;
    int32_t         channels;       // always 3 (BGR)
    int32_t         data_size;      // bytes in data region
    pthread_mutex_t mtx;            // PTHREAD_PROCESS_SHARED
    pthread_cond_t  cond;           // signalled after each new frame
    // Immediately followed by: uint8_t data[data_size]
};

class ShmWriter {
public:
    ShmWriter(const std::string& name, int w, int h, int ch)
        : name_(name), w_(w), h_(h) {
        data_bytes_ = static_cast<size_t>(w) * h * ch;
        total_      = sizeof(ShmHeader) + data_bytes_;

        shm_unlink(name_.c_str());  // remove stale segment

        fd_ = shm_open(name_.c_str(), O_CREAT | O_RDWR, 0666);
        if (fd_ < 0)
            throw std::runtime_error("shm_open failed: " + name_);

        if (ftruncate(fd_, static_cast<off_t>(total_)) < 0) {
            close(fd_);
            throw std::runtime_error("ftruncate failed: " + name_);
        }

        hdr_ = static_cast<ShmHeader*>(
            mmap(nullptr, total_, PROT_READ | PROT_WRITE, MAP_SHARED, fd_, 0));
        if (hdr_ == MAP_FAILED) {
            close(fd_);
            throw std::runtime_error("mmap failed: " + name_);
        }

        hdr_->magic        = 0x5A454443u;
        hdr_->version      = 1;
        hdr_->width        = w_;
        hdr_->height       = h_;
        hdr_->channels     = ch;
        hdr_->data_size    = static_cast<int32_t>(data_bytes_);
        hdr_->frame_count  = 0;
        hdr_->timestamp_ns = 0;

        pthread_mutexattr_t ma;
        pthread_mutexattr_init(&ma);
        pthread_mutexattr_setpshared(&ma, PTHREAD_PROCESS_SHARED);
        pthread_mutex_init(&hdr_->mtx, &ma);
        pthread_mutexattr_destroy(&ma);

        pthread_condattr_t ca;
        pthread_condattr_init(&ca);
        pthread_condattr_setpshared(&ca, PTHREAD_PROCESS_SHARED);
        pthread_cond_init(&hdr_->cond, &ca);
        pthread_condattr_destroy(&ca);

        data_ = reinterpret_cast<uint8_t*>(hdr_) + sizeof(ShmHeader);

        std::cout << "[SHM] " << name_ << "  "
                  << w_ << "x" << h_ << "x" << ch
                  << "  (" << total_ << " bytes)\n";
    }

    ~ShmWriter() {
        if (hdr_ && hdr_ != MAP_FAILED) {
            pthread_mutex_destroy(&hdr_->mtx);
            pthread_cond_destroy(&hdr_->cond);
            munmap(hdr_, total_);
        }
        if (fd_ >= 0) close(fd_);
        shm_unlink(name_.c_str());
    }

    void write(const cv::Mat& frame, int64_t ts) {
        cv::Mat cont = frame.isContinuous() ? frame : frame.clone();
        size_t sz = static_cast<size_t>(cont.cols) * cont.rows * cont.channels();

        pthread_mutex_lock(&hdr_->mtx);
        hdr_->timestamp_ns = ts;
        hdr_->channels     = cont.channels();
        hdr_->data_size    = static_cast<int32_t>(sz);
        std::memcpy(data_, cont.data, sz);
        hdr_->frame_count++;
        pthread_cond_broadcast(&hdr_->cond);
        pthread_mutex_unlock(&hdr_->mtx);
    }

private:
    std::string name_;
    int    w_, h_;
    size_t data_bytes_, total_;
    int    fd_   = -1;
    ShmHeader* hdr_  = nullptr;
    uint8_t*   data_ = nullptr;
};

// =============================================================================
// RTSP streamer — GStreamer gst-rtsp-server with per-source appsrc
// =============================================================================
//
// Each source gets its own RTSP path:
//   rtsp://<host>:<port>/left
//   rtsp://<host>:<port>/right   etc.
//
// GStreamer pipeline (h264 example):
//   appsrc name=mysrc ! videoconvert ! x264enc tune=zerolatency … !
//   rtph264pay name=pay0 pt=96
//
// The appsrc caps are configured when the first RTSP client connects
// (media-configure signal). The pointer is released when all clients
// disconnect (media unprepared signal) so subsequent connections work cleanly.

struct SrcState {
    std::string path;
    int         width  = 0;
    int         height = 0;
    int         fps    = 30;
    int         port   = 8554;
    std::mutex  mu;
    GstAppSrc*  appsrc = nullptr;   // valid only while a client is connected
    uint64_t    pts    = 0;         // running presentation timestamp (ns)
    uint64_t    dur    = 0;         // ns per frame = GST_SECOND / fps
};

// Called when all RTSP clients disconnect — release the appsrc reference.
static void on_media_unprepared(GstRTSPMedia*, gpointer ud) {
    SrcState* ss = static_cast<SrcState*>(ud);
    std::lock_guard<std::mutex> lk(ss->mu);
    if (ss->appsrc) {
        gst_object_unref(ss->appsrc);
        ss->appsrc = nullptr;
    }
    ss->pts = 0;
}

// Called when the first RTSP client connects — configure appsrc caps.
static void on_media_configure(GstRTSPMediaFactory*, GstRTSPMedia* media, gpointer ud) {
    SrcState* ss   = static_cast<SrcState*>(ud);
    GstElement* pipe = gst_rtsp_media_get_element(media);
    GstElement* src  = gst_bin_get_by_name_recurse_up(GST_BIN(pipe), "mysrc");

    if (!src) {
        std::cerr << "[RTSP] appsrc element not found for path " << ss->path << "\n";
        gst_object_unref(pipe);
        return;
    }

    GstCaps* caps = gst_caps_new_simple("video/x-raw",
        "format",    G_TYPE_STRING,     "BGR",
        "width",     G_TYPE_INT,        ss->width,
        "height",    G_TYPE_INT,        ss->height,
        "framerate", GST_TYPE_FRACTION, ss->fps, 1,
        nullptr);

    g_object_set(src,
        "caps",         caps,
        "format",       GST_FORMAT_TIME,
        "is-live",      TRUE,
        "do-timestamp", FALSE,
        nullptr);
    gst_caps_unref(caps);

    // Reset PTS and store the (ref-counted) appsrc pointer
    {
        std::lock_guard<std::mutex> lk(ss->mu);
        if (ss->appsrc) gst_object_unref(ss->appsrc);
        ss->appsrc = GST_APP_SRC(src);   // src still holds one ref from get_by_name
        ss->pts    = 0;
    }

    // Release when all clients disconnect
    g_signal_connect(media, "unprepared", G_CALLBACK(on_media_unprepared), ud);

    std::cout << "[RTSP] Client connected → rtsp://0.0.0.0:"
              << ss->port << ss->path << "\n";
    gst_object_unref(pipe);
}

class RTSPStreamer {
public:
    explicit RTSPStreamer(const Config& cfg) : cfg_(cfg) {
        if (!gst_is_initialized()) gst_init(nullptr, nullptr);
        server_  = gst_rtsp_server_new();
        gst_rtsp_server_set_service(server_, std::to_string(cfg_.rtsp_port).c_str());
        mounts_  = gst_rtsp_server_get_mount_points(server_);
    }

    ~RTSPStreamer() {
        stop();
        for (auto& kv : states_) {
            std::lock_guard<std::mutex> lk(kv.second->mu);
            if (kv.second->appsrc) {
                gst_object_unref(kv.second->appsrc);
                kv.second->appsrc = nullptr;
            }
            delete kv.second;
        }
        if (mounts_) g_object_unref(mounts_);
        if (server_) g_object_unref(server_);
        if (loop_)   g_main_loop_unref(loop_);
    }

    void add_source(Source src, int w, int h) {
        auto* ss    = new SrcState();
        ss->path    = "/" + source_name(src);
        ss->width   = w;
        ss->height  = h;
        ss->fps     = cfg_.fps;
        ss->port    = cfg_.rtsp_port;
        ss->dur     = static_cast<uint64_t>(GST_SECOND) / static_cast<uint64_t>(cfg_.fps);

        std::string launch = build_launch();
        GstRTSPMediaFactory* fac = gst_rtsp_media_factory_new();
        gst_rtsp_media_factory_set_launch(fac, launch.c_str());
        gst_rtsp_media_factory_set_shared(fac, TRUE);
        g_signal_connect(fac, "media-configure", G_CALLBACK(on_media_configure), ss);
        gst_rtsp_mount_points_add_factory(mounts_, ss->path.c_str(), fac);
        g_object_unref(fac);

        states_[src] = ss;
        std::cout << "[RTSP] Path registered: rtsp://0.0.0.0:"
                  << cfg_.rtsp_port << ss->path << "\n";
    }

    void start() {
        if (!gst_rtsp_server_attach(server_, nullptr))
            throw std::runtime_error("RTSP server attach failed");
        loop_       = g_main_loop_new(nullptr, FALSE);
        glib_thread_ = std::thread([this] { g_main_loop_run(loop_); });
        std::cout << "[RTSP] Server listening on port " << cfg_.rtsp_port << "\n";
    }

    void stop() {
        if (loop_) g_main_loop_quit(loop_);
        if (glib_thread_.joinable()) glib_thread_.join();
    }

    void push(Source src, const cv::Mat& frame) {
        auto it = states_.find(src);
        if (it == states_.end()) return;

        SrcState* ss = it->second;
        std::lock_guard<std::mutex> lk(ss->mu);
        if (!ss->appsrc) return;    // no client connected yet

        // Ensure continuous BGR8 data
        cv::Mat bgr;
        if (!frame.isContinuous() || frame.type() != CV_8UC3)
            bgr = frame.clone();
        else
            bgr = frame;

        gsize sz = static_cast<gsize>(bgr.total()) * bgr.elemSize();
        GstBuffer* buf = gst_buffer_new_allocate(nullptr, sz, nullptr);
        gst_buffer_fill(buf, 0, bgr.data, sz);
        GST_BUFFER_PTS(buf)      = ss->pts;
        GST_BUFFER_DTS(buf)      = ss->pts;
        GST_BUFFER_DURATION(buf) = ss->dur;
        ss->pts += ss->dur;

        GstFlowReturn ret = gst_app_src_push_buffer(ss->appsrc, buf);
        if (ret != GST_FLOW_OK) {
            // Client gone; release ref (on_media_unprepared might not have fired yet)
            gst_object_unref(ss->appsrc);
            ss->appsrc = nullptr;
            ss->pts    = 0;
        }
    }

private:
    // Build the GStreamer launch description string based on codec choice.
    // The appsrc element is named "mysrc" so on_media_configure can find it.
    std::string build_launch() const {
        std::ostringstream o;
        o << "( appsrc name=mysrc ! videoconvert ! ";

        if (cfg_.rtsp_codec == "h264") {
            if (cfg_.rtsp_hw_enc)
                // NVIDIA hardware encoder — minimal CPU load, requires NVENC
                o << "nvh264enc preset=low-latency-hp zerolatency=true ! h264parse ! ";
            else
                // Software x264 — tune=zerolatency removes B-frames for low latency;
                // speed-preset=ultrafast keeps encoding CPU load minimal
                o << "x264enc tune=zerolatency bitrate=" << cfg_.rtsp_bitrate
                  << " speed-preset=ultrafast ! ";
            o << "rtph264pay name=pay0 pt=96";

        } else if (cfg_.rtsp_codec == "h265") {
            if (cfg_.rtsp_hw_enc)
                o << "nvh265enc zerolatency=true ! h265parse ! ";
            else
                o << "x265enc tune=zerolatency bitrate=" << cfg_.rtsp_bitrate
                  << " speed-preset=ultrafast ! ";
            o << "rtph265pay name=pay0 pt=96";

        } else if (cfg_.rtsp_codec == "mjpeg") {
            // MJPEG: every frame is a self-contained JPEG — easiest possible
            // decode on the receiver, at the cost of higher bandwidth.
            o << "jpegenc quality=85 ! rtpjpegpay name=pay0 pt=26";

        } else {
            throw std::runtime_error("Unknown RTSP codec: " + cfg_.rtsp_codec);
        }
        o << " )";
        return o.str();
    }

    const Config&       cfg_;
    GstRTSPServer*      server_      = nullptr;
    GstRTSPMountPoints* mounts_      = nullptr;
    GMainLoop*          loop_        = nullptr;
    std::thread         glib_thread_;
    std::map<Source, SrcState*> states_;
};

// =============================================================================
// Video recorder — OpenCV VideoWriter
// =============================================================================

class Recorder {
public:
    Recorder(const Config& cfg, Source src, int w, int h) {
        makedirs(cfg.record_dir);

        // Timestamp the filename so recordings never collide
        int64_t ns  = now_ns();
        int64_t sec = ns / 1000000000LL;
        std::time_t t  = static_cast<std::time_t>(sec);
        std::tm*   tm  = std::gmtime(&t);
        char ts[32];
        std::strftime(ts, sizeof(ts), "%Y%m%dT%H%M%SZ", tm);

        std::string path = cfg.record_dir + "/" + source_name(src) + "_"
                         + std::string(ts) + "." + cfg.record_format;

        // Prefer H.264; AVI containers typically use XVID
        int fourcc = (cfg.record_format == "avi")
            ? cv::VideoWriter::fourcc('X', 'V', 'I', 'D')
            : cv::VideoWriter::fourcc('X', '2', '6', '4');

        writer_.open(path, fourcc, cfg.fps, {w, h}, true);
        if (!writer_.isOpened()) {
            std::cerr << "[REC] H.264/XVID unavailable, falling back to MJPEG: "
                      << path << "\n";
            writer_.open(path, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'),
                         cfg.fps, {w, h}, true);
        }
        if (!writer_.isOpened())
            throw std::runtime_error("Cannot open VideoWriter: " + path);

        std::cout << "[REC] " << path << "\n";
    }

    void write(const cv::Mat& frame) {
        if (writer_.isOpened()) writer_.write(frame);
    }

private:
    cv::VideoWriter writer_;
};

// =============================================================================
// Resolution helpers
// =============================================================================

static sl::RESOLUTION parse_resolution(const std::string& s) {
    if (s == "HD2K")   return sl::RESOLUTION::HD2K;
    if (s == "HD1080") return sl::RESOLUTION::HD1080;
    if (s == "HD720")  return sl::RESOLUTION::HD720;
    if (s == "VGA")    return sl::RESOLUTION::VGA;
    throw std::runtime_error("Unknown resolution: '" + s + "'  (HD2K | HD1080 | HD720 | VGA)");
}

static std::string res_str(sl::RESOLUTION r) {
    switch (r) {
        case sl::RESOLUTION::HD2K:   return "HD2K (2208x1242)";
        case sl::RESOLUTION::HD1080: return "HD1080 (1920x1080)";
        case sl::RESOLUTION::HD720:  return "HD720 (1280x720)";
        case sl::RESOLUTION::VGA:    return "VGA (672x376)";
        default:                     return "unknown";
    }
}

// =============================================================================
// CLI
// =============================================================================

static void print_usage(const char* prog) {
    // clang-format off
    std::cout <<
"ZED Headless Camera Streamer\n"
"\n"
"Usage: " << prog << " [OPTIONS]\n"
"\n"
"At least one output mode (--shm / --rtsp / --record) must be enabled.\n"
"A UTC timestamp is always burned into each frame.\n"
"\n"
"OUTPUT MODES\n"
"  --shm              Write frames to POSIX shared memory\n"
"  --rtsp             Stream via RTSP (GStreamer gst-rtsp-server)\n"
"  --record           Record to video files on disk\n"
"\n"
"SOURCES  (comma-separated list, default: left)\n"
"  --sources=LIST     left, right, depth, stereo, disparity\n"
"                     Examples: --sources=left\n"
"                               --sources=left,right,depth\n"
"\n"
"CAMERA\n"
"  --resolution=R     HD2K | HD1080 | HD720 | VGA   (default: HD720)\n"
"  --fps=N            Frame rate: 15, 30, 60, 100    (default: 30)\n"
"  --depth-max=M      Max depth in metres used for depth colour normalisation\n"
"                     (default: 10.0)\n"
"\n"
"RTSP OPTIONS  (require --rtsp)\n"
"  --rtsp-port=N      Server port                    (default: 8554)\n"
"  --rtsp-codec=C     h264 | h265 | mjpeg            (default: h264)\n"
"\n"
"                     Codec trade-offs:\n"
"                       h264  — low bandwidth, baseline profile, hardware-\n"
"                               decodable on virtually every device / GPU.\n"
"                               Best overall choice for minimal receiver CPU.\n"
"                       h265  — ~40% less bandwidth than h264 at same quality;\n"
"                               requires newer hardware decoder on receiver.\n"
"                       mjpeg — highest bandwidth; each frame is an independent\n"
"                               JPEG with zero inter-frame dependencies and\n"
"                               absolute minimal decode complexity per frame.\n"
"\n"
"  --rtsp-bitrate=N   Target bitrate kbps for h264/h265 (default: 4000)\n"
"  --rtsp-hw          Use NVIDIA hardware encoder (nvh264enc / nvh265enc)\n"
"                     Requires NVENC-capable GPU (all ZED-supported GPUs qualify)\n"
"\n"
"SHARED MEMORY OPTIONS  (require --shm)\n"
"  --shm-prefix=S     Name prefix; creates /S_left, /S_right, …  (default: /zed)\n"
"\n"
"RECORDING OPTIONS  (require --record)\n"
"  --record-dir=D     Output directory                (default: ./recordings)\n"
"  --record-fmt=F     Container format: mp4 | avi | mkv (default: mp4)\n"
"                     Video codec inside container: X264 (H.264) or MJPEG fallback\n"
"\n"
"EXAMPLES\n"
"  # RTSP stream left camera at HD720 / 30 fps\n"
"  " << prog << " --rtsp --sources=left\n"
"\n"
"  # Shared memory for left + depth, viewed by another process\n"
"  " << prog << " --shm --sources=left,depth\n"
"\n"
"  # Record stereo side-by-side at 15 fps\n"
"  " << prog << " --record --sources=stereo --fps=15 --record-dir=/data/zed\n"
"\n"
"  # Full pipeline: SHM + RTSP (H.264) + recording, HD1080\n"
"  " << prog << " --shm --rtsp --record --sources=left,right,depth \\\n"
"    --resolution=HD1080 --fps=30 --rtsp-codec=h264 --rtsp-bitrate=6000\n"
"\n"
"  # RTSP with NVIDIA hardware encoder (lowest CPU on sender)\n"
"  " << prog << " --rtsp --rtsp-hw --rtsp-codec=h264 --sources=left,right\n"
"\n"
"RTSP STREAM URLS  (when --rtsp is active)\n"
"  rtsp://<host>:<port>/left\n"
"  rtsp://<host>:<port>/right\n"
"  rtsp://<host>:<port>/depth\n"
"  rtsp://<host>:<port>/stereo\n"
"  rtsp://<host>:<port>/disparity\n"
"\n"
"SHARED MEMORY LAYOUT  (when --shm is active)\n"
"  Segment names:  /zed_left,  /zed_right,  /zed_depth, …\n"
"  Open with:      shm_open(\"/zed_left\", O_RDONLY, 0)\n"
"\n"
"  Header at offset 0 (struct ShmHeader):\n"
"    uint32  magic         0x5A454443  (\"ZEDC\")\n"
"    uint32  version       1\n"
"    int64   timestamp_ns  UTC nanoseconds since Unix epoch\n"
"    uint64  frame_count   monotonically increasing counter\n"
"    int32   width, height, channels (always 3), data_size\n"
"    pthread_mutex_t  mtx   PTHREAD_PROCESS_SHARED\n"
"    pthread_cond_t   cond  broadcast after each new frame\n"
"\n"
"  Immediately after the header:\n"
"    uint8_t data[]         raw BGR8 bytes  (width × height × 3)\n"
"\n";
    // clang-format on
}

static Config parse_args(int argc, char* argv[]) {
    Config cfg;

    static const option long_opts[] = {
        {"shm",          no_argument,       nullptr,  1},
        {"rtsp",         no_argument,       nullptr,  2},
        {"record",       no_argument,       nullptr,  3},
        {"sources",      required_argument, nullptr,  4},
        {"resolution",   required_argument, nullptr,  5},
        {"fps",          required_argument, nullptr,  6},
        {"depth-max",    required_argument, nullptr,  7},
        {"rtsp-port",    required_argument, nullptr,  8},
        {"rtsp-codec",   required_argument, nullptr,  9},
        {"rtsp-bitrate", required_argument, nullptr, 10},
        {"rtsp-hw",      no_argument,       nullptr, 11},
        {"shm-prefix",   required_argument, nullptr, 12},
        {"record-dir",   required_argument, nullptr, 13},
        {"record-fmt",   required_argument, nullptr, 14},
        {"help",         no_argument,       nullptr, 'h'},
        {nullptr, 0, nullptr, 0}
    };

    int opt, idx;
    while ((opt = getopt_long(argc, argv, "h", long_opts, &idx)) != -1) {
        switch (opt) {
            case  1: cfg.enable_shm    = true; break;
            case  2: cfg.enable_rtsp   = true; break;
            case  3: cfg.enable_record = true; break;
            case  4: {
                cfg.sources.clear();
                std::istringstream ss(optarg);
                std::string tok;
                while (std::getline(ss, tok, ',')) {
                    // trim leading/trailing whitespace
                    auto b = tok.find_first_not_of(" \t");
                    auto e = tok.find_last_not_of(" \t");
                    if (b == std::string::npos) continue;
                    tok = tok.substr(b, e - b + 1);
                    auto it = kSourceMap.find(tok);
                    if (it == kSourceMap.end()) {
                        std::cerr << "[ERROR] Unknown source: '" << tok << "'\n"
                                  << "        Valid: left, right, depth, stereo, disparity\n";
                        exit(1);
                    }
                    cfg.sources.push_back(it->second);
                }
                break;
            }
            case  5: cfg.resolution    = parse_resolution(optarg); break;
            case  6: cfg.fps           = std::stoi(optarg); break;
            case  7: cfg.depth_max_m   = std::stof(optarg); break;
            case  8: cfg.rtsp_port     = std::stoi(optarg); break;
            case  9: cfg.rtsp_codec    = optarg; break;
            case 10: cfg.rtsp_bitrate  = std::stoi(optarg); break;
            case 11: cfg.rtsp_hw_enc   = true; break;
            case 12: cfg.shm_prefix    = optarg; break;
            case 13: cfg.record_dir    = optarg; break;
            case 14: cfg.record_format = optarg; break;
            case 'h': print_usage(argv[0]); exit(0);
            default:  print_usage(argv[0]); exit(1);
        }
    }
    return cfg;
}

// =============================================================================
// main
// =============================================================================

int main(int argc, char* argv[]) {
    std::signal(SIGINT,  signal_handler);
    std::signal(SIGTERM, signal_handler);

    Config cfg = parse_args(argc, argv);

    // ---- Validate: at least one output mode required ------------------------
    if (!cfg.enable_shm && !cfg.enable_rtsp && !cfg.enable_record) {
        std::cerr
            << "[ERROR] No output mode enabled — nothing to do.\n"
            << "        Specify at least one of:  --shm  --rtsp  --record\n"
            << "        Run '" << argv[0] << " --help' for full usage.\n";
        return 1;
    }

    if (cfg.sources.empty()) {
        std::cerr << "[ERROR] No sources specified. Use --sources=left,right,…\n";
        return 1;
    }

    // ---- Print configuration summary ----------------------------------------
    std::cout << "=== ZED Headless Streamer ===\n"
              << "Resolution : " << res_str(cfg.resolution) << "\n"
              << "FPS        : " << cfg.fps << "\n"
              << "Sources    :";
    for (auto s : cfg.sources) std::cout << " " << source_name(s);
    std::cout << "\nOutputs    :";
    if (cfg.enable_shm)    std::cout << " shm";
    if (cfg.enable_rtsp)   std::cout << " rtsp(:" << cfg.rtsp_port
                                     << "/" << cfg.rtsp_codec << ")";
    if (cfg.enable_record) std::cout << " record(" << cfg.record_dir << ")";
    std::cout << "\n\n";

    // ---- Open ZED camera ----------------------------------------------------
    sl::Camera zed;
    {
        sl::InitParameters ip;
        ip.camera_resolution      = cfg.resolution;
        ip.camera_fps             = cfg.fps;
        ip.depth_mode             = sl::DEPTH_MODE::ULTRA;
        ip.coordinate_units       = sl::UNIT::METER;
        ip.depth_minimum_distance = 0.3f;

        sl::ERROR_CODE err = zed.open(ip);
        if (err != sl::ERROR_CODE::SUCCESS) {
            std::cerr << "[ERROR] Cannot open ZED camera: " << sl::toString(err) << "\n";
            return 1;
        }
    }

    auto info  = zed.getCameraInformation();
    int  cam_w = info.camera_configuration.resolution.width;
    int  cam_h = info.camera_configuration.resolution.height;
    std::cout << "[ZED] Opened: " << cam_w << "x" << cam_h
              << " @ " << cfg.fps << " fps\n\n";

    // Helper: stereo is twice as wide, all others are standard camera size
    auto src_wh = [&](Source s) -> std::pair<int, int> {
        return (s == Source::STEREO)
            ? std::make_pair(cam_w * 2, cam_h)
            : std::make_pair(cam_w, cam_h);
    };

    // ---- Create output handlers ---------------------------------------------
    std::map<Source, std::unique_ptr<ShmWriter>> shm_writers;
    std::map<Source, std::unique_ptr<Recorder>>  recorders;
    std::unique_ptr<RTSPStreamer>                 rtsp;

    if (cfg.enable_shm) {
        for (auto s : cfg.sources) {
            auto wh = src_wh(s);
            std::string nm = cfg.shm_prefix + "_" + source_name(s);
            shm_writers[s] = std::make_unique<ShmWriter>(nm, wh.first, wh.second, 3);
        }
    }

    if (cfg.enable_rtsp) {
        rtsp = std::make_unique<RTSPStreamer>(cfg);
        for (auto s : cfg.sources) {
            auto wh = src_wh(s);
            rtsp->add_source(s, wh.first, wh.second);
        }
        rtsp->start();
    }

    if (cfg.enable_record) {
        for (auto s : cfg.sources) {
            auto wh = src_wh(s);
            recorders[s] = std::make_unique<Recorder>(cfg, s, wh.first, wh.second);
        }
    }

    std::cout << "\n[INFO] Capture running — press Ctrl+C to stop.\n\n";

    // ---- Main capture loop --------------------------------------------------
    sl::RuntimeParameters rtp;
    uint64_t frame_count = 0;

    while (g_running) {
        if (zed.grab(rtp) != sl::ERROR_CODE::SUCCESS) {
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
            continue;
        }
        ++frame_count;

        for (auto src : cfg.sources) {
            cv::Mat bgr;
            int64_t ts;

            if (!capture_frame(zed, src, cfg, bgr, ts)) {
                std::cerr << "[WARN] capture_frame failed for source: "
                          << source_name(src) << "\n";
                continue;
            }

            // Burn UTC timestamp onto every frame
            draw_timestamp(bgr, ts);

            if (cfg.enable_shm) {
                auto it = shm_writers.find(src);
                if (it != shm_writers.end()) it->second->write(bgr, ts);
            }

            if (cfg.enable_rtsp && rtsp)
                rtsp->push(src, bgr);

            if (cfg.enable_record) {
                auto it = recorders.find(src);
                if (it != recorders.end()) it->second->write(bgr);
            }
        }

        // Periodic console heartbeat (every 5 seconds at 30 fps)
        if (frame_count % static_cast<uint64_t>(cfg.fps * 5) == 0) {
            std::cout << "\r[INFO] frames=" << frame_count
                      << "  ts=" << format_timestamp(now_ns()) << std::flush;
        }
    }

    // ---- Shutdown -----------------------------------------------------------
    std::cout << "\n[INFO] Shutting down...\n";
    recorders.clear();
    rtsp.reset();
    shm_writers.clear();
    zed.close();
    std::cout << "[INFO] Done. Total frames captured: " << frame_count << "\n";
    return 0;
}
