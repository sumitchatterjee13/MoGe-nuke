// MoGe3 OFX plugin: a pull-based Nuke node that runs the COMPLETE MoGe-3
// model (Triton sparse refiner included) via a resident Python daemon.
//
// The plugin itself holds no model and no torch. Per render it takes the full
// source frame, sends it to moge_daemon.py over localhost TCP, and writes the
// chosen pass (depth or normals, mask in alpha) into the output image. The
// reply is cached per instance so switching the output dropdown, or Nuke
// re-requesting the same frame, costs nothing.
//
// Raw OFX C API (no Support library) so the only dependency is the open
// header set in ../include/openfx.
//
// Build: see ../CMakeLists.txt / ../build.ps1 (or use ../prebuilt)

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#define _CRT_SECURE_NO_WARNINGS
#include <winsock2.h>
#include <ws2tcpip.h>
#include <windows.h>
typedef SOCKET sock_t;
#else
#include <arpa/inet.h>
#include <dlfcn.h>
#include <errno.h>
#include <fcntl.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <signal.h>
#include <spawn.h>
#include <sys/select.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>
typedef int sock_t;
#define INVALID_SOCKET (-1)
#define SOCKET_ERROR (-1)
extern char **environ;
#endif

#include <algorithm>
#include <cmath>
#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>
#include <vector>

#include "ofxCore.h"
#include "ofxImageEffect.h"
#include "ofxMemory.h"
#include "ofxMessage.h"
#include "ofxParam.h"
#include "ofxProgress.h"
#include "ofxProperty.h"

#ifdef _WIN32
#pragma comment(lib, "ws2_32.lib")
#endif

#define PLUGIN_ID "com.sumit.moge3"
#define PLUGIN_LABEL "MoGe3"
#define PLUGIN_VERSION_MAJOR 1
#define PLUGIN_VERSION_MINOR 0

// Where the repo (venv, daemon, models) lives is discovered at load time, in
// this order:
//   1. MOGE_NUKE_ROOT environment variable
//   2. moge3.cfg next to the .ofx (written by install.ps1): key=value lines
//      root=..., and optional python=, daemon=, model=, port=
//   3. nothing -> the Setup tab has to be filled in by hand
static std::string gRoot, gDefaultModel, gDefaultPython, gDefaultDaemon;
static const char *kHfModel = "Ruicheng/moge-3-vitg";  // downloaded by the daemon if no local file
static int kDefaultPort = 47821;

// Refiner step choices: index == number of steps. Keep in sync with
// kRefineLabels below.
static const char *kRefineLabels[] = {
    "0  (off - same as the live .cat node)",
    "1",
    "2",
    "3  (MoGe default)",
    "4",
    "5",
};
static const int kRefineCount = sizeof(kRefineLabels) / sizeof(kRefineLabels[0]);
static const int kRefineDefault = 3;

// ---------------------------------------------------------------------------
// host suites
// ---------------------------------------------------------------------------

static OfxHost *gHost = nullptr;
static OfxImageEffectSuiteV1 *gEffect = nullptr;
static OfxPropertySuiteV1 *gProp = nullptr;
static OfxParameterSuiteV1 *gParam = nullptr;
static OfxMessageSuiteV1 *gMessage = nullptr;
static OfxProgressSuiteV1 *gProgress = nullptr;

static void logf(const char *fmt, ...) {
    char buf[2048];
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(buf, sizeof(buf), fmt, ap);
    va_end(ap);
    fprintf(stderr, "[MoGe3 OFX] %s\n", buf);
    fflush(stderr);
}

// ---------------------------------------------------------------------------
// parameters
// ---------------------------------------------------------------------------

struct Params {
    std::string model;
    int output = 0;          // 0 depth, 1 normals
    int refineSteps = kRefineDefault;
    int resolutionLevel = 9;
    int numTokens = 0;
    double fovX = 0.0;
    int fp16 = 0;
    int colorspace = 0;      // 0 scene-linear, 1 sRGB
    int normalSpace = 0;     // 0 nuke, 1 opencv
    int applyMask = 1;
    std::string python;
    std::string daemon;
    int port = kDefaultPort;
    int autoStart = 1;
    int exitWithHost = 1;

    // Everything that changes what the daemon computes (NOT `output`, which
    // only picks planes from a cached reply).
    bool sameInference(const Params &o) const {
        return model == o.model && refineSteps == o.refineSteps &&
               resolutionLevel == o.resolutionLevel && numTokens == o.numTokens &&
               fovX == o.fovX && fp16 == o.fp16 && colorspace == o.colorspace &&
               normalSpace == o.normalSpace && applyMask == o.applyMask;
    }
};

struct Instance {
    OfxImageEffectHandle effect = nullptr;
    OfxImageClipHandle source = nullptr;
    OfxImageClipHandle output = nullptr;
    OfxParamHandle pModel, pOutput, pRefine, pResLevel, pTokens, pFov, pFp16,
        pColorspace, pNormalSpace, pApplyMask, pPython, pDaemon, pPort, pAutoStart,
        pExitWithHost;

    sock_t sock = INVALID_SOCKET;

    // cache of the last daemon reply
    bool cacheValid = false;
    Params cacheParams;
    int cacheW = 0, cacheH = 0;
    uint64_t cacheHash = 0;
    std::vector<float> cachePlanes;  // 5 * W * H, top row first
};

static std::mutex gSpawnMutex;

// ---------------------------------------------------------------------------
// small helpers
// ---------------------------------------------------------------------------

static std::string jsonEscape(const std::string &s) {
    std::string out;
    out.reserve(s.size() + 8);
    for (char c : s) {
        switch (c) {
        case '"': out += "\\\""; break;
        case '\\': out += "\\\\"; break;
        case '\n': out += "\\n"; break;
        case '\r': out += "\\r"; break;
        case '\t': out += "\\t"; break;
        default:
            if ((unsigned char)c < 0x20) {
                char b[8];
                snprintf(b, sizeof(b), "\\u%04x", (unsigned char)c);
                out += b;
            } else {
                out += c;
            }
        }
    }
    return out;
}

static uint64_t fnv1a(const void *data, size_t n) {
    const unsigned char *p = (const unsigned char *)data;
    uint64_t h = 1469598103934665603ULL;
    for (size_t i = 0; i < n; ++i) {
        h ^= p[i];
        h *= 1099511628211ULL;
    }
    return h;
}

#ifdef _WIN32
static std::wstring widen(const std::string &s) {
    if (s.empty()) return L"";
    int n = MultiByteToWideChar(CP_UTF8, 0, s.c_str(), (int)s.size(), nullptr, 0);
    std::wstring w(n, 0);
    MultiByteToWideChar(CP_UTF8, 0, s.c_str(), (int)s.size(), &w[0], n);
    return w;
}
#endif

// ---------------------------------------------------------------------------
// platform layer
// ---------------------------------------------------------------------------

static void closeSock(sock_t s) {
#ifdef _WIN32
    closesocket(s);
#else
    close(s);
#endif
}

static void sleepMs(int ms) {
#ifdef _WIN32
    Sleep(ms);
#else
    usleep(ms * 1000);
#endif
}

static bool fileExists(const std::string &p) {
    if (p.empty()) return false;
#ifdef _WIN32
    return GetFileAttributesW(widen(p).c_str()) != INVALID_FILE_ATTRIBUTES;
#else
    struct stat st;
    return stat(p.c_str(), &st) == 0;
#endif
}

static long currentPid() {
#ifdef _WIN32
    return (long)GetCurrentProcessId();
#else
    return (long)getpid();
#endif
}

static std::string envVar(const char *name) {
#ifdef _WIN32
    wchar_t buf[4096];
    DWORD n = GetEnvironmentVariableW(widen(name).c_str(), buf, 4096);
    if (n == 0 || n >= 4096) return "";
    int m = WideCharToMultiByte(CP_UTF8, 0, buf, (int)n, nullptr, 0, nullptr, nullptr);
    std::string out(m, 0);
    WideCharToMultiByte(CP_UTF8, 0, buf, (int)n, &out[0], m, nullptr, nullptr);
    return out;
#else
    const char *v = getenv(name);
    return v ? v : "";
#endif
}

// Directory containing this .ofx.
static std::string pluginDir() {
    std::string path;
#ifdef _WIN32
    HMODULE hm = nullptr;
    GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                       (LPCWSTR)&pluginDir, &hm);
    wchar_t buf[4096];
    DWORD n = GetModuleFileNameW(hm, buf, 4096);
    if (n == 0) return "";
    int m = WideCharToMultiByte(CP_UTF8, 0, buf, (int)n, nullptr, 0, nullptr, nullptr);
    path.assign(m, 0);
    WideCharToMultiByte(CP_UTF8, 0, buf, (int)n, &path[0], m, nullptr, nullptr);
#else
    Dl_info info;
    if (dladdr((void *)&pluginDir, &info) == 0 || !info.dli_fname) return "";
    path = info.dli_fname;
#endif
    size_t cut = path.find_last_of("/\\");
    return cut == std::string::npos ? "" : path.substr(0, cut);
}

static FILE *openRead(const std::string &p) {
#ifdef _WIN32
    return _wfopen(widen(p).c_str(), L"rb");
#else
    return fopen(p.c_str(), "rb");
#endif
}

static void showError(Instance *inst, const std::string &msg) {
    logf("ERROR: %s", msg.c_str());
    if (gMessage && inst && inst->effect)
        gMessage->message(inst->effect, kOfxMessageError, nullptr, "MoGe3: %s", msg.c_str());
}

static std::string getStringParam(OfxParamHandle h, double t) {
    char *v = nullptr;
    if (gParam->paramGetValueAtTime(h, t, &v) != kOfxStatOK || !v) return "";
    return std::string(v);
}

static int getIntParam(OfxParamHandle h, double t) {
    int v = 0;
    gParam->paramGetValueAtTime(h, t, &v);
    return v;
}

static double getDoubleParam(OfxParamHandle h, double t) {
    double v = 0;
    gParam->paramGetValueAtTime(h, t, &v);
    return v;
}

static Params readParams(Instance *in, double t) {
    Params p;
    p.model = getStringParam(in->pModel, t);
    p.output = getIntParam(in->pOutput, t);
    p.refineSteps = getIntParam(in->pRefine, t);
    p.resolutionLevel = getIntParam(in->pResLevel, t);
    p.numTokens = getIntParam(in->pTokens, t);
    p.fovX = getDoubleParam(in->pFov, t);
    p.fp16 = getIntParam(in->pFp16, t);
    p.colorspace = getIntParam(in->pColorspace, t);
    p.normalSpace = getIntParam(in->pNormalSpace, t);
    p.applyMask = getIntParam(in->pApplyMask, t);
    p.python = getStringParam(in->pPython, t);
    p.daemon = getStringParam(in->pDaemon, t);
    p.port = getIntParam(in->pPort, t);
    p.autoStart = getIntParam(in->pAutoStart, t);
    p.exitWithHost = getIntParam(in->pExitWithHost, t);
    if (p.model.empty()) p.model = gDefaultModel;
    if (p.python.empty()) p.python = gDefaultPython;
    if (p.daemon.empty()) p.daemon = gDefaultDaemon;
    if (p.port <= 0) p.port = kDefaultPort;
    return p;
}

// ---------------------------------------------------------------------------
// daemon client
// ---------------------------------------------------------------------------

static bool sendAll(sock_t s, const void *data, size_t n) {
    const char *p = (const char *)data;
    while (n > 0) {
        int k = send(s, p, (int)std::min<size_t>(n, 1 << 20), 0);
        if (k <= 0) return false;
        p += k;
        n -= k;
    }
    return true;
}

static bool recvAll(sock_t s, void *data, size_t n) {
    char *p = (char *)data;
    while (n > 0) {
        int k = recv(s, p, (int)std::min<size_t>(n, 1 << 20), 0);
        if (k <= 0) return false;
        p += k;
        n -= k;
    }
    return true;
}

static void setNonBlocking(sock_t s, bool on) {
#ifdef _WIN32
    u_long v = on ? 1 : 0;
    ioctlsocket(s, FIONBIO, &v);
#else
    int flags = fcntl(s, F_GETFL, 0);
    fcntl(s, F_SETFL, on ? (flags | O_NONBLOCK) : (flags & ~O_NONBLOCK));
#endif
}

static bool connectInProgress() {
#ifdef _WIN32
    return WSAGetLastError() == WSAEWOULDBLOCK;
#else
    return errno == EINPROGRESS;
#endif
}

static sock_t tryConnect(int port, int timeoutMs) {
    sock_t s = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
    if (s == INVALID_SOCKET) return INVALID_SOCKET;
    sockaddr_in addr;
    memset(&addr, 0, sizeof(addr));
    addr.sin_family = AF_INET;
    addr.sin_port = htons((unsigned short)port);
    addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);

    setNonBlocking(s, true);
    int r = connect(s, (sockaddr *)&addr, sizeof(addr));
    if (r == SOCKET_ERROR && !connectInProgress()) {
        closeSock(s);
        return INVALID_SOCKET;
    }
    fd_set wfds, efds;
    FD_ZERO(&wfds);
    FD_ZERO(&efds);
    FD_SET(s, &wfds);
    FD_SET(s, &efds);
    timeval tv;
    tv.tv_sec = timeoutMs / 1000;
    tv.tv_usec = (timeoutMs % 1000) * 1000;
    r = select((int)s + 1, nullptr, &wfds, &efds, &tv);
    int soerr = 0;
    socklen_t slen = sizeof(soerr);
    if (r > 0) getsockopt(s, SOL_SOCKET, SO_ERROR, (char *)&soerr, &slen);
    if (r <= 0 || FD_ISSET(s, &efds) || soerr != 0) {
        closeSock(s);
        return INVALID_SOCKET;
    }
    setNonBlocking(s, false);
    int one = 1;
    setsockopt(s, IPPROTO_TCP, TCP_NODELAY, (const char *)&one, sizeof(one));
#ifdef _WIN32
    DWORD to = 600 * 1000;  // model load + a 4K refine can take a while
#else
    timeval to;
    to.tv_sec = 600;
    to.tv_usec = 0;
#endif
    setsockopt(s, SOL_SOCKET, SO_RCVTIMEO, (const char *)&to, sizeof(to));
    setsockopt(s, SOL_SOCKET, SO_SNDTIMEO, (const char *)&to, sizeof(to));
    return s;
}

// Nuke sets PYTHONHOME (and friends) to its own install; a child interpreter
// that inherits them loads Nuke's stdlib and dies with "SRE module mismatch".
static bool isPythonBootstrapVar(const std::string &entry) {
    std::string upper = entry.substr(0, 18);
    for (auto &ch : upper) ch = (char)toupper((unsigned char)ch);
    return upper.rfind("PYTHONHOME=", 0) == 0 || upper.rfind("PYTHONPATH=", 0) == 0 ||
           upper.rfind("PYTHONEXECUTABLE=", 0) == 0 || upper.rfind("PYTHONSTARTUP=", 0) == 0;
}

static std::string daemonLogPath() {
#ifdef _WIN32
    return "";  // the daemon gets its own console window instead
#else
    std::string dir = envVar("XDG_CACHE_HOME");
    if (dir.empty()) dir = envVar("HOME") + "/.cache";
    dir += "/moge-nuke";
    mkdir(dir.c_str(), 0755);
    return dir + "/daemon-" + std::to_string((long)getuid()) + ".log";
#endif
}

#ifdef _WIN32
static std::vector<wchar_t> daemonEnvBlock() {
    std::vector<wchar_t> block;
    wchar_t *env = GetEnvironmentStringsW();
    if (!env) return block;
    for (wchar_t *p = env; *p;) {
        size_t len = wcslen(p);
        std::wstring entry(p, len);
        int m = WideCharToMultiByte(CP_UTF8, 0, entry.c_str(), (int)entry.size(), nullptr, 0, nullptr, nullptr);
        std::string narrowEntry(m, 0);
        WideCharToMultiByte(CP_UTF8, 0, entry.c_str(), (int)entry.size(), &narrowEntry[0], m, nullptr, nullptr);
        if (!isPythonBootstrapVar(narrowEntry)) {
            block.insert(block.end(), entry.begin(), entry.end());
            block.push_back(0);
        }
        p += len + 1;
    }
    block.push_back(0);
    FreeEnvironmentStringsW(env);
    return block;
}
#endif

#ifdef _WIN32
typedef HANDLE proc_t;
#else
typedef pid_t proc_t;
#endif

// True while the process we spawned is still running.
static bool processAlive(proc_t h) {
#ifdef _WIN32
    return h && WaitForSingleObject(h, 0) == WAIT_TIMEOUT;
#else
    int status = 0;
    return h > 0 && waitpid(h, &status, WNOHANG) == 0;
#endif
}

static void releaseProcess(proc_t h) {
#ifdef _WIN32
    if (h) CloseHandle(h);
#else
    (void)h;
#endif
}

static bool spawnDaemon(const Params &p, proc_t &proc, std::string &err) {
    proc = (proc_t)0;
    if (!fileExists(p.python)) {
        err = "python not found: '" + p.python + "'. Run the installer from the MoGe-nuke repo, "
              "or set 'python' on the Setup tab to the venv interpreter.";
        return false;
    }
    if (!fileExists(p.daemon)) {
        err = "daemon script not found: '" + p.daemon + "'. Set it on the Setup tab.";
        return false;
    }
    std::string cwd = p.daemon.substr(0, p.daemon.find_last_of("/\\"));
    std::vector<std::string> argv = {p.python, p.daemon, "--model", p.model, "--port",
                                     std::to_string(p.port)};
    if (p.exitWithHost) {  // daemon watches our pid and exits with us
        argv.push_back("--parent-pid");
        argv.push_back(std::to_string(currentPid()));
    }
    std::string pretty;
    for (auto &a : argv) pretty += (pretty.empty() ? "" : " ") + (a.find(' ') != std::string::npos ? "\"" + a + "\"" : a);

#ifdef _WIN32
    std::string cmd;
    for (auto &a : argv) cmd += (cmd.empty() ? "\"" : " \"") + a + "\"";
    std::wstring wcmd = widen(cmd);
    std::vector<wchar_t> cmdbuf(wcmd.begin(), wcmd.end());
    cmdbuf.push_back(0);
    std::vector<wchar_t> env = daemonEnvBlock();

    STARTUPINFOW si;
    memset(&si, 0, sizeof(si));
    si.cb = sizeof(si);
    si.dwFlags = STARTF_USESHOWWINDOW;
    si.wShowWindow = SW_SHOWMINNOACTIVE;  // own console, minimised, keeps focus in Nuke
    PROCESS_INFORMATION pi;
    memset(&pi, 0, sizeof(pi));
    std::wstring wcwd = widen(cwd);
    BOOL ok = CreateProcessW(nullptr, cmdbuf.data(), nullptr, nullptr, FALSE,
                             CREATE_NEW_CONSOLE | CREATE_UNICODE_ENVIRONMENT,
                             env.empty() ? nullptr : env.data(),
                             wcwd.empty() ? nullptr : wcwd.c_str(), &si, &pi);
    if (!ok) {
        char b[256];
        snprintf(b, sizeof(b), "CreateProcess failed (%lu) for: %s", GetLastError(), pretty.c_str());
        err = b;
        return false;
    }
    logf("started daemon pid %lu: %s", pi.dwProcessId, pretty.c_str());
    CloseHandle(pi.hThread);
    proc = pi.hProcess;
    return true;
#else
    std::vector<std::string> envStore;
    for (char **e = environ; e && *e; ++e)
        if (!isPythonBootstrapVar(*e)) envStore.push_back(*e);
    std::vector<char *> envp;
    for (auto &e : envStore) envp.push_back(&e[0]);
    envp.push_back(nullptr);
    std::vector<char *> args;
    for (auto &a : argv) args.push_back(&a[0]);
    args.push_back(nullptr);

    std::string logPath = daemonLogPath();
    posix_spawn_file_actions_t fa;
    posix_spawn_file_actions_init(&fa);
    posix_spawn_file_actions_addopen(&fa, 0, "/dev/null", O_RDONLY, 0);
    posix_spawn_file_actions_addopen(&fa, 1, logPath.c_str(), O_WRONLY | O_CREAT | O_APPEND, 0644);
    posix_spawn_file_actions_adddup2(&fa, 1, 2);
    posix_spawn_file_actions_addchdir_np(&fa, cwd.c_str());
    posix_spawnattr_t attr;
    posix_spawnattr_init(&attr);
    posix_spawnattr_setflags(&attr, POSIX_SPAWN_SETSID);  // own session: survives host SIGHUP
    pid_t pid = 0;
    int rc = posix_spawn(&pid, p.python.c_str(), &fa, &attr, args.data(), envp.data());
    posix_spawn_file_actions_destroy(&fa);
    posix_spawnattr_destroy(&attr);
    if (rc != 0) {
        err = std::string("posix_spawn failed (") + strerror(rc) + ") for: " + pretty;
        return false;
    }
    logf("started daemon pid %ld (log: %s): %s", (long)pid, logPath.c_str(), pretty.c_str());
    proc = pid;
    return true;
#endif
}

static bool ensureConnected(Instance *in, const Params &p, std::string &err) {
    if (in->sock != INVALID_SOCKET) return true;
    in->sock = tryConnect(p.port, 500);
    if (in->sock != INVALID_SOCKET) return true;

    if (!p.autoStart) {
        err = "no MoGe daemon on port " + std::to_string(p.port) +
              ". Enable 'auto-start daemon' or run:\n  \"" + p.python + "\" \"" + p.daemon +
              "\" --port " + std::to_string(p.port);
        return false;
    }

    std::lock_guard<std::mutex> lock(gSpawnMutex);
    // another instance may have spawned it while we waited for the lock
    in->sock = tryConnect(p.port, 500);
    if (in->sock != INVALID_SOCKET) return true;

    proc_t proc;
    if (!spawnDaemon(p, proc, err)) return false;

    if (gProgress) gProgress->progressStart(in->effect, "MoGe3: starting daemon and loading the model ...");
    const int maxWaitMs = 10 * 60 * 1000;
    int waited = 0;
    bool aborted = false, died = false;
    while (waited < maxWaitMs) {
        in->sock = tryConnect(p.port, 1000);
        if (in->sock != INVALID_SOCKET) break;
        sleepMs(1000);
        waited += 2000;
        if (gProgress) gProgress->progressUpdate(in->effect, std::min(0.95, waited / 90000.0));
        if (!processAlive(proc)) {  // crashed on startup: do not sit out the timeout
            died = true;
            break;
        }
        if (gEffect->abort(in->effect)) {
            aborted = true;
            break;
        }
    }
    releaseProcess(proc);
    if (gProgress) gProgress->progressEnd(in->effect);
    if (in->sock == INVALID_SOCKET) {
#ifdef _WIN32
        std::string where = "check its console window";
#else
        std::string where = "see " + daemonLogPath();
#endif
        if (aborted) err = "cancelled while waiting for the daemon";
        else if (died) err = "the daemon exited during startup (bad python/model path, missing "
                             "packages, CUDA?) -- " + where;
        else err = "daemon started but never answered on port " + std::to_string(p.port) + " -- " + where;
        return false;
    }
    return true;
}

// One request/reply. Returns false and sets err on any failure. On transport
// failure the socket is dropped so the next call reconnects.
static bool daemonInfer(Instance *in, const Params &p, int w, int h,
                        const std::vector<float> &rgb, std::vector<float> &planes,
                        std::string &err) {
    char hdr[2048];
    snprintf(hdr, sizeof(hdr),
             "{\"cmd\":\"infer\",\"width\":%d,\"height\":%d,\"model\":\"%s\","
             "\"refine_steps\":%d,\"resolution_level\":%d,\"num_tokens\":%d,"
             "\"fov_x\":%.6f,\"fp16\":%d,\"input_colorspace\":\"%s\","
             "\"normal_space\":\"%s\",\"apply_mask\":%d}",
             w, h, jsonEscape(p.model).c_str(), p.refineSteps, p.resolutionLevel,
             p.numTokens, p.fovX, p.fp16, p.colorspace == 0 ? "linear" : "srgb",
             p.normalSpace == 0 ? "nuke" : "opencv", p.applyMask);
    uint32_t jlen = (uint32_t)strlen(hdr);
    uint32_t dlen = (uint32_t)(rgb.size() * sizeof(float));

    for (int attempt = 0; attempt < 2; ++attempt) {
        if (!ensureConnected(in, p, err)) return false;
        bool ok = sendAll(in->sock, "MOGE", 4) && sendAll(in->sock, &jlen, 4) &&
                  sendAll(in->sock, hdr, jlen) && sendAll(in->sock, &dlen, 4) &&
                  sendAll(in->sock, rgb.data(), dlen);
        unsigned char rh[24];
        if (ok) ok = recvAll(in->sock, rh, 24);
        if (!ok) {
            closeSock(in->sock);
            in->sock = INVALID_SOCKET;
            err = "connection to the daemon dropped";
            continue;  // reconnect once
        }
        if (memcmp(rh, "MOGR", 4) != 0) {
            closeSock(in->sock);
            in->sock = INVALID_SOCKET;
            err = "garbled reply from the daemon";
            return false;
        }
        int32_t status;
        uint32_t rw, rhgt, rc, mlen;
        memcpy(&status, rh + 4, 4);
        memcpy(&rw, rh + 8, 4);
        memcpy(&rhgt, rh + 12, 4);
        memcpy(&rc, rh + 16, 4);
        memcpy(&mlen, rh + 20, 4);
        std::string msg(mlen, 0);
        if (mlen && !recvAll(in->sock, &msg[0], mlen)) {
            closeSock(in->sock);
            in->sock = INVALID_SOCKET;
            err = "connection dropped mid-reply";
            return false;
        }
        uint32_t plen;
        if (!recvAll(in->sock, &plen, 4)) {
            closeSock(in->sock);
            in->sock = INVALID_SOCKET;
            err = "connection dropped mid-reply";
            return false;
        }
        std::vector<float> payload(plen / sizeof(float));
        if (plen && !recvAll(in->sock, payload.data(), plen)) {
            closeSock(in->sock);
            in->sock = INVALID_SOCKET;
            err = "connection dropped mid-payload";
            return false;
        }
        if (status != 0) {
            err = msg;
            return false;
        }
        if (rw != (uint32_t)w || rhgt != (uint32_t)h || rc != 5 ||
            payload.size() != (size_t)5 * w * h) {
            err = "daemon reply has unexpected shape";
            return false;
        }
        planes.swap(payload);
        logf("frame %dx%d refine=%d: %s", w, h, p.refineSteps, msg.c_str());
        return true;
    }
    return false;
}

// ---------------------------------------------------------------------------
// configuration
// ---------------------------------------------------------------------------

static std::string trim(std::string s) {
    while (!s.empty() && (s.back() == ' ' || s.back() == '\r' || s.back() == '\n' || s.back() == '\t')) s.pop_back();
    size_t i = 0;
    while (i < s.size() && (s[i] == ' ' || s[i] == '\t')) ++i;
    return s.substr(i);
}

static void loadConfig() {
    std::string python, daemon, model;
    int port = 0;

    // 2. moge3.cfg beside the .ofx
    std::string cfgPath = pluginDir() + "/moge3.cfg";
    if (FILE *f = openRead(cfgPath)) {
        char line[4096];
        while (fgets(line, sizeof(line), f)) {
            std::string l = trim(line);
            if (l.empty() || l[0] == '#') continue;
            size_t eq = l.find('=');
            if (eq == std::string::npos) continue;
            std::string key = trim(l.substr(0, eq)), val = trim(l.substr(eq + 1));
            if (key == "root") gRoot = val;
            else if (key == "python") python = val;
            else if (key == "daemon") daemon = val;
            else if (key == "model") model = val;
            else if (key == "port") port = atoi(val.c_str());
        }
        fclose(f);
        logf("config: %s", cfgPath.c_str());
    }
    // 1. environment wins over the file
    std::string envRoot = envVar("MOGE_NUKE_ROOT");
    if (!envRoot.empty()) gRoot = envRoot;

    for (auto &c : gRoot) if (c == '\\') c = '/';
    while (!gRoot.empty() && gRoot.back() == '/') gRoot.pop_back();

#ifdef _WIN32
    const char *venvPython = "/.venv/Scripts/python.exe";
#else
    const char *venvPython = "/.venv/bin/python";
#endif
    gDefaultPython = !python.empty() ? python : (gRoot.empty() ? "" : gRoot + venvPython);
    gDefaultDaemon = !daemon.empty() ? daemon : (gRoot.empty() ? "" : gRoot + "/daemon/moge_daemon.py");
    if (!model.empty()) gDefaultModel = model;
    else if (!gRoot.empty() && fileExists(gRoot + "/models/moge-3-vitg.safetensors")) gDefaultModel = gRoot + "/models/moge-3-vitg.safetensors";
    else if (!gRoot.empty() && fileExists(gRoot + "/models/moge-3-vitg.pt")) gDefaultModel = gRoot + "/models/moge-3-vitg.pt";
    else gDefaultModel = kHfModel;
    if (port > 0) kDefaultPort = port;

    if (gRoot.empty())
        logf("no MOGE_NUKE_ROOT and no moge3.cfg -- fill in the Setup tab on the node");
    else
        logf("root %s | python %s | model %s", gRoot.c_str(), gDefaultPython.c_str(), gDefaultModel.c_str());
}

// ---------------------------------------------------------------------------
// OFX actions
// ---------------------------------------------------------------------------

static OfxStatus onLoad() {
    if (!gHost) return kOfxStatErrMissingHostFeature;
    loadConfig();
    gEffect = (OfxImageEffectSuiteV1 *)gHost->fetchSuite(gHost->host, kOfxImageEffectSuite, 1);
    gProp = (OfxPropertySuiteV1 *)gHost->fetchSuite(gHost->host, kOfxPropertySuite, 1);
    gParam = (OfxParameterSuiteV1 *)gHost->fetchSuite(gHost->host, kOfxParameterSuite, 1);
    gMessage = (OfxMessageSuiteV1 *)gHost->fetchSuite(gHost->host, kOfxMessageSuite, 1);
    gProgress = (OfxProgressSuiteV1 *)gHost->fetchSuite(gHost->host, kOfxProgressSuite, 1);
    if (!gEffect || !gProp || !gParam) return kOfxStatErrMissingHostFeature;
#ifdef _WIN32
    WSADATA wsa;
    WSAStartup(MAKEWORD(2, 2), &wsa);
#else
    signal(SIGPIPE, SIG_IGN);  // a dead daemon must not kill the host
#endif
    return kOfxStatOK;
}

static OfxStatus onUnload() {
#ifdef _WIN32
    WSACleanup();
#endif
    return kOfxStatOK;
}

static OfxStatus describe(OfxImageEffectHandle effect) {
    OfxPropertySetHandle props;
    gEffect->getPropertySet(effect, &props);
    gProp->propSetString(props, kOfxPropLabel, 0, PLUGIN_LABEL);
    gProp->propSetString(props, kOfxPropShortLabel, 0, PLUGIN_LABEL);
    gProp->propSetString(props, kOfxPropLongLabel, 0, "MoGe3 depth + normals (full model)");
    gProp->propSetString(props, kOfxImageEffectPluginPropGrouping, 0, "ML");
    gProp->propSetString(props, kOfxPropPluginDescription, 0,
                         "Monocular depth and normals from MoGe-3 (ViT-G), refiner included. "
                         "The model runs in a resident Python daemon; this node is the client.");
    gProp->propSetString(props, kOfxImageEffectPropSupportedContexts, 0, kOfxImageEffectContextFilter);
    gProp->propSetString(props, kOfxImageEffectPropSupportedContexts, 1, kOfxImageEffectContextGeneral);
    gProp->propSetString(props, kOfxImageEffectPropSupportedPixelDepths, 0, kOfxBitDepthFloat);
    gProp->propSetInt(props, kOfxImageEffectPropSupportsTiles, 0, 0);
    gProp->propSetInt(props, kOfxImageEffectPropSupportsMultiResolution, 0, 0);
    gProp->propSetInt(props, kOfxImageEffectPropSupportsMultipleClipDepths, 0, 0);
    gProp->propSetInt(props, kOfxImageEffectPluginPropHostFrameThreading, 0, 0);
    gProp->propSetInt(props, kOfxImageEffectPropTemporalClipAccess, 0, 0);
    gProp->propSetString(props, kOfxImageEffectPluginRenderThreadSafety, 0, kOfxImageEffectRenderInstanceSafe);
    gProp->propSetInt(props, kOfxImageEffectPluginPropFieldRenderTwiceAlways, 0, 0);
    return kOfxStatOK;
}

static OfxPropertySetHandle defineParam(OfxParamSetHandle set, const char *type, const char *name,
                                        const char *label, const char *hint, const char *page) {
    OfxPropertySetHandle props;
    gParam->paramDefine(set, type, name, &props);
    gProp->propSetString(props, kOfxPropLabel, 0, label);
    if (hint) gProp->propSetString(props, kOfxParamPropHint, 0, hint);
    gProp->propSetInt(props, kOfxParamPropAnimates, 0, 0);
    (void)page;
    return props;
}

static OfxStatus describeInContext(OfxImageEffectHandle effect) {
    OfxPropertySetHandle props;
    gEffect->clipDefine(effect, kOfxImageEffectSimpleSourceClipName, &props);
    gProp->propSetString(props, kOfxImageEffectPropSupportedComponents, 0, kOfxImageComponentRGBA);
    gProp->propSetString(props, kOfxImageEffectPropSupportedComponents, 1, kOfxImageComponentRGB);
    gProp->propSetInt(props, kOfxImageClipPropOptional, 0, 0);

    gEffect->clipDefine(effect, kOfxImageEffectOutputClipName, &props);
    gProp->propSetString(props, kOfxImageEffectPropSupportedComponents, 0, kOfxImageComponentRGBA);

    OfxParamSetHandle set;
    gEffect->getParamSet(effect, &set);
    OfxPropertySetHandle p;

    p = defineParam(set, kOfxParamTypeString, "model", "model",
                    "MoGe-3 checkpoint: a local .pt / .safetensors, or a Hugging Face repo id such as "
                    "Ruicheng/moge-3-vitg (downloaded by the daemon on first use).", "MoGe");
    gProp->propSetString(p, kOfxParamPropStringMode, 0, kOfxParamStringIsFilePath);
    gProp->propSetInt(p, kOfxParamPropStringFilePathExists, 0, 0);
    gProp->propSetString(p, kOfxParamPropDefault, 0, gDefaultModel.c_str());

    p = defineParam(set, kOfxParamTypeChoice, "output", "output",
                    "Which pass to output. Both come from one inference, so switching is free.\n"
                    "depth: RGB = metric depth (scene units), A = validity mask\n"
                    "normals: RGB = normal, A = validity mask", "MoGe");
    gProp->propSetString(p, kOfxParamPropChoiceOption, 0, "depth");
    gProp->propSetString(p, kOfxParamPropChoiceOption, 1, "normals");
    gProp->propSetInt(p, kOfxParamPropDefault, 0, 0);

    p = defineParam(set, kOfxParamTypeChoice, "refineSteps", "refine steps",
                    "Sparse 3D refiner iterations (the Triton stage the live .cat node cannot run).\n"
                    "Affects depth only -- normals are identical at every setting.\n"
                    "0 is the only setting that is bit-exact run to run.", "MoGe");
    for (int i = 0; i < kRefineCount; ++i)
        gProp->propSetString(p, kOfxParamPropChoiceOption, i, kRefineLabels[i]);
    gProp->propSetInt(p, kOfxParamPropDefault, 0, kRefineDefault);

    p = defineParam(set, kOfxParamTypeInteger, "resolutionLevel", "resolution level",
                    "0-9, detail vs speed. Ignored when num tokens > 0.", "MoGe");
    gProp->propSetInt(p, kOfxParamPropDefault, 0, 9);
    gProp->propSetInt(p, kOfxParamPropMin, 0, 0);
    gProp->propSetInt(p, kOfxParamPropMax, 0, 9);
    gProp->propSetInt(p, kOfxParamPropDisplayMin, 0, 0);
    gProp->propSetInt(p, kOfxParamPropDisplayMax, 0, 9);

    p = defineParam(set, kOfxParamTypeInteger, "numTokens", "num tokens",
                    "ViT tokens, 1200-3600 trained range. 0 = derive from resolution level.", "MoGe");
    gProp->propSetInt(p, kOfxParamPropDefault, 0, 0);
    gProp->propSetInt(p, kOfxParamPropMin, 0, 0);
    gProp->propSetInt(p, kOfxParamPropMax, 0, 4000);
    gProp->propSetInt(p, kOfxParamPropDisplayMin, 0, 0);
    gProp->propSetInt(p, kOfxParamPropDisplayMax, 0, 4000);

    p = defineParam(set, kOfxParamTypeDouble, "fovX", "lock fov x",
                    "Degrees. 0 = estimate per frame. Locking only rescales depth "
                    "(camera moves, geometry does not). Normals unaffected.", "MoGe");
    gProp->propSetDouble(p, kOfxParamPropDefault, 0, 0.0);
    gProp->propSetDouble(p, kOfxParamPropMin, 0, 0.0);
    gProp->propSetDouble(p, kOfxParamPropMax, 0, 179.0);
    gProp->propSetDouble(p, kOfxParamPropDisplayMin, 0, 0.0);
    gProp->propSetDouble(p, kOfxParamPropDisplayMax, 0, 120.0);

    p = defineParam(set, kOfxParamTypeBoolean, "fp16", "fp16",
                    "Mixed precision. Faster; the refiner's voxel stage stays fp32.", "MoGe");
    gProp->propSetInt(p, kOfxParamPropDefault, 0, 0);

    p = defineParam(set, kOfxParamTypeChoice, "inputColorspace", "input colorspace",
                    "MoGe was trained on sRGB-encoded images. Nuke feeds scene-linear, so the "
                    "default encodes to sRGB first. Pick 'already sRGB' if you bypassed the "
                    "Read's colorspace (raw).", "MoGe");
    gProp->propSetString(p, kOfxParamPropChoiceOption, 0, "scene-linear (encode to sRGB)");
    gProp->propSetString(p, kOfxParamPropChoiceOption, 1, "already sRGB");
    gProp->propSetInt(p, kOfxParamPropDefault, 0, 0);

    p = defineParam(set, kOfxParamTypeChoice, "normalSpace", "normal space",
                    "nuke = x right, y up, z toward camera (matches the bake/live nodes and the "
                    "relight group). opencv = raw MoGe axes (y down, z away).", "MoGe");
    gProp->propSetString(p, kOfxParamPropChoiceOption, 0, "nuke");
    gProp->propSetString(p, kOfxParamPropChoiceOption, 1, "opencv");
    gProp->propSetInt(p, kOfxParamPropDefault, 0, 0);

    p = defineParam(set, kOfxParamTypeBoolean, "applyMask", "apply mask",
                    "Zero depth/normals where the model marks pixels invalid (sky, etc). "
                    "Off keeps the raw values; the mask is always in alpha.", "MoGe");
    gProp->propSetInt(p, kOfxParamPropDefault, 0, 1);

    // ---- setup -------------------------------------------------------------
    p = defineParam(set, kOfxParamTypeString, "pythonExe", "python",
                    "Interpreter of the MoGe venv (torch + Triton). NOT Nuke's Python.", "Setup");
    gProp->propSetString(p, kOfxParamPropStringMode, 0, kOfxParamStringIsFilePath);
    gProp->propSetInt(p, kOfxParamPropStringFilePathExists, 0, 0);
    gProp->propSetString(p, kOfxParamPropDefault, 0, gDefaultPython.c_str());

    p = defineParam(set, kOfxParamTypeString, "daemonScript", "daemon script", nullptr, "Setup");
    gProp->propSetString(p, kOfxParamPropStringMode, 0, kOfxParamStringIsFilePath);
    gProp->propSetInt(p, kOfxParamPropStringFilePathExists, 0, 0);
    gProp->propSetString(p, kOfxParamPropDefault, 0, gDefaultDaemon.c_str());

    p = defineParam(set, kOfxParamTypeInteger, "port", "port",
                    "TCP port of the daemon on localhost.", "Setup");
    gProp->propSetInt(p, kOfxParamPropDefault, 0, kDefaultPort);
    gProp->propSetInt(p, kOfxParamPropMin, 0, 1024);
    gProp->propSetInt(p, kOfxParamPropMax, 0, 65535);
    gProp->propSetInt(p, kOfxParamPropDisplayMin, 0, 1024);
    gProp->propSetInt(p, kOfxParamPropDisplayMax, 0, 65535);

    p = defineParam(set, kOfxParamTypeBoolean, "autoStart", "auto-start daemon",
                    "If nothing answers on the port, launch the daemon (its own console "
                    "window) and wait for the model to load.", "Setup");
    gProp->propSetInt(p, kOfxParamPropDefault, 0, 1);

    p = defineParam(set, kOfxParamTypeBoolean, "exitWithHost", "daemon exits with Nuke",
                    "A daemon started by this node shuts down (freeing the GPU) when this "
                    "Nuke process exits, crash included. Off keeps it resident across "
                    "sessions so the model never reloads.", "Setup");
    gProp->propSetInt(p, kOfxParamPropDefault, 0, 1);

    // pages -> tabs in Nuke
    OfxPropertySetHandle pg;
    gParam->paramDefine(set, kOfxParamTypePage, "pageMoGe", &pg);
    gProp->propSetString(pg, kOfxPropLabel, 0, "MoGe");
    const char *mainKids[] = {"model", "output", "refineSteps", "resolutionLevel", "numTokens",
                              "fovX", "fp16", "inputColorspace", "normalSpace", "applyMask"};
    for (int i = 0; i < 10; ++i) gProp->propSetString(pg, kOfxParamPropPageChild, i, mainKids[i]);
    gParam->paramDefine(set, kOfxParamTypePage, "pageSetup", &pg);
    gProp->propSetString(pg, kOfxPropLabel, 0, "Setup");
    const char *setupKids[] = {"pythonExe", "daemonScript", "port", "autoStart", "exitWithHost"};
    for (int i = 0; i < 5; ++i) gProp->propSetString(pg, kOfxParamPropPageChild, i, setupKids[i]);
    return kOfxStatOK;
}

static OfxStatus createInstance(OfxImageEffectHandle effect) {
    Instance *in = new Instance();
    in->effect = effect;
    gEffect->clipGetHandle(effect, kOfxImageEffectSimpleSourceClipName, &in->source, nullptr);
    gEffect->clipGetHandle(effect, kOfxImageEffectOutputClipName, &in->output, nullptr);
    OfxParamSetHandle set;
    gEffect->getParamSet(effect, &set);
    struct { const char *name; OfxParamHandle *dst; } table[] = {
        {"model", &in->pModel}, {"output", &in->pOutput}, {"refineSteps", &in->pRefine},
        {"resolutionLevel", &in->pResLevel}, {"numTokens", &in->pTokens}, {"fovX", &in->pFov},
        {"fp16", &in->pFp16}, {"inputColorspace", &in->pColorspace},
        {"normalSpace", &in->pNormalSpace}, {"applyMask", &in->pApplyMask},
        {"pythonExe", &in->pPython}, {"daemonScript", &in->pDaemon}, {"port", &in->pPort},
        {"autoStart", &in->pAutoStart}, {"exitWithHost", &in->pExitWithHost},
    };
    for (auto &row : table) gParam->paramGetHandle(set, row.name, row.dst, nullptr);
    OfxPropertySetHandle props;
    gEffect->getPropertySet(effect, &props);
    gProp->propSetPointer(props, kOfxPropInstanceData, 0, in);
    return kOfxStatOK;
}

static Instance *getInstance(OfxImageEffectHandle effect) {
    OfxPropertySetHandle props;
    gEffect->getPropertySet(effect, &props);
    void *p = nullptr;
    gProp->propGetPointer(props, kOfxPropInstanceData, 0, &p);
    return (Instance *)p;
}

static OfxStatus destroyInstance(OfxImageEffectHandle effect) {
    Instance *in = getInstance(effect);
    if (in) {
        if (in->sock != INVALID_SOCKET) closeSock(in->sock);
        delete in;
    }
    return kOfxStatOK;
}

static OfxStatus getClipPreferences(OfxImageEffectHandle, OfxPropertySetHandle outArgs) {
    gProp->propSetString(outArgs, "OfxImageClipPropComponents_Output", 0, kOfxImageComponentRGBA);
    gProp->propSetString(outArgs, "OfxImageClipPropDepth_Output", 0, kOfxBitDepthFloat);
    gProp->propSetString(outArgs, kOfxImageEffectPropPreMultiplication, 0, kOfxImageUnPreMultiplied);
    gProp->propSetInt(outArgs, kOfxImageEffectFrameVarying, 0, 1);
    return kOfxStatOK;
}

// The model needs the whole frame regardless of what the host wants back.
static OfxStatus getRegionsOfInterest(OfxImageEffectHandle effect, OfxPropertySetHandle inArgs,
                                      OfxPropertySetHandle outArgs) {
    Instance *in = getInstance(effect);
    double time;
    gProp->propGetDouble(inArgs, kOfxPropTime, 0, &time);
    OfxRectD rod;
    if (gEffect->clipGetRegionOfDefinition(in->source, time, &rod) != kOfxStatOK)
        return kOfxStatReplyDefault;
    gProp->propSetDoubleN(outArgs, "OfxImageClipPropRoI_Source", 4, &rod.x1);
    return kOfxStatOK;
}

struct ImageView {
    OfxPropertySetHandle handle = nullptr;
    float *data = nullptr;
    int x1 = 0, y1 = 0, x2 = 0, y2 = 0;
    int rowBytes = 0;
    int nComp = 4;
    int w() const { return x2 - x1; }
    int h() const { return y2 - y1; }
    float *row(int y) const { return (float *)((char *)data + (size_t)(y - y1) * rowBytes); }
};

static bool fetchImage(OfxImageClipHandle clip, double time, ImageView &v, std::string &err) {
    if (gEffect->clipGetImage(clip, time, nullptr, &v.handle) != kOfxStatOK || !v.handle) {
        err = "host gave no image for the clip";
        return false;
    }
    void *ptr = nullptr;
    gProp->propGetPointer(v.handle, kOfxImagePropData, 0, &ptr);
    v.data = (float *)ptr;
    int b[4];
    gProp->propGetIntN(v.handle, kOfxImagePropBounds, 4, b);
    v.x1 = b[0]; v.y1 = b[1]; v.x2 = b[2]; v.y2 = b[3];
    gProp->propGetInt(v.handle, kOfxImagePropRowBytes, 0, &v.rowBytes);
    char *comps = nullptr, *depth = nullptr;
    gProp->propGetString(v.handle, kOfxImageEffectPropComponents, 0, &comps);
    gProp->propGetString(v.handle, kOfxImageEffectPropPixelDepth, 0, &depth);
    if (!depth || strcmp(depth, kOfxBitDepthFloat) != 0) {
        err = "only float images are supported";
        return false;
    }
    if (comps && strcmp(comps, kOfxImageComponentRGB) == 0) v.nComp = 3;
    else if (comps && strcmp(comps, kOfxImageComponentRGBA) == 0) v.nComp = 4;
    else if (comps && strcmp(comps, kOfxImageComponentAlpha) == 0) v.nComp = 1;
    else { err = std::string("unsupported components: ") + (comps ? comps : "?"); return false; }
    if (!v.data || v.w() <= 0 || v.h() <= 0) {
        err = "empty image";
        return false;
    }
    return true;
}

static OfxStatus render(OfxImageEffectHandle effect, OfxPropertySetHandle inArgs) {
    Instance *in = getInstance(effect);
    double time;
    gProp->propGetDouble(inArgs, kOfxPropTime, 0, &time);
    int rw[4];
    gProp->propGetIntN(inArgs, kOfxImageEffectPropRenderWindow, 4, rw);

    Params p = readParams(in, time);
    std::string err;

    ImageView src, dst;
    if (!fetchImage(in->source, time, src, err)) {
        if (src.handle) gEffect->clipReleaseImage(src.handle);
        showError(in, "source: " + err);
        return kOfxStatFailed;
    }
    OfxStatus result = kOfxStatOK;
    if (!fetchImage(in->output, time, dst, err)) {
        gEffect->clipReleaseImage(src.handle);
        if (dst.handle) gEffect->clipReleaseImage(dst.handle);
        showError(in, "output: " + err);
        return kOfxStatFailed;
    }

    const int W = src.w(), H = src.h();

    // pack RGB, top row first (OFX images are bottom-up)
    std::vector<float> rgb((size_t)3 * W * H);
    for (int y = 0; y < H; ++y) {
        const float *s = src.row(src.y2 - 1 - y);
        float *d = &rgb[(size_t)3 * W * y];
        if (src.nComp == 1) {
            for (int x = 0; x < W; ++x) d[3 * x] = d[3 * x + 1] = d[3 * x + 2] = s[x];
        } else {
            for (int x = 0; x < W; ++x) {
                d[3 * x] = s[x * src.nComp];
                d[3 * x + 1] = s[x * src.nComp + 1];
                d[3 * x + 2] = s[x * src.nComp + 2];
            }
        }
    }
    uint64_t hash = fnv1a(rgb.data(), rgb.size() * sizeof(float));

    bool hit = in->cacheValid && in->cacheW == W && in->cacheH == H && in->cacheHash == hash &&
               in->cacheParams.sameInference(p);
    if (!hit) {
        std::vector<float> planes;
        if (!daemonInfer(in, p, W, H, rgb, planes, err)) {
            gEffect->clipReleaseImage(src.handle);
            gEffect->clipReleaseImage(dst.handle);
            showError(in, err);
            return kOfxStatFailed;
        }
        in->cachePlanes.swap(planes);
        in->cacheValid = true;
        in->cacheW = W;
        in->cacheH = H;
        in->cacheHash = hash;
        in->cacheParams = p;
    }

    // write the requested window
    const size_t plane = (size_t)W * H;
    const float *nx = in->cachePlanes.data();
    const float *ny = nx + plane, *nz = nx + 2 * plane, *mk = nx + 3 * plane, *dz = nx + 4 * plane;
    const int y0 = std::max(rw[1], dst.y1), y1 = std::min(rw[3], dst.y2);
    const int x0 = std::max(rw[0], dst.x1), x1 = std::min(rw[2], dst.x2);
    for (int y = y0; y < y1; ++y) {
        float *d = dst.row(y);
        int sy = src.y2 - 1 - y;  // daemon row index (top-first)
        bool rowValid = y >= src.y1 && y < src.y2;
        for (int x = x0; x < x1; ++x) {
            float *o = d + (size_t)(x - dst.x1) * dst.nComp;
            int sx = x - src.x1;
            if (!rowValid || sx < 0 || sx >= W) {
                for (int c = 0; c < dst.nComp; ++c) o[c] = 0.f;
                continue;
            }
            size_t i = (size_t)sy * W + sx;
            if (dst.nComp == 1) {
                o[0] = mk[i];
            } else if (p.output == 1) {
                o[0] = nx[i]; o[1] = ny[i]; o[2] = nz[i];
                if (dst.nComp == 4) o[3] = mk[i];
            } else {
                o[0] = o[1] = o[2] = dz[i];
                if (dst.nComp == 4) o[3] = mk[i];
            }
        }
    }

    gEffect->clipReleaseImage(src.handle);
    gEffect->clipReleaseImage(dst.handle);
    return result;
}

static OfxStatus pluginMain(const char *action, const void *handle, OfxPropertySetHandle inArgs,
                            OfxPropertySetHandle outArgs) {
    OfxImageEffectHandle effect = (OfxImageEffectHandle)handle;
    try {
        if (strcmp(action, kOfxActionLoad) == 0) return onLoad();
        if (strcmp(action, kOfxActionUnload) == 0) return onUnload();
        if (strcmp(action, kOfxActionDescribe) == 0) return describe(effect);
        if (strcmp(action, kOfxImageEffectActionDescribeInContext) == 0) return describeInContext(effect);
        if (strcmp(action, kOfxActionCreateInstance) == 0) return createInstance(effect);
        if (strcmp(action, kOfxActionDestroyInstance) == 0) return destroyInstance(effect);
        if (strcmp(action, kOfxImageEffectActionGetClipPreferences) == 0) return getClipPreferences(effect, outArgs);
        if (strcmp(action, kOfxImageEffectActionGetRegionsOfInterest) == 0) return getRegionsOfInterest(effect, inArgs, outArgs);
        if (strcmp(action, kOfxImageEffectActionRender) == 0) return render(effect, inArgs);
    } catch (const std::exception &e) {
        logf("exception in %s: %s", action, e.what());
        return kOfxStatFailed;
    } catch (...) {
        logf("unknown exception in %s", action);
        return kOfxStatFailed;
    }
    return kOfxStatReplyDefault;
}

static void setHost(OfxHost *host) { gHost = host; }

static OfxPlugin gPlugin = {
    kOfxImageEffectPluginApi,
    1,
    PLUGIN_ID,
    PLUGIN_VERSION_MAJOR,
    PLUGIN_VERSION_MINOR,
    setHost,
    pluginMain,
};

#ifndef _WIN32
#define OFX_VISIBLE __attribute__((visibility("default")))
#else
#define OFX_VISIBLE
#endif

extern "C" {
OFX_VISIBLE OfxExport int OfxGetNumberOfPlugins(void) { return 1; }
OFX_VISIBLE OfxExport OfxPlugin *OfxGetPlugin(int nth) { return nth == 0 ? &gPlugin : nullptr; }
}
