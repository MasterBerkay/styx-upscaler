// StyxUpscaler - ReShade add-on for Styx: Master of Shadows (UE3, -dx11)
//
// Stage 1: fix the invisible menus under DX11.
// In windowed mode IDXGISwapChain::ResizeBuffers fails, so the back buffer keeps the requested
// resolution (e.g. 2560x1440) while UE3 sizes its Scaleform stencil buffer to the window client
// area (e.g. 2508x1411). D3D11 silently drops OMSetRenderTargets calls whose RTV and DSV differ
// in size, so after a Scaleform filter pass the back buffer is never re-bound and the rest of the
// UI lands in an offscreen texture. We substitute a depth-stencil of matching size whenever the
// game binds such a mismatched pair, and mirror the game's clears onto it.

#define IMGUI_DISABLE_INCLUDE_IMCONFIG_H
#define ImTextureID ImU64
#include <imgui.h>  // Dear ImGui 1.92.5 headers (MIT): ReShade provides the implementation for overlays
#include <reshade.hpp>
#include <d3d11.h>
#include <d3d11shader.h>
#include "ffx_antilag2_dx11.h"  // AMD Anti-Lag 2 SDK (MIT), header-only
#include <cstddef>
#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdarg>
#include <cstdio>
#include <cstring>
#include <mutex>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

using namespace reshade::api;

extern "C" __declspec(dllexport) const char* NAME = "Styx Upscaler";
extern "C" __declspec(dllexport) const char* DESCRIPTION =
    "FSR 4 / DLSS / XeSS for Styx: Master of Shadows (DX11) through OptiScaler, plus DX11 menu fix and AMD Anti-Lag 2.";

static void Log(reshade::log::level level, const char* fmt, ...)
{
    char buf[512];
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(buf, sizeof(buf), fmt, ap);
    va_end(ap);
    reshade::log::message(level, buf);
}
#define LOGI(...) Log(reshade::log::level::info, __VA_ARGS__)
#define LOGW(...) Log(reshade::log::level::warning, __VA_ARGS__)

// Where a vertex shader keeps the view transform (VSOffsetConstants), see ReflectVertexShader.
struct VsCbInfo {
    int slot = -1;
    uint32_t vpOffset = 0, pvtOffset = 0;
};

// GPU-skinned vertex shaders (characters): where to find LocalToWorld / MeshOrigin / MeshExtension in $Globals (b0).
struct SkinInfo {
    uint32_t l2w = 0, origin = 0, extension = 0;
};

// Where the final pass (UE3 gamma pass: HDR scene color -> back buffer) keeps its constants (all in $Globals).
struct FinalPsInfo {
    uint32_t slot = 0, colorScale = 0, overlayColor = 0, inverseGamma = 0;
};

// What the draw callbacks need to know about a bound shader. Looked up once when the shader is bound,
// so the ~3000 draws per frame need no lock or map lookup.
struct PipeInfo {
    VsCbInfo cb;
    SkinInfo skin;
    uint32_t rigidL2W = 0;
    bool skinned = false, rigid = false, isFinal = false, haveFinalInfo = false;
    FinalPsInfo fpi;
};

// Currently bound state per command list (only what the add-on needs). Index/vertex buffers, viewport and
// output-merger states are read from the context on the few draws that need them instead of tracked.
struct BindState {
    uint64_t vs = 0, ps = 0;
    PipeInfo vsInfo, psInfo;
    uint32_t vsGen = 0, psGen = 0;
    uint64_t vsCb[14] = {};
    resource_view psSrv0 = {0};
    resource_view rtv = {0};
    resource_view dsv = {0};
    resource_view rtvs[8] = {};
    uint32_t rtvCount = 0;
    uint64_t dsvRes = 0;     // resource behind dsv (resolved once per bind, not per draw)
    bool mainScene = false;  // HDR scene color + matching depth bound (base pass of the main view)
};
static std::unordered_map<command_list*, BindState> g_bind;

// Every callback needs the bind state of its command list; the game only uses the immediate context,
// so remember the last lookup (unordered_map references stay valid across rehashes).
static BindState& GetBind(command_list* cmd)
{
    static command_list* lastCmd = nullptr;
    static BindState* last = nullptr;
    if (cmd != lastCmd) {
        last = &g_bind[cmd];
        lastCmd = cmd;
    }
    return *last;
}

// View -> resource and its size/format, resolved once per view instead of on every bind/draw.
struct ViewInfo {
    uint64_t res = 0;
    uint32_t width = 0, height = 0;
    format fmt = format::unknown;
};
static std::unordered_map<uint64_t, ViewInfo> g_viewInfo;
static std::mutex g_viewMutex;  // views can be released on other threads

static ViewInfo GetViewInfo(device* dev, resource_view v)
{
    std::lock_guard<std::mutex> lock(g_viewMutex);
    auto it = g_viewInfo.find(v.handle);
    if (it != g_viewInfo.end())
        return it->second;
    ViewInfo vi;
    resource r = dev->get_resource_from_view(v);
    vi.res = r.handle;
    if (r.handle) {
        resource_desc d = dev->get_resource_desc(r);
        vi.width = d.texture.width;
        vi.height = d.texture.height;
        vi.fmt = d.texture.format;
    }
    return g_viewInfo.emplace(v.handle, vi).first->second;
}

// ---------------------------------------------------------------- profiling

static LARGE_INTEGER g_qpcFreq = {};
static int64_t g_addonTicks = 0, g_evalTicks = 0, g_evalMaxTicks = 0, g_frameMaxTicks = 0, g_frameTicksSum = 0;
static unsigned g_profFrames = 0, g_profSpikes = 0;
static int64_t g_pipeTicks = 0, g_pipeMaxTicks = 0;      // time spent in our init_pipeline analysis
static unsigned g_pipeCount = 0, g_pipesThisFrame = 0;    // shaders/states created
static unsigned g_spikesWithPipes = 0;                    // spike frames that also created shaders
static unsigned g_pipeKinds[3] = {};                      // vertex shaders, pixel shaders, other objects created
static unsigned g_pipeUnique = 0;                         // shaders not seen before (analysed)
static unsigned g_layoutHits = 0, g_layoutMisses = 0;     // input layout cache
static bool g_debugKeys = false;  // [Debug] DeveloperMode: developer hotkeys and statistics in ReShade.log
static bool g_bypass = false;  // Ctrl+Shift+B: add-on does nothing but the menu fix (A/B stutter test)

static int64_t Now()
{
    LARGE_INTEGER t;
    QueryPerformanceCounter(&t);
    return t.QuadPart;
}

static double TicksToMs(int64_t t)
{
    if (!g_qpcFreq.QuadPart)
        QueryPerformanceFrequency(&g_qpcFreq);
    return 1000.0 * static_cast<double>(t) / static_cast<double>(g_qpcFreq.QuadPart);
}

static void ProfileFrameEnd()
{
    static int64_t lastPresent = 0;
    static ULONGLONG lastLog = 0;
    int64_t now = Now();
    if (lastPresent) {
        int64_t ft = now - lastPresent;
        g_frameTicksSum += ft;
        g_frameMaxTicks = std::max(g_frameMaxTicks, ft);
        if (TicksToMs(ft) > 25.0) {
            ++g_profSpikes;
            if (g_pipesThisFrame)
                ++g_spikesWithPipes;
        }
        ++g_profFrames;
    }
    g_pipesThisFrame = 0;
    lastPresent = now;
    if (g_debugKeys && GetTickCount64() - lastLog >= 5000 && g_profFrames) {
        lastLog = GetTickCount64();
        LOGI("perf%s: %u frames, avg %.2f ms, max %.2f ms, spikes>25ms %u | add-on CPU %.2f ms/frame | upscaler eval avg %.2f ms max %.2f ms",
             g_bypass ? " [BYPASS]" : "", g_profFrames, TicksToMs(g_frameTicksSum) / g_profFrames, TicksToMs(g_frameMaxTicks), g_profSpikes,
             TicksToMs(g_addonTicks) / g_profFrames, TicksToMs(g_evalTicks) / g_profFrames, TicksToMs(g_evalMaxTicks));
        if (g_pipeCount)
            LOGI("perf: %u pipelines created (VS %u, PS %u, other %u, new shaders %u), analysis %.2f ms total (max %.2f ms), "
                 "%u of the spikes created pipelines",
                 g_pipeCount, g_pipeKinds[0], g_pipeKinds[1], g_pipeKinds[2], g_pipeUnique, TicksToMs(g_pipeTicks),
                 TicksToMs(g_pipeMaxTicks), g_spikesWithPipes);
        if (g_layoutHits || g_layoutMisses)
            LOGI("perf: input layouts reused %u, created %u", g_layoutHits, g_layoutMisses);
        g_layoutHits = g_layoutMisses = 0;
        g_pipeCount = g_spikesWithPipes = g_pipeUnique = 0;
        g_pipeKinds[0] = g_pipeKinds[1] = g_pipeKinds[2] = 0;
        g_pipeTicks = g_pipeMaxTicks = 0;
        g_profFrames = g_profSpikes = 0;
        g_addonTicks = g_evalTicks = g_evalMaxTicks = g_frameMaxTicks = g_frameTicksSum = 0;
    }
}


// ---------------------------------------------------------------- frame trace (debug)
// Enabled by creating StyxUpscaler_trace.flag next to the game exe; dumps two frames ~35 s after start.

static bool g_traceEnabled = false;
static bool g_tracing = false;
static int g_traceFrames = 0;
static std::vector<std::string> g_trace;

static void Trace(const char* fmt, ...)
{
    if (!g_tracing)
        return;
    char buf[256];
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(buf, sizeof(buf), fmt, ap);
    va_end(ap);
    g_trace.emplace_back(buf);
}

static std::string GameDirFile(const char* name)
{
    char path[MAX_PATH];
    GetModuleFileNameA(nullptr, path, MAX_PATH);
    char* slash = strrchr(path, '\\');
    if (slash)
        slash[1] = '\0';
    return std::string(path) + name;
}

static void ViewDesc(device* dev, resource_view v, char* out, size_t n)
{
    if (!v.handle) {
        snprintf(out, n, "-");
        return;
    }
    resource_desc d = dev->get_resource_desc(dev->get_resource_from_view(v));
    snprintf(out, n, "%llx(%ux%u)", (unsigned long long)(v.handle & 0xffffff), d.texture.width, d.texture.height);
}

static void TraceFrameEnd()
{
    static ULONGLONG start = GetTickCount64();
    if (!g_traceEnabled)
        return;
    Trace("PRESENT");
    if (g_tracing && ++g_traceFrames == 2) {
        g_tracing = false;
        std::string all;
        for (const auto& line : g_trace) {
            all += line;
            all += '\n';
        }
        FILE* f = nullptr;
        if (fopen_s(&f, GameDirFile("StyxUpscaler_trace.txt").c_str(), "w") == 0 && f) {
            fwrite(all.data(), 1, all.size(), f);
            fclose(f);
        }
        LOGI("trace written (%zu lines)", g_trace.size());
    }
    if (!g_traceFrames && !g_tracing && GetTickCount64() - start > 35000) {
        g_tracing = true;
        g_trace.clear();
    }
}

// ---------------------------------------------------------------- depth-stencil size fix

struct Substitute {
    resource res = {0};
    resource_view dsv = {0};
    uint32_t width = 0, height = 0;
};

static std::recursive_mutex g_subMutex;  // recursive: destroying our views may re-enter via events
static std::unordered_map<uint64_t, Substitute> g_substitutes;  // key: game DSV handle
static unsigned g_substituteBinds = 0;

static bool TextureSize(device* dev, resource_view view, uint32_t* w, uint32_t* h)
{
    resource_desc d = dev->get_resource_desc(dev->get_resource_from_view(view));
    if (d.type != resource_type::texture_2d)
        return false;
    *w = d.texture.width;
    *h = d.texture.height;
    return true;
}

static resource_view GetSubstitute(device* dev, resource_view gameDsv, uint32_t w, uint32_t h)
{
    std::lock_guard<std::recursive_mutex> lock(g_subMutex);
    Substitute& sub = g_substitutes[gameDsv.handle];
    if (sub.dsv.handle && sub.width == w && sub.height == h)
        return sub.dsv;
    if (sub.dsv.handle) {
        dev->destroy_resource_view(sub.dsv);
        dev->destroy_resource(sub.res);
        sub = {};
    }
    format viewFormat = dev->get_resource_view_desc(gameDsv).format;
    resource_desc desc(w, h, 1, 1, format_to_typeless(viewFormat), 1, memory_heap::default_, resource_usage::depth_stencil);
    if (!dev->create_resource(desc, nullptr, resource_usage::depth_stencil_write, &sub.res)) {
        LOGW("failed to create substitute depth-stencil %ux%u", w, h);
        return {0};
    }
    if (!dev->create_resource_view(sub.res, resource_usage::depth_stencil, resource_view_desc(viewFormat), &sub.dsv)) {
        dev->destroy_resource(sub.res);
        sub = {};
        LOGW("failed to create substitute depth-stencil view");
        return {0};
    }
    sub.width = w;
    sub.height = h;
    LOGI("created substitute depth-stencil %ux%u (format %u)", w, h, (unsigned)viewFormat);
    return sub.dsv;
}

static void DestroySubstitutes(device* dev)
{
    std::lock_guard<std::recursive_mutex> lock(g_subMutex);
    for (auto& [key, sub] : g_substitutes) {
        if (sub.dsv.handle)
            dev->destroy_resource_view(sub.dsv);
        if (sub.res.handle)
            dev->destroy_resource(sub.res);
    }
    g_substitutes.clear();
}

static void OnBindRenderTargets(command_list* cmd, uint32_t count, const resource_view* rtvs, resource_view dsv)
{
    device* dev = cmd->get_device();
    resource_view rtv = (count > 0 && rtvs) ? rtvs[0] : resource_view{0};
    BindState& bs = GetBind(cmd);
    bs.rtv = rtv;
    bs.dsv = dsv;
    bs.rtvCount = count > 8 ? 8 : count;
    for (uint32_t i = 0; i < bs.rtvCount; ++i)
        bs.rtvs[i] = rtvs ? rtvs[i] : resource_view{0};
    bs.dsvRes = 0;
    bs.mainScene = false;
    if (g_tracing) {
        char a[64], b[64];
        ViewDesc(dev, rtv, a, sizeof(a));
        ViewDesc(dev, dsv, b, sizeof(b));
        Trace("OM rt=%s ds=%s", a, b);
    }
    if (!dsv.handle)
        return;
    const ViewInfo dvi = GetViewInfo(dev, dsv);
    bs.dsvRes = dvi.res;
    if (!rtv.handle)
        return;
    const ViewInfo rvi = GetViewInfo(dev, rtv);
    const uint32_t rw = rvi.width, rh = rvi.height, dw = dvi.width, dh = dvi.height;
    // The base pass writes the HDR scene color (RGBA16F) with the main depth buffer bound.
    bs.mainScene = rvi.fmt == format::r16g16b16a16_float && rw == dw && rh == dh && rw >= 640;
    if (!rw || !dw || (rw == dw && rh == dh))
        return;
    // The runtime drops this binding; re-issue it with a depth-stencil of the render target's size.
    resource_view sub = GetSubstitute(dev, dsv, rw, rh);
    if (!sub.handle)
        return;
    cmd->bind_render_targets_and_depth_stencil(count, rtvs, sub);
    if (g_substituteBinds++ < 5)
        LOGI("substituted depth-stencil: target %ux%u, game depth-stencil %ux%u", rw, rh, dw, dh);
}

static void PreserveSceneDepth(command_list* cmd, resource_view dsv);

static bool OnClearDsv(command_list* cmd, resource_view dsv, const float* depth, const uint8_t* stencil, uint32_t rectCount,
                       const rect* rects)
{
    PreserveSceneDepth(cmd, dsv);
    resource_view sub = {0};
    {
        std::lock_guard<std::recursive_mutex> lock(g_subMutex);
        auto it = g_substitutes.find(dsv.handle);
        if (it != g_substitutes.end())
            sub = it->second.dsv;
    }
    if (sub.handle)
        cmd->clear_depth_stencil_view(sub, depth, stencil, rectCount, rects);
    return false;
}

static void OnDestroyResourceView(device* dev, resource_view view)
{
    {
        std::lock_guard<std::mutex> viewLock(g_viewMutex);
        g_viewInfo.erase(view.handle);
    }
    std::lock_guard<std::recursive_mutex> lock(g_subMutex);
    auto it = g_substitutes.find(view.handle);
    if (it == g_substitutes.end())
        return;
    if (it->second.dsv.handle)
        dev->destroy_resource_view(it->second.dsv);
    if (it->second.res.handle)
        dev->destroy_resource(it->second.res);
    g_substitutes.erase(it);
}

static void DestroyMotionPass(device* dev);
static void DestroySkinVelocity(device* dev);
static void ShutdownUpscaler(device* dev);
static void ReleaseInputLayoutCache();
static void ShutdownLatency(device* dev);

static void OnDestroyDevice(device* dev)
{
    ShutdownLatency(dev);
    ReleaseInputLayoutCache();
    DestroySubstitutes(dev);
    DestroyMotionPass(dev);
    DestroySkinVelocity(dev);
    ShutdownUpscaler(dev);
}

// ---------------------------------------------------------------- camera tracking
// UE3 keeps the per-view transform in the VSOffsetConstants cbuffer:
//   ViewProjectionMatrix (translated world -> clip), CameraPositionVS, PreViewTranslation.
// We find its slot/offsets by reflecting vertex shaders, shadow constant buffer contents on
// map/update, and snapshot the values bound while the base pass draws into the HDR scene color.

struct Camera {
    float viewProj[16] = {};
    float preViewTranslation[4] = {};
    float cameraPosition[4] = {};
    resource depth = {0};  // main scene depth buffer
    uint32_t renderWidth = 0, renderHeight = 0;
    float jitterNdc[2] = {};  // sub-pixel jitter applied to this frame's projection (NDC units)
    bool valid = false;
};

// ---------------------------------------------------------------- jitter
// Halton(2,3) sub-pixel offsets added to the main view's projection. Applied by re-uploading the
// VSOffsetConstants buffer with a jittered ViewProjectionMatrix before main-view draws.
static bool g_jitterEnabled = false;
static uint32_t g_frameId = 1;
static uint32_t g_jitterIndex = 0;
static uint32_t g_jitterPhases = 8;
static float g_jitterPx[2] = {};   // in render pixels, range [-0.5, 0.5)
static float g_jitterNdc[2] = {};
static resource g_mainDepth = {0};  // learned from the main pass, used to recognise main-view draws
static unsigned g_jitterUploads = 0;
static bool g_upscalerActive = false;  // set each frame the upscaler ran; forces jitter on

static float Halton(uint32_t index, uint32_t base)
{
    float f = 1.0f, r = 0.0f;
    while (index > 0) {
        f /= static_cast<float>(base);
        r += f * static_cast<float>(index % base);
        index /= base;
    }
    return r;
}

static std::mutex g_camMutex;
static std::unordered_map<uint64_t, PipeInfo> g_pipeInfo;  // shader -> info, only shaders the add-on cares about
static std::atomic<uint32_t> g_pipeGen{1};                 // bumped on every change of g_pipeInfo
static uint32_t g_cbReadEnd = 128;  // see OnUpdateBuffer: only this prefix of each buffer is shadowed
static Camera g_camCur, g_camPrev, g_camPrev2;
static unsigned g_camSnapshots = 0;
static Camera g_camFirst;            // first main-pass camera of the frame
static unsigned g_camDistinct = 0;   // snapshots whose matrix differs from the first one
static unsigned g_camFrames = 0, g_camSameAsPrev = 0;  // frames with a camera / unchanged since last frame
static unsigned g_cbMaps = 0, g_cbUnmaps = 0, g_cbUpdates = 0, g_cbMapNull = 0;
static bool g_layoutLogged = false;

using PFN_D3DReflect = HRESULT(WINAPI*)(const void*, SIZE_T, REFIID, void**);

static PFN_D3DReflect LoadReflect()
{
    static PFN_D3DReflect fn = [] {
        HMODULE m = LoadLibraryA("d3dcompiler_47.dll");
        return m ? reinterpret_cast<PFN_D3DReflect>(GetProcAddress(m, "D3DReflect")) : nullptr;
    }();
    return fn;
}

static VsCbInfo ReflectVertexShader(const void* code, size_t size)
{
    VsCbInfo info;
    PFN_D3DReflect reflect = LoadReflect();
    ID3D11ShaderReflection* refl = nullptr;
    if (!reflect || FAILED(reflect(code, size, IID_ID3D11ShaderReflection, reinterpret_cast<void**>(&refl))))
        return info;

    // Survey: log the full layout of the first few GPU-skinned vertex shaders.
    static int skinnedLogged = 0;
    D3D11_SHADER_DESC sd;
    if (skinnedLogged < 3 && SUCCEEDED(refl->GetDesc(&sd))) {
        bool skinned = false;
        for (UINT i = 0; i < sd.ConstantBuffers && !skinned; ++i) {
            ID3D11ShaderReflectionConstantBuffer* c = refl->GetConstantBufferByIndex(i);
            D3D11_SHADER_BUFFER_DESC bd;
            if (SUCCEEDED(c->GetDesc(&bd)))
                for (UINT v = 0; v < bd.Variables; ++v) {
                    D3D11_SHADER_VARIABLE_DESC vd;
                    if (SUCCEEDED(c->GetVariableByIndex(v)->GetDesc(&vd)) && strcmp(vd.Name, "BoneMatrices") == 0)
                        skinned = true;
                }
        }
        if (skinned) {
            ++skinnedLogged;
            LOGI("skinned VS #%d: %u cbuffers, %u inputs, %u bound resources", skinnedLogged, sd.ConstantBuffers,
                 sd.InputParameters, sd.BoundResources);
            for (UINT i = 0; i < sd.BoundResources; ++i) {
                D3D11_SHADER_INPUT_BIND_DESC bd;
                if (SUCCEEDED(refl->GetResourceBindingDesc(i, &bd)))
                    LOGI("  bind %s type %d slot %u count %u", bd.Name, (int)bd.Type, bd.BindPoint, bd.BindCount);
            }
            for (UINT i = 0; i < sd.ConstantBuffers; ++i) {
                ID3D11ShaderReflectionConstantBuffer* c = refl->GetConstantBufferByIndex(i);
                D3D11_SHADER_BUFFER_DESC bd;
                if (FAILED(c->GetDesc(&bd)))
                    continue;
                LOGI("  cbuffer %s size %u", bd.Name, bd.Size);
                for (UINT v = 0; v < bd.Variables; ++v) {
                    D3D11_SHADER_VARIABLE_DESC vd;
                    if (SUCCEEDED(c->GetVariableByIndex(v)->GetDesc(&vd)) && (vd.uFlags & D3D_SVF_USED))
                        LOGI("    %s @%u size %u", vd.Name, vd.StartOffset, vd.Size);
                }
            }
            for (UINT i = 0; i < sd.InputParameters; ++i) {
                D3D11_SIGNATURE_PARAMETER_DESC pd;
                if (SUCCEEDED(refl->GetInputParameterDesc(i, &pd)))
                    LOGI("  input %s%u reg %u type %d mask 0x%x", pd.SemanticName, pd.SemanticIndex, pd.Register,
                         (int)pd.ComponentType, pd.Mask);
            }
        }
    }

    D3D11_SHADER_INPUT_BIND_DESC bind;
    ID3D11ShaderReflectionConstantBuffer* cb = refl->GetConstantBufferByName("VSOffsetConstants");
    if (cb && SUCCEEDED(refl->GetResourceBindingDescByName("VSOffsetConstants", &bind))) {
        D3D11_SHADER_VARIABLE_DESC vd;
        ID3D11ShaderReflectionVariable* vp = cb->GetVariableByName("ViewProjectionMatrix");
        ID3D11ShaderReflectionVariable* pvt = cb->GetVariableByName("PreViewTranslation");
        if (vp && SUCCEEDED(vp->GetDesc(&vd))) {
            info.slot = static_cast<int>(bind.BindPoint);
            info.vpOffset = vd.StartOffset;
            if (pvt && SUCCEEDED(pvt->GetDesc(&vd)))
                info.pvtOffset = vd.StartOffset;
            if (!g_layoutLogged) {
                g_layoutLogged = true;
                D3D11_SHADER_BUFFER_DESC bd;
                cb->GetDesc(&bd);
                LOGI("VSOffsetConstants: slot b%d, size %u, ViewProjectionMatrix @%u, PreViewTranslation @%u",
                     info.slot, bd.Size, info.vpOffset, info.pvtOffset);
                for (UINT i = 0; i < bd.Variables; ++i)
                    if (SUCCEEDED(cb->GetVariableByIndex(i)->GetDesc(&vd)))
                        LOGI("  %s @%u size %u", vd.Name, vd.StartOffset, vd.Size);
            }
        }
    }
    refl->Release();
    return info;
}


static bool Contains(const void* data, size_t size, const char* needle)
{
    size_t n = strlen(needle);
    const char* p = static_cast<const char*>(data);
    for (size_t i = 0; i + n <= size; ++i)
        if (memcmp(p + i, needle, n) == 0)
            return true;
    return false;
}

static bool ReflectSkin(const void* code, size_t size, SkinInfo* out)
{
    PFN_D3DReflect reflect = LoadReflect();
    ID3D11ShaderReflection* refl = nullptr;
    if (!reflect || FAILED(reflect(code, size, IID_ID3D11ShaderReflection, reinterpret_cast<void**>(&refl))))
        return false;
    bool ok = false;
    D3D11_SHADER_INPUT_BIND_DESC bones, globals;
    if (SUCCEEDED(refl->GetResourceBindingDescByName("VSBoneConstants", &bones)) && bones.BindPoint == 3 &&
        SUCCEEDED(refl->GetResourceBindingDescByName("$Globals", &globals)) && globals.BindPoint == 0) {
        ID3D11ShaderReflectionConstantBuffer* g = refl->GetConstantBufferByName("$Globals");
        D3D11_SHADER_VARIABLE_DESC a, b, c;
        if (SUCCEEDED(g->GetVariableByName("LocalToWorld")->GetDesc(&a)) &&
            SUCCEEDED(g->GetVariableByName("MeshOrigin")->GetDesc(&b)) &&
            SUCCEEDED(g->GetVariableByName("MeshExtension")->GetDesc(&c))) {
            out->l2w = a.StartOffset;
            out->origin = b.StartOffset;
            out->extension = c.StartOffset;
            ok = true;
        }
        // Our velocity shader only knows 4 influences: skip "extra influence" variants.
        D3D11_SHADER_DESC sd;
        if (ok && SUCCEEDED(refl->GetDesc(&sd)))
            for (UINT i = 0; i < sd.InputParameters; ++i) {
                D3D11_SIGNATURE_PARAMETER_DESC pd;
                if (SUCCEEDED(refl->GetInputParameterDesc(i, &pd)) && strcmp(pd.SemanticName, "BLENDINDICES") == 0 &&
                    pd.SemanticIndex > 0)
                    ok = false;
            }
    }
    refl->Release();
    return ok;
}

// Non-skinned vertex shaders with a LocalToWorld in $Globals (b0) and POSITION in v0.
static bool ReflectRigid(const void* code, size_t size, uint32_t* l2wOffset)
{
    PFN_D3DReflect reflect = LoadReflect();
    ID3D11ShaderReflection* refl = nullptr;
    if (!reflect || FAILED(reflect(code, size, IID_ID3D11ShaderReflection, reinterpret_cast<void**>(&refl))))
        return false;
    bool ok = false;
    D3D11_SHADER_INPUT_BIND_DESC globals, bones;
    D3D11_SHADER_DESC sd;
    if (FAILED(refl->GetResourceBindingDescByName("VSBoneConstants", &bones)) &&
        SUCCEEDED(refl->GetResourceBindingDescByName("$Globals", &globals)) && globals.BindPoint == 0 &&
        SUCCEEDED(refl->GetDesc(&sd))) {
        D3D11_SHADER_VARIABLE_DESC vd;
        ID3D11ShaderReflectionVariable* v = refl->GetConstantBufferByName("$Globals")->GetVariableByName("LocalToWorld");
        if (v && SUCCEEDED(v->GetDesc(&vd)) && (vd.uFlags & D3D_SVF_USED)) {
            for (UINT i = 0; i < sd.InputParameters; ++i) {
                D3D11_SIGNATURE_PARAMETER_DESC pd;
                if (SUCCEEDED(refl->GetInputParameterDesc(i, &pd)) && strcmp(pd.SemanticName, "POSITION") == 0 &&
                    pd.SemanticIndex == 0 && pd.Register == 0 && pd.ComponentType == D3D_REGISTER_COMPONENT_FLOAT32) {
                    *l2wOffset = vd.StartOffset;
                    ok = true;
                }
            }
        }
    }
    refl->Release();
    return ok;
}

// Everything the add-on needs to know about one shader, cached by bytecode.
struct ShaderAnalysis {
    bool isFinal = false, haveFinalInfo = false;
    FinalPsInfo fpi;
    VsCbInfo vsInfo;
    bool skinned = false, rigid = false;
    SkinInfo skin;
    uint32_t rigidL2W = 0;
};

// DXBC bytecode carries an MD5 checksum at bytes 4..19; together with the size it identifies the shader.
struct ShaderKey {
    uint64_t a = 0, b = 0, size = 0;
    bool operator==(const ShaderKey& o) const { return a == o.a && b == o.b && size == o.size; }
};
struct ShaderKeyHash {
    size_t operator()(const ShaderKey& k) const { return static_cast<size_t>(k.a ^ (k.b * 0x9E3779B97F4A7C15ull) ^ k.size); }
};

static ShaderKey MakeShaderKey(const void* code, size_t size)
{
    ShaderKey k;
    k.size = size;
    if (size >= 20 && memcmp(code, "DXBC", 4) == 0) {
        memcpy(&k.a, static_cast<const uint8_t*>(code) + 4, 8);
        memcpy(&k.b, static_cast<const uint8_t*>(code) + 12, 8);
    } else {  // not DXBC: FNV-1a over the bytes
        uint64_t h = 1469598103934665603ull;
        for (size_t i = 0; i < size; ++i)
            h = (h ^ static_cast<const uint8_t*>(code)[i]) * 1099511628211ull;
        k.a = h;
    }
    return k;
}

static std::unordered_map<ShaderKey, ShaderAnalysis, ShaderKeyHash> g_shaderCache;

// ---------------------------------------------------------------- input layout cache
// While the camera turns quickly UE3 re-creates up to ~200 identical input layouts per frame, and every
// one costs the driver a fetch-shader compile (frame spikes of 30-50 ms). Hand out the existing object
// for an identical request instead (input layouts are immutable, so sharing them is safe).

using PFN_CreateInputLayout = HRESULT(STDMETHODCALLTYPE*)(ID3D11Device*, const D3D11_INPUT_ELEMENT_DESC*, UINT,
                                                          const void*, SIZE_T, ID3D11InputLayout**);
static PFN_CreateInputLayout g_origCreateInputLayout = nullptr;
static std::mutex g_layoutMutex;
static std::unordered_map<ShaderKey, ID3D11InputLayout*, ShaderKeyHash> g_layoutCache;  // holds one reference each

static ShaderKey MakeLayoutKey(const D3D11_INPUT_ELEMENT_DESC* elems, UINT count, const void* code, SIZE_T size)
{
    ShaderKey k = MakeShaderKey(code, size);
    uint64_t h = 1469598103934665603ull;
    auto mix = [&h](const void* p, size_t n) {
        for (size_t i = 0; i < n; ++i)
            h = (h ^ static_cast<const uint8_t*>(p)[i]) * 1099511628211ull;
    };
    for (UINT i = 0; i < count; ++i) {
        if (elems[i].SemanticName)
            mix(elems[i].SemanticName, strlen(elems[i].SemanticName));
        mix(&elems[i].SemanticIndex, sizeof(D3D11_INPUT_ELEMENT_DESC) - offsetof(D3D11_INPUT_ELEMENT_DESC, SemanticIndex));
    }
    k.size ^= h * 31 + count;
    return k;
}

static HRESULT STDMETHODCALLTYPE HookCreateInputLayout(ID3D11Device* dev, const D3D11_INPUT_ELEMENT_DESC* elems, UINT count,
                                                      const void* code, SIZE_T size, ID3D11InputLayout** out)
{
    if (!out || !elems || !code)  // validation-only calls go straight through
        return g_origCreateInputLayout(dev, elems, count, code, size, out);
    ShaderKey key = MakeLayoutKey(elems, count, code, size);
    key.b ^= reinterpret_cast<uintptr_t>(dev) * 0x9E3779B97F4A7C15ull;  // layouts belong to one device
    {
        std::lock_guard<std::mutex> lock(g_layoutMutex);
        auto it = g_layoutCache.find(key);
        if (it != g_layoutCache.end()) {
            ++g_layoutHits;
            it->second->AddRef();
            *out = it->second;
            return S_OK;
        }
    }
    HRESULT hr = g_origCreateInputLayout(dev, elems, count, code, size, out);
    if (SUCCEEDED(hr) && *out) {
        std::lock_guard<std::mutex> lock(g_layoutMutex);
        ++g_layoutMisses;
        auto ins = g_layoutCache.emplace(key, *out);
        if (ins.second)
            (*out)->AddRef();
    }
    return hr;
}

static const int kCreateInputLayout = 11;  // IUnknown(3), CreateBuffer, CreateTexture1D/2D/3D, 4 view creators
static void** g_hookedVtbl = nullptr;

static bool WriteVtblEntry(void** slot, void* value)
{
    DWORD old;
    if (!VirtualProtect(slot, sizeof(void*), PAGE_READWRITE, &old))
        return false;
    *slot = value;
    VirtualProtect(slot, sizeof(void*), old, &old);
    return true;
}

static void InstallInputLayoutCache(ID3D11Device* dev)
{
    void** vtbl = *reinterpret_cast<void***>(dev);
    if (g_hookedVtbl || vtbl[kCreateInputLayout] == reinterpret_cast<void*>(&HookCreateInputLayout))
        return;  // all D3D11 devices share one vtable
    g_origCreateInputLayout = reinterpret_cast<PFN_CreateInputLayout>(vtbl[kCreateInputLayout]);
    if (!WriteVtblEntry(&vtbl[kCreateInputLayout], reinterpret_cast<void*>(&HookCreateInputLayout))) {
        LOGW("input layout cache: VirtualProtect failed");
        return;
    }
    g_hookedVtbl = vtbl;
    LOGI("input layout cache installed");
}

// ReShade may unload and reload the add-on while the game starts: never leave the vtable pointing into
// an unloaded module.
static void UninstallInputLayoutCache()
{
    if (g_hookedVtbl)
        WriteVtblEntry(&g_hookedVtbl[kCreateInputLayout], reinterpret_cast<void*>(g_origCreateInputLayout));
    g_hookedVtbl = nullptr;
}

static void ReleaseInputLayoutCache()
{
    std::lock_guard<std::mutex> lock(g_layoutMutex);
    for (auto& e : g_layoutCache)
        e.second->Release();
    g_layoutCache.clear();
}

// ---------------------------------------------------------------- latency
// [Latency] AntiLag2: AMD Anti-Lag 2 (Radeon driver, DX11 via amdxx64.dll). Its Update() call belongs right
// before the game polls input; the closest point an injected mod has is the present of the previous frame.
// [Latency] MaxFrameLatency: fallback when Anti-Lag 2 is unavailable (NVIDIA, Intel, old drivers): how many
// frames the CPU may queue ahead of the GPU (DXGI default 3); 0 leaves the game's setting alone.

static bool g_antiLag2Wanted = true;
static int g_maxFrameLatency = 1;
static AMD::AntiLag2DX11::Context g_antiLag2 = {};
static enum { kLatencyPending, kLatencyAntiLag2, kLatencyFallback } g_latencyMode = kLatencyPending;
static ID3D11Device* g_latencyDevice = nullptr;  // not owned; valid between init_device and destroy_device

static void ApplyMaxFrameLatency(ID3D11Device* d3d)
{
    IDXGIDevice1* dxgi = nullptr;
    if (g_maxFrameLatency > 0 && SUCCEEDED(d3d->QueryInterface(__uuidof(IDXGIDevice1), reinterpret_cast<void**>(&dxgi)))) {
        dxgi->SetMaximumFrameLatency(static_cast<UINT>(g_maxFrameLatency));
        dxgi->Release();
        LOGI("latency: maximum frame latency set to %d", g_maxFrameLatency);
    }
}

// Once per frame, at present. The driver module is only loaded once the device renders, so Anti-Lag 2 is
// initialised on the first frame rather than at device creation.
static void LatencyFrameEnd()
{
    if (!g_latencyDevice)
        return;
    if (g_latencyMode == kLatencyPending) {
        HRESULT hr = g_antiLag2Wanted ? AMD::AntiLag2DX11::Initialize(&g_antiLag2) : E_ABORT;
        if (hr == S_OK) {
            g_latencyMode = kLatencyAntiLag2;
            LOGI("latency: AMD Anti-Lag 2 active");
        } else {
            g_latencyMode = kLatencyFallback;
            if (g_antiLag2Wanted)
                LOGI("latency: AMD Anti-Lag 2 not available (0x%08x), using the frame queue limit", static_cast<unsigned>(hr));
            ApplyMaxFrameLatency(g_latencyDevice);
        }
    }
    if (g_latencyMode != kLatencyAntiLag2)
        return;
    // The driver inserts its latency-reducing wait inside this call: time it to show Anti-Lag 2 is working.
    static int64_t waitSum = 0, waitMax = 0;
    static unsigned calls = 0, okCount = 0;
    static ULONGLONG lastLog = GetTickCount64();
    const int64_t t0 = Now();
    const HRESULT hr = AMD::AntiLag2DX11::Update(&g_antiLag2, g_antiLag2Wanted, 0);  // switchable in the overlay
    const int64_t waited = Now() - t0;
    waitSum += waited;
    waitMax = std::max(waitMax, waited);
    ++calls;
    okCount += hr == S_OK;
    if (g_debugKeys && GetTickCount64() - lastLog >= 5000) {
        lastLog = GetTickCount64();
        LOGI("latency: Anti-Lag 2 %u updates (%u S_OK), driver wait avg %.3f ms max %.3f ms", calls, okCount,
             TicksToMs(waitSum) / calls, TicksToMs(waitMax));
        waitSum = waitMax = 0;
        calls = okCount = 0;
    }
}

static void ShutdownLatency(device* dev)
{
    if (!g_latencyDevice || reinterpret_cast<ID3D11Device*>(dev->get_native()) != g_latencyDevice)
        return;
    if (g_latencyMode == kLatencyAntiLag2)
        AMD::AntiLag2DX11::DeInitialize(&g_antiLag2);
    g_latencyDevice = nullptr;
    g_latencyMode = kLatencyPending;
}

static void OnInitDevice(device* dev)
{
    if (dev->get_api() != device_api::d3d11)
        return;
    auto* d3d = reinterpret_cast<ID3D11Device*>(dev->get_native());
    InstallInputLayoutCache(d3d);
}
static ShaderAnalysis AnalysePixelShader(const shader_desc* desc);
static ShaderAnalysis AnalyseVertexShader(const shader_desc* desc);

static void OnInitPipeline(device*, pipeline_layout, uint32_t count, const pipeline_subobject* subobjects, pipeline pipe)
{
    const int64_t t0 = g_debugKeys ? Now() : 0;
    struct AddPipeTicks {
        int64_t t0;
        ~AddPipeTicks()
        {
            if (!g_debugKeys)
                return;
            int64_t t = Now() - t0;
            g_pipeTicks += t;
            g_pipeMaxTicks = std::max(g_pipeMaxTicks, t);
            ++g_pipeCount;
            ++g_pipesThisFrame;
        }
    } addPipeTicks{t0};
    for (uint32_t i = 0; i < count; ++i)  // CreateInputLayout also reports the shader bytecode: nothing to learn
        if (subobjects[i].type == pipeline_subobject_type::input_layout) {
            ++g_pipeKinds[2];
            return;
        }
    for (uint32_t i = 0; i < count; ++i) {
        const bool isPs = subobjects[i].type == pipeline_subobject_type::pixel_shader;
        const bool isVs = subobjects[i].type == pipeline_subobject_type::vertex_shader;
        ++g_pipeKinds[isVs ? 0 : isPs ? 1 : 2];
        if (!(isPs || isVs) || !subobjects[i].data)
            continue;
        const auto* desc = static_cast<const shader_desc*>(subobjects[i].data);
        // The game re-creates the same shaders constantly: analyse each unique bytecode once.
        ShaderKey key = MakeShaderKey(desc->code, desc->code_size);
        ShaderAnalysis a;
        bool cached;
        {
            std::lock_guard<std::mutex> lock(g_camMutex);
            auto it = g_shaderCache.find(key);
            cached = it != g_shaderCache.end();
            if (cached)
                a = it->second;
        }
        if (!cached) {
            ++g_pipeUnique;
            a = isPs ? AnalysePixelShader(desc) : AnalyseVertexShader(desc);
            std::lock_guard<std::mutex> lock(g_camMutex);
            g_shaderCache[key] = a;
        }
        std::lock_guard<std::mutex> lock(g_camMutex);
        auto need = [](uint32_t end) { g_cbReadEnd = std::max(g_cbReadEnd, (end + 15) & ~15u); };
        if (a.haveFinalInfo) {
            need(a.fpi.colorScale + 16);
            need(a.fpi.overlayColor + 16);
            need(a.fpi.inverseGamma + 16);
        }
        if (a.vsInfo.slot >= 0) {
            need(a.vsInfo.vpOffset + 80);  // ViewProjectionMatrix + CameraPositionVS
            need(a.vsInfo.pvtOffset + 16);
        }
        if (a.skinned) {
            need(a.skin.l2w + 64);
            need(a.skin.origin + 16);
            need(a.skin.extension + 16);
        }
        if (a.rigid)
            need(a.rigidL2W + 64);
        // A new shader may reuse the address of a destroyed one: always replace or drop the old entry.
        PipeInfo pi;
        if (isPs) {
            pi.isFinal = a.isFinal;
            pi.haveFinalInfo = a.haveFinalInfo;
            pi.fpi = a.fpi;
        } else {
            pi.cb = a.vsInfo;
            pi.skinned = a.skinned;
            pi.skin = a.skin;
            pi.rigid = a.rigid;
            pi.rigidL2W = a.rigidL2W;
        }
        if (pi.isFinal || pi.skinned || pi.rigid || pi.cb.slot >= 0)
            g_pipeInfo[pipe.handle] = pi;
        else if (!g_pipeInfo.erase(pipe.handle))
            continue;
        g_pipeGen.fetch_add(1, std::memory_order_relaxed);
    }
}

static ShaderAnalysis AnalysePixelShader(const shader_desc* desc)
{
    ShaderAnalysis r;
    {
        {
            bool isFinal = Contains(desc->code, desc->code_size, "SceneColorTexture") &&
                           Contains(desc->code, desc->code_size, "ColorScale") &&
                           Contains(desc->code, desc->code_size, "InverseGamma");
            FinalPsInfo fpi;
            bool haveInfo = false;
            if (isFinal) {
                PFN_D3DReflect reflect = LoadReflect();
                ID3D11ShaderReflection* refl = nullptr;
                if (reflect && SUCCEEDED(reflect(desc->code, desc->code_size, IID_ID3D11ShaderReflection, reinterpret_cast<void**>(&refl)))) {
                    D3D11_SHADER_INPUT_BIND_DESC bd;
                    ID3D11ShaderReflectionConstantBuffer* g = refl->GetConstantBufferByName("$Globals");
                    D3D11_SHADER_VARIABLE_DESC a, b, c;
                    if (SUCCEEDED(refl->GetResourceBindingDescByName("$Globals", &bd)) &&
                        SUCCEEDED(g->GetVariableByName("ColorScale")->GetDesc(&a)) &&
                        SUCCEEDED(g->GetVariableByName("OverlayColor")->GetDesc(&b)) &&
                        SUCCEEDED(g->GetVariableByName("InverseGamma")->GetDesc(&c))) {
                        fpi.slot = bd.BindPoint;
                        fpi.colorScale = a.StartOffset;
                        fpi.overlayColor = b.StartOffset;
                        fpi.inverseGamma = c.StartOffset;
                        haveInfo = true;
                        LOGI("final pass PS: $Globals b%u, ColorScale @%u, OverlayColor @%u, InverseGamma @%u", fpi.slot,
                             fpi.colorScale, fpi.overlayColor, fpi.inverseGamma);
                    }
                    refl->Release();
                }
            }
            r.isFinal = isFinal;
            r.haveFinalInfo = haveInfo;
            r.fpi = fpi;
        }
    }
    return r;
}

static ShaderAnalysis AnalyseVertexShader(const shader_desc* desc)
{
    ShaderAnalysis r;
    r.vsInfo = ReflectVertexShader(desc->code, desc->code_size);
    r.skinned = ReflectSkin(desc->code, desc->code_size, &r.skin);
    r.rigid = !r.skinned && ReflectRigid(desc->code, desc->code_size, &r.rigidL2W);
    return r;
}

// Everything about one constant buffer in one map entry: the game updates ~3000 constant buffers per frame,
// so each update costs one lookup. Entries with size 0 remember "not a small constant buffer" (the
// description is read once per buffer, not per update). Guarded by g_camMutex.
struct CbState {
    uint32_t size = 0;
    uint32_t jitterFrame = 0;  // frame id its jittered copy belongs to; 0 = holds the game's data
    bool wanted = false;       // a draw callback asked for its contents: only these buffers are shadowed
    bool whole = false;        // VSBoneConstants (b3 of a skinned draw): shadowed whole, not only g_cbReadEnd
    void* mapped = nullptr;
    std::vector<uint8_t> shadow;
};
static std::unordered_map<uint64_t, CbState> g_cb;

// g_cbReadEnd is the highest byte the add-on ever reads from a shadowed buffer (bone buffers aside). It grows
// as shaders are analysed; copying only that prefix of each update saves most of the copying.

// The few buffers that are shadowed (typically 5 of ~240), readable without the lock: the game updates
// ~5000 constant buffers per frame and all other updates return right away. Written under g_camMutex.
static std::atomic<uint64_t> g_wantedCb[32];
static std::atomic<uint32_t> g_wantedCount{0};
static std::atomic<bool> g_wantedOverflow{false};  // more than 32: every update takes the slow path

static bool IsWantedCb(uint64_t handle)
{
    if (g_wantedOverflow.load(std::memory_order_relaxed))
        return true;
    const uint32_t n = g_wantedCount.load(std::memory_order_acquire);
    for (uint32_t i = 0; i < n; ++i)
        if (g_wantedCb[i].load(std::memory_order_relaxed) == handle)
            return true;
    return false;
}

static void AddWantedCb(uint64_t handle)
{
    const uint32_t n = g_wantedCount.load(std::memory_order_relaxed);
    if (n >= std::size(g_wantedCb)) {
        g_wantedOverflow = true;
        return;
    }
    g_wantedCb[n].store(handle, std::memory_order_relaxed);
    g_wantedCount.store(n + 1, std::memory_order_release);
}

static void RemoveWantedCb(uint64_t handle)
{
    const uint32_t n = g_wantedCount.load(std::memory_order_relaxed);
    for (uint32_t i = 0; i < n; ++i)
        if (g_wantedCb[i].load(std::memory_order_relaxed) == handle) {
            g_wantedCb[i].store(g_wantedCb[n - 1].load(std::memory_order_relaxed), std::memory_order_relaxed);
            g_wantedCount.store(n - 1, std::memory_order_release);
            return;
        }
}

static CbState& GetCb(device* dev, resource res)
{
    auto it = g_cb.find(res.handle);
    if (it != g_cb.end())
        return it->second;
    CbState& cb = g_cb[res.handle];
    resource_desc d = dev->get_resource_desc(res);
    if (d.type == resource_type::buffer && (d.usage & resource_usage::constant_buffer) == resource_usage::constant_buffer &&
        d.buffer.size <= 4096)
        cb.size = static_cast<uint32_t>(d.buffer.size);
    return cb;
}

static void MarkWanted(CbState& cb, uint64_t handle)
{
    if (cb.wanted || !cb.size)
        return;
    cb.wanted = true;
    AddWantedCb(handle);
}

// Shadowed contents of a constant buffer a draw callback wants to read, or null. The first request only
// marks the buffer, its contents are copied from the game's next update on. Caller holds g_camMutex.
static const std::vector<uint8_t>* Shadow(device* dev, uint64_t buffer)
{
    if (!buffer)
        return nullptr;
    CbState& cb = GetCb(dev, resource{buffer});
    if (!cb.wanted) {
        MarkWanted(cb, buffer);
        return nullptr;
    }
    return cb.shadow.empty() ? nullptr : &cb.shadow;
}

static uint64_t g_jitteredCb = 0, g_snapshotCb = 0;  // see ApplyJitter / SnapshotCamera (rendering thread only)

static void OnDestroyResource(device*, resource res)
{
    std::lock_guard<std::mutex> lock(g_camMutex);
    auto it = g_cb.find(res.handle);
    if (it == g_cb.end())
        return;
    if (it->second.wanted)
        RemoveWantedCb(res.handle);
    g_cb.erase(it);
    if (g_jitteredCb == res.handle)
        g_jitteredCb = 0;
    if (g_snapshotCb == res.handle)
        g_snapshotCb = 0;
}

static void OnMapBuffer(device* dev, resource res, uint64_t offset, uint64_t, map_access, void** data)
{
    std::lock_guard<std::mutex> lock(g_camMutex);
    CbState& cb = GetCb(dev, res);
    if (!cb.size)
        return;
    if (!data || !*data || offset != 0) {
        ++g_cbMapNull;
        return;
    }
    ++g_cbMaps;
    cb.mapped = *data;
}

static void OnUnmapBuffer(device*, resource res)
{
    std::lock_guard<std::mutex> lock(g_camMutex);
    auto it = g_cb.find(res.handle);
    if (it == g_cb.end() || !it->second.mapped)
        return;
    CbState& cb = it->second;
    ++g_cbUnmaps;
    cb.jitterFrame = 0;  // game wrote fresh (unjittered) data
    if (cb.wanted)
        cb.shadow.assign(static_cast<const uint8_t*>(cb.mapped), static_cast<const uint8_t*>(cb.mapped) + cb.size);
    cb.mapped = nullptr;
    if (g_jitteredCb == res.handle)
        g_jitteredCb = 0;
    if (g_snapshotCb == res.handle)
        g_snapshotCb = 0;
}

static bool g_inOwnUpload = false;  // our jittered uploads must not reach the shadows

static void ShadowUpdate(device* dev, const void* data, resource res, uint64_t offset, uint64_t size)
{
    if (g_debugKeys)
        ++g_cbUpdates;
    if (!IsWantedCb(res.handle))
        return;  // nobody reads this buffer (only a handful are ever read)
    // Game wrote fresh (unjittered) data: the cached jitter / camera state of this buffer is stale.
    if (g_jitteredCb == res.handle)
        g_jitteredCb = 0;
    if (g_snapshotCb == res.handle)
        g_snapshotCb = 0;
    std::lock_guard<std::mutex> lock(g_camMutex);
    CbState& cb = GetCb(dev, res);
    const uint64_t full = cb.size;
    if (!full)
        return;
    cb.jitterFrame = 0;
    if (!cb.wanted)
        return;
    if (cb.shadow.size() < full)
        cb.shadow.resize(full);
    if (size == 0 || size == UINT64_MAX)  // whole buffer (UpdateSubresource without a box)
        size = full - offset;
    if (offset + size > full)
        return;
    const uint64_t readEnd = cb.whole ? full : std::min<uint64_t>(full, g_cbReadEnd);
    if (offset < readEnd)
        memcpy(cb.shadow.data() + offset, data, static_cast<size_t>(std::min(size, readEnd - offset)));
}

static bool OnUpdateBuffer(device* dev, const void* data, resource res, uint64_t offset, uint64_t size)
{
    if (g_bypass || g_inOwnUpload || !data)
        return false;
    if (!g_debugKeys) {
        ShadowUpdate(dev, data, res, offset, size);
        return false;
    }
    const int64_t t0 = Now();
    ShadowUpdate(dev, data, res, offset, size);
    g_addonTicks += Now() - t0;
    return false;
}


// Refreshes the cached info of a bound shader; skipped when the same shader is bound again and no shader
// was created or destroyed in between.
static void BindShaderInfo(uint64_t handle, uint64_t& bound, uint32_t& gen, PipeInfo& info)
{
    const uint32_t g = g_pipeGen.load(std::memory_order_relaxed);
    if (bound == handle && gen == g)
        return;
    bound = handle;
    gen = g;
    info = {};
    if (!handle)
        return;
    std::lock_guard<std::mutex> lock(g_camMutex);
    auto it = g_pipeInfo.find(handle);
    if (it != g_pipeInfo.end())
        info = it->second;
}

static void OnBindPipeline(command_list* cmd, pipeline_stage stages, pipeline pipe)
{
    BindState& s = GetBind(cmd);
    if ((stages & pipeline_stage::vertex_shader) == pipeline_stage::vertex_shader)
        BindShaderInfo(pipe.handle, s.vs, s.vsGen, s.vsInfo);
    if ((stages & pipeline_stage::pixel_shader) == pipeline_stage::pixel_shader)
        BindShaderInfo(pipe.handle, s.ps, s.psGen, s.psInfo);
}

static void OnPushDescriptors(command_list* cmd, shader_stage stages, pipeline_layout, uint32_t,
                              const descriptor_table_update& u)
{
    BindState& s = GetBind(cmd);
    if ((stages & shader_stage::pixel) == shader_stage::pixel && u.type == descriptor_type::shader_resource_view &&
        u.binding == 0 && u.count > 0) {
        s.psSrv0 = static_cast<const resource_view*>(u.descriptors)[0];
        return;
    }
    if ((stages & shader_stage::vertex) != shader_stage::vertex || u.type != descriptor_type::constant_buffer)
        return;
    const auto* ranges = static_cast<const buffer_range*>(u.descriptors);
    for (uint32_t i = 0; i < u.count; ++i)
        if (u.binding + i < 14)
            s.vsCb[u.binding + i] = ranges[i].buffer.handle;
}

static void SnapshotCamera(command_list* cmd)
{
    BindState& s = GetBind(cmd);
    const VsCbInfo& vi = s.vsInfo.cb;
    if (!s.mainScene || vi.slot < 0 || vi.slot >= 14)
        return;
    const uint64_t buffer = s.vsCb[vi.slot];
    if (buffer == g_snapshotCb && g_camCur.valid && g_camCur.depth.handle == s.dsvRes)
        return;  // same view constants as the last snapshot this frame (most main-pass draws)
    D3D11_VIEWPORT vp = {};
    UINT numVp = 1;
    reinterpret_cast<ID3D11DeviceContext*>(cmd->get_native())->RSGetViewports(&numVp, &vp);
    if (!numVp || vp.Width < 256.0f || vp.Height < 144.0f)
        return;  // tiny scene captures share the depth size but are not the main view
    std::lock_guard<std::mutex> lock(g_camMutex);
    const std::vector<uint8_t>* sh = Shadow(cmd->get_device(), buffer);
    if (!sh || sh->size() < vi.vpOffset + 64)
        return;
    g_snapshotCb = buffer;
    memcpy(g_camCur.viewProj, sh->data() + vi.vpOffset, 64);
    if (!g_camFirst.valid) {
        memcpy(g_camFirst.viewProj, g_camCur.viewProj, 64);
        g_camFirst.valid = true;
    } else {
        for (int k = 0; k < 16; ++k)
            if (fabsf(g_camFirst.viewProj[k] - g_camCur.viewProj[k]) > 1e-4f) {
                ++g_camDistinct;
                break;
            }
    }
    if (vi.pvtOffset && sh->size() >= vi.pvtOffset + 16) {
        memcpy(g_camCur.preViewTranslation, sh->data() + vi.pvtOffset, 16);
        memcpy(g_camCur.cameraPosition, sh->data() + vi.vpOffset + 64, 16);  // CameraPositionVS @64
    }
    g_camCur.valid = true;
    g_camCur.depth = resource{s.dsvRes};
    memcpy(g_camCur.jitterNdc, g_jitterNdc, sizeof(g_jitterNdc));
    g_camCur.renderWidth = static_cast<uint32_t>(vp.Width + 0.5f);
    g_camCur.renderHeight = static_cast<uint32_t>(vp.Height + 0.5f);
    ++g_camSnapshots;
}

static void JitterFrameEnd(uint32_t renderWidth, uint32_t renderHeight)
{
    ++g_frameId;
    const bool jitter = g_jitterEnabled || g_upscalerActive;
    g_upscalerActive = false;
    if (!jitter || !renderWidth || !renderHeight) {
        g_jitterPx[0] = g_jitterPx[1] = g_jitterNdc[0] = g_jitterNdc[1] = 0.0f;
        return;
    }
    g_jitterIndex = g_jitterIndex % g_jitterPhases + 1;
    g_jitterPx[0] = Halton(g_jitterIndex, 2) - 0.5f;
    g_jitterPx[1] = Halton(g_jitterIndex, 3) - 0.5f;
    g_jitterNdc[0] = 2.0f * g_jitterPx[0] / static_cast<float>(renderWidth);
    g_jitterNdc[1] = -2.0f * g_jitterPx[1] / static_cast<float>(renderHeight);
}

// Before a main-view draw: make sure the bound VSOffsetConstants carries this frame's jitter.
static void ApplyJitter(command_list* cmd)
{
    if (g_jitterNdc[0] == 0.0f && g_jitterNdc[1] == 0.0f)
        return;
    BindState& s = GetBind(cmd);
    const VsCbInfo& vi = s.vsInfo.cb;
    // g_mainDepth only changes at present, on this thread: no lock needed for the common early-out.
    if (vi.slot < 0 || vi.slot >= 14 || !g_mainDepth.handle || s.dsvRes != g_mainDepth.handle)
        return;  // not the main view (shadow maps, downsampled passes, UI)
    const uint64_t buffer = s.vsCb[vi.slot];
    static uint32_t jitteredFrame = 0;
    if (buffer == g_jitteredCb && jitteredFrame == g_frameId)
        return;  // already jittered for this frame and not rewritten since (most main-view draws)
    device* dev = cmd->get_device();
    std::vector<uint8_t> data;
    const uint32_t vpOffset = vi.vpOffset;
    {
        std::lock_guard<std::mutex> lock(g_camMutex);
        const std::vector<uint8_t>* sh = Shadow(dev, buffer);
        if (!sh)
            return;
        CbState& cb = g_cb[buffer];
        if (cb.jitterFrame == g_frameId)
            return;  // already jittered for this frame
        if (sh->size() < vpOffset + 64 || sh->size() > g_cbReadEnd)
            return;  // (re-uploading needs the whole buffer, and only the first g_cbReadEnd bytes are shadowed)
        data = *sh;
        cb.jitterFrame = g_frameId;
    }
    g_jitteredCb = buffer;
    jitteredFrame = g_frameId;
    // clip.xy += jitter * clip.w  ->  add jitter * (w column) to the x and y columns.
    float* m = reinterpret_cast<float*>(data.data() + vpOffset);
    for (int r = 0; r < 4; ++r) {
        m[r * 4 + 0] += g_jitterNdc[0] * m[r * 4 + 3];
        m[r * 4 + 1] += g_jitterNdc[1] * m[r * 4 + 3];
    }
    g_inOwnUpload = true;
    dev->update_buffer_region(data.data(), resource{buffer}, 0, data.size());
    g_inOwnUpload = false;
    ++g_jitterUploads;
}

static void LogSkinVelocityStats();

static void CameraFrameEnd()
{
    static ULONGLONG last = 0;
    std::lock_guard<std::mutex> lock(g_camMutex);
    ULONGLONG now = GetTickCount64();
    if (g_debugKeys && now - last >= 5000 && g_camCur.valid) {
        last = now;
        const float* m = g_camCur.viewProj;
        size_t shadowed = 0;
        for (const auto& e : g_cb)
            shadowed += e.second.wanted;
        LOGI("cbuffers: maps %u (null %u), unmaps %u, updates %u, tracked %zu, shadowed %zu", g_cbMaps, g_cbMapNull,
             g_cbUnmaps, g_cbUpdates, g_cb.size(), shadowed);
        g_cbMaps = g_cbMapNull = g_cbUnmaps = g_cbUpdates = 0;
        LOGI("camera frames %u, unchanged vs previous frame %u | jitter %s, uploads %u, last (%.3f, %.3f) px",
             g_camFrames, g_camSameAsPrev, g_jitterEnabled ? "on" : "off", g_jitterUploads, g_jitterPx[0], g_jitterPx[1]);
        g_jitterUploads = 0;
        LogSkinVelocityStats();
        g_camFrames = g_camSameAsPrev = 0;
        LOGI("camera: %u snapshots | VP r0 %.3f %.3f %.3f %.3f | r1 %.3f %.3f %.3f %.3f | r2 %.4f %.4f %.4f %.4f | r3 %.2f %.2f %.2f %.2f | PVT %.1f %.1f %.1f | CamPosVS %.1f %.1f %.1f %.3f",
             g_camSnapshots, m[0], m[1], m[2], m[3], m[4], m[5], m[6], m[7], m[8], m[9], m[10], m[11], m[12], m[13], m[14],
             m[15], g_camCur.preViewTranslation[0], g_camCur.preViewTranslation[1], g_camCur.preViewTranslation[2],
             g_camCur.cameraPosition[0], g_camCur.cameraPosition[1], g_camCur.cameraPosition[2], g_camCur.cameraPosition[3]);
    }
    if (g_camDistinct && now - last < 50) {
        const float* a = g_camFirst.viewProj;
        const float* b = g_camCur.viewProj;
        LOGI("camera: %u of %u snapshots differ from the first | first r0 %.3f %.3f %.3f %.3f r2 %.4f %.4f %.4f %.4f | last r0 %.3f %.3f %.3f %.3f r2 %.4f %.4f %.4f %.4f",
             g_camDistinct, g_camSnapshots, a[0], a[1], a[2], a[3], a[8], a[9], a[10], a[11], b[0], b[1], b[2], b[3], b[8], b[9], b[10], b[11]);
    }
    if (g_camCur.valid) {
        ++g_camFrames;
        if (g_camPrev.valid && memcmp(g_camCur.viewProj, g_camPrev.viewProj, 64) == 0 &&
            memcmp(g_camCur.preViewTranslation, g_camPrev.preViewTranslation, 12) == 0)
            ++g_camSameAsPrev;
    }
    g_camDistinct = 0;
    g_camFirst.valid = false;
    g_camSnapshots = 0;
    if (g_camCur.valid)
        g_mainDepth = g_camCur.depth;
    JitterFrameEnd(g_camCur.valid ? g_camCur.renderWidth : g_camPrev.renderWidth,
                   g_camCur.valid ? g_camCur.renderHeight : g_camPrev.renderHeight);
    g_camPrev2 = g_camPrev;
    g_camPrev = g_camCur;
    g_camCur.valid = false;
}

// ---------------------------------------------------------------- skinned velocity
// Before the game draws a skinned mesh in the main view, draw it once more into our velocity target
// with the current and the previous frame's bones / LocalToWorld / view projection.

#include "skin_velocity_vs.h"
#include "skin_velocity_ps.h"
#include "rigid_velocity_vs.h"

struct SkinVelocityParams {
    float curL2W[16], prevL2W[16], curVP[16], prevVP[16], jitVP[16];
    float meshOrigin[4], meshExtension[4];
};

struct SkinHistory {
    uint32_t frame = 0;
    float l2w[16];
    uint8_t bones[3600];
};

struct SkinVelocity {
    pipeline vs = {0}, rigidVs = {0}, ps = {0}, depthState = {0}, blendState = {0};
    resource params = {0}, prevBones = {0};
    resource tex = {0}, depth = {0};
    resource_view rtv = {0}, srv = {0}, dsv = {0};
    uint32_t width = 0, height = 0;
    uint32_t clearedFrame = 0;
    bool failed = false;
    unsigned draws = 0, withHistory = 0;
};
static SkinVelocity g_sv;
static const float kOne = 1.0f;
static std::unordered_map<uint64_t, SkinHistory> g_skinHistory;
static bool g_skinVelocityEnabled = true;

static void DestroySkinVelocity(device* dev)
{
    if (g_sv.rtv.handle) dev->destroy_resource_view(g_sv.rtv);
    if (g_sv.srv.handle) dev->destroy_resource_view(g_sv.srv);
    if (g_sv.tex.handle) dev->destroy_resource(g_sv.tex);
    if (g_sv.dsv.handle) dev->destroy_resource_view(g_sv.dsv);
    if (g_sv.depth.handle) dev->destroy_resource(g_sv.depth);
    if (g_sv.params.handle) dev->destroy_resource(g_sv.params);
    if (g_sv.prevBones.handle) dev->destroy_resource(g_sv.prevBones);
    if (g_sv.vs.handle) dev->destroy_pipeline(g_sv.vs);
    if (g_sv.rigidVs.handle) dev->destroy_pipeline(g_sv.rigidVs);
    if (g_sv.ps.handle) dev->destroy_pipeline(g_sv.ps);
    if (g_sv.depthState.handle) dev->destroy_pipeline(g_sv.depthState);
    if (g_sv.blendState.handle) dev->destroy_pipeline(g_sv.blendState);
    g_sv = {};
    g_skinHistory.clear();
}

static bool EnsureSkinVelocity(device* dev, uint32_t width, uint32_t height)
{
    if (g_sv.failed)
        return false;
    if (!g_sv.vs.handle) {
        shader_desc vsd = {};
        vsd.code = g_SkinVelocityVS;
        vsd.code_size = sizeof(g_SkinVelocityVS);
        shader_desc psd = {};
        psd.code = g_SkinVelocityPS;
        psd.code_size = sizeof(g_SkinVelocityPS);
        depth_stencil_desc dsd = {};
        // Own depth buffer: nearest skinned surface wins; visibility in the final scene is checked later.
        dsd.depth_enable = true;
        dsd.depth_write_mask = true;
        dsd.depth_func = compare_op::less;
        dsd.stencil_enable = false;
        blend_desc bd = {};
        pipeline_subobject vso = {pipeline_subobject_type::vertex_shader, 1, &vsd};
        shader_desc rvsd = {};
        rvsd.code = g_RigidVelocityVS;
        rvsd.code_size = sizeof(g_RigidVelocityVS);
        pipeline_subobject rvso = {pipeline_subobject_type::vertex_shader, 1, &rvsd};
        pipeline_subobject pso = {pipeline_subobject_type::pixel_shader, 1, &psd};
        pipeline_subobject dso = {pipeline_subobject_type::depth_stencil_state, 1, &dsd};
        pipeline_subobject bso = {pipeline_subobject_type::blend_state, 1, &bd};
        if (!dev->create_pipeline({0}, 1, &vso, &g_sv.vs) || !dev->create_pipeline({0}, 1, &rvso, &g_sv.rigidVs) ||
            !dev->create_pipeline({0}, 1, &pso, &g_sv.ps) ||
            !dev->create_pipeline({0}, 1, &dso, &g_sv.depthState) || !dev->create_pipeline({0}, 1, &bso, &g_sv.blendState) ||
            !dev->create_resource(resource_desc(sizeof(SkinVelocityParams), memory_heap::default_, resource_usage::constant_buffer),
                                  nullptr, resource_usage::constant_buffer, &g_sv.params) ||
            !dev->create_resource(resource_desc(3600, memory_heap::default_, resource_usage::constant_buffer), nullptr,
                                  resource_usage::constant_buffer, &g_sv.prevBones)) {
            LOGW("skin velocity: failed to create pipeline objects");
            g_sv.failed = true;
            return false;
        }
        LOGI("skin velocity: pipeline ready");
    }
    if (g_sv.width != width || g_sv.height != height) {
        if (g_sv.rtv.handle) dev->destroy_resource_view(g_sv.rtv);
        if (g_sv.srv.handle) dev->destroy_resource_view(g_sv.srv);
        if (g_sv.tex.handle) dev->destroy_resource(g_sv.tex);
        if (g_sv.dsv.handle) dev->destroy_resource_view(g_sv.dsv);
        if (g_sv.depth.handle) dev->destroy_resource(g_sv.depth);
        g_sv.rtv = g_sv.srv = g_sv.dsv = {0};
        g_sv.tex = g_sv.depth = {0};
        resource_desc td(width, height, 1, 1, format::r32g32b32a32_float, 1, memory_heap::default_,
                         resource_usage::render_target | resource_usage::shader_resource);
        if (!dev->create_resource(td, nullptr, resource_usage::render_target, &g_sv.tex) ||
            !dev->create_resource_view(g_sv.tex, resource_usage::render_target, resource_view_desc(format::r32g32b32a32_float), &g_sv.rtv) ||
            !dev->create_resource_view(g_sv.tex, resource_usage::shader_resource, resource_view_desc(format::r32g32b32a32_float), &g_sv.srv) ||
            !dev->create_resource(resource_desc(width, height, 1, 1, format::d32_float, 1, memory_heap::default_, resource_usage::depth_stencil),
                                  nullptr, resource_usage::depth_stencil_write, &g_sv.depth) ||
            !dev->create_resource_view(g_sv.depth, resource_usage::depth_stencil, resource_view_desc(format::d32_float), &g_sv.dsv)) {
            LOGW("skin velocity: failed to create %ux%u target", width, height);
            g_sv.failed = true;
            return false;
        }
        g_sv.width = width;
        g_sv.height = height;
        g_sv.clearedFrame = 0;
    }
    return true;
}

// Identifies a mesh section by its index buffer, first vertex buffer and draw range. The buffers are read
// from the context here (only for main-view skinned/rigid draws) instead of tracking every IA bind.
static uint64_t SkinKey(command_list* cmd, uint32_t indexCount, uint32_t firstIndex, int32_t vertexOffset)
{
    auto* ctx = reinterpret_cast<ID3D11DeviceContext*>(cmd->get_native());
    ID3D11Buffer* ib = nullptr;
    ID3D11Buffer* vb = nullptr;
    DXGI_FORMAT fmt;
    UINT offset, stride;
    ctx->IAGetIndexBuffer(&ib, &fmt, &offset);
    ctx->IAGetVertexBuffers(0, 1, &vb, &stride, &offset);
    const uint64_t ibHandle = reinterpret_cast<uintptr_t>(ib), vbHandle = reinterpret_cast<uintptr_t>(vb);
    if (ib)
        ib->Release();
    if (vb)
        vb->Release();
    uint64_t h = 1469598103934665603ull;
    auto mix = [&h](uint64_t v) { h = (h ^ v) * 1099511628211ull; };
    mix(ibHandle);
    mix(vbHandle);
    mix(indexCount);
    mix(firstIndex);
    mix(static_cast<uint32_t>(vertexOffset));
    return h;
}

// Draws the bound mesh once more into the velocity target with one of our vertex shaders, then puts
// back everything it changed (the game's draw follows right after with its own state).
static void DrawVelocityAndRestore(command_list* cmd, const BindState& s, pipeline vs, uint32_t indexCount,
                                   uint32_t instanceCount, uint32_t firstIndex, int32_t vertexOffset, uint32_t firstInstance)
{
    const uint64_t gameVs = s.vs, gamePs = s.ps;
    const buffer_range gameCb[2] = {{resource{s.vsCb[5]}, 0, UINT64_MAX}, {resource{s.vsCb[6]}, 0, UINT64_MAX}};
    const buffer_range ourCb[2] = {{g_sv.params, 0, UINT64_MAX}, {g_sv.prevBones, 0, UINT64_MAX}};
    // Output-merger states are read back from the context rather than tracked on every bind.
    auto* ctx = reinterpret_cast<ID3D11DeviceContext*>(cmd->get_native());
    ID3D11DepthStencilState* gameDs = nullptr;
    ID3D11BlendState* gameBlend = nullptr;
    UINT gameRef = 0, gameSampleMask = 0;
    FLOAT gameBlendFactor[4];
    ctx->OMGetDepthStencilState(&gameDs, &gameRef);
    ctx->OMGetBlendState(&gameBlend, gameBlendFactor, &gameSampleMask);

    cmd->bind_pipeline(pipeline_stage::vertex_shader, vs);
    cmd->bind_pipeline(pipeline_stage::pixel_shader, g_sv.ps);
    cmd->bind_pipeline(pipeline_stage::depth_stencil, g_sv.depthState);
    cmd->bind_pipeline(pipeline_stage::output_merger, g_sv.blendState);
    cmd->push_descriptors(shader_stage::vertex, {0}, 0, descriptor_table_update{{}, 5, 0, 2, descriptor_type::constant_buffer, ourCb});
    cmd->bind_render_targets_and_depth_stencil(1, &g_sv.rtv, g_sv.dsv);
    cmd->draw_indexed(indexCount, instanceCount, firstIndex, vertexOffset, firstInstance);

    cmd->bind_render_targets_and_depth_stencil(s.rtvCount, s.rtvs, s.dsv);
    cmd->push_descriptors(shader_stage::vertex, {0}, 0, descriptor_table_update{{}, 5, 0, 2, descriptor_type::constant_buffer, gameCb});
    cmd->bind_pipeline(pipeline_stage::vertex_shader, pipeline{gameVs});
    cmd->bind_pipeline(pipeline_stage::pixel_shader, pipeline{gamePs});
    ctx->OMSetDepthStencilState(gameDs, gameRef);
    ctx->OMSetBlendState(gameBlend, gameBlendFactor, gameSampleMask);
    if (gameDs)
        gameDs->Release();
    if (gameBlend)
        gameBlend->Release();
}

static void BeginVelocityFrame(command_list* cmd)
{
    if (g_sv.clearedFrame != g_frameId) {
        const float zero[4] = {0, 0, 0, 0};
        cmd->clear_render_target_view(g_sv.rtv, zero);
        cmd->clear_depth_stencil_view(g_sv.dsv, &kOne, nullptr);
        g_sv.clearedFrame = g_frameId;
    }
}

// Called from the draw_indexed event, before the game's draw executes.
static void DrawSkinVelocity(command_list* cmd, uint32_t indexCount, uint32_t instanceCount, uint32_t firstIndex,
                             int32_t vertexOffset, uint32_t firstInstance)
{
    if (!g_skinVelocityEnabled)
        return;
    BindState& s = GetBind(cmd);
    if (!s.vsInfo.skinned || !s.dsv.handle || s.rtvCount == 0)
        return;  // color passes only (skip depth-only and shadow passes)
    if (!g_mainDepth.handle || s.dsvRes != g_mainDepth.handle)
        return;
    device* dev = cmd->get_device();

    const SkinInfo skin = s.vsInfo.skin;
    SkinVelocityParams p = {};
    static uint8_t bones[3600];  // draw callbacks run on the rendering thread only
    Camera prevCam;
    {
        std::lock_guard<std::mutex> lock(g_camMutex);
        if (!s.vsCb[3])
            return;
        CbState& bc = GetCb(dev, resource{s.vsCb[3]});
        if (!bc.size)
            return;
        if (!bc.whole) {
            bc.whole = true;
            MarkWanted(bc, s.vsCb[3]);
            return;  // first sighting: copied whole from the next update on
        }
        const std::vector<uint8_t>* g0 = Shadow(dev, s.vsCb[0]);
        const std::vector<uint8_t>* b1 = Shadow(dev, s.vsCb[1]);
        const std::vector<uint8_t>* b3 = Shadow(dev, s.vsCb[3]);
        if (!g0 || !b1 || !b3 || b3->size() < 3600 || b1->size() < 64 || g0->size() < skin.extension + 12)
            return;
        memcpy(p.curL2W, g0->data() + skin.l2w, 64);
        memcpy(p.meshOrigin, g0->data() + skin.origin, 12);
        memcpy(p.meshExtension, g0->data() + skin.extension, 12);
        memcpy(p.curVP, b1->data(), 64);  // shadow = unjittered
        memcpy(bones, b3->data(), 3600);
        prevCam = g_camPrev;
    }
    if (!prevCam.valid)
        return;
    memcpy(p.prevVP, prevCam.viewProj, 64);
    memcpy(p.jitVP, p.curVP, 64);
    for (int r = 0; r < 4; ++r) {
        p.jitVP[r * 4 + 0] += g_jitterNdc[0] * p.jitVP[r * 4 + 3];
        p.jitVP[r * 4 + 1] += g_jitterNdc[1] * p.jitVP[r * 4 + 3];
    }

    // Previous frame's bones / LocalToWorld for this mesh section, then remember this frame's.
    uint64_t key = SkinKey(cmd, indexCount, firstIndex, vertexOffset);
    SkinHistory& hist = g_skinHistory[key];
    const uint8_t* prevBones = bones;
    memcpy(p.prevL2W, p.curL2W, 64);
    if (hist.frame != 0 && hist.frame + 1 == g_frameId) {
        prevBones = hist.bones;
        memcpy(p.prevL2W, hist.l2w, 64);
        ++g_sv.withHistory;
    } else if (hist.frame == g_frameId) {
        return;  // same section drawn twice this frame (e.g. another pass): already done
    }

    resource_desc dd = dev->get_resource_desc(g_mainDepth);
    if (!EnsureSkinVelocity(dev, dd.texture.width, dd.texture.height))
        return;
    BeginVelocityFrame(cmd);

    g_inOwnUpload = true;
    dev->update_buffer_region(&p, g_sv.params, 0, sizeof(p));
    dev->update_buffer_region(prevBones, g_sv.prevBones, 0, 3600);
    g_inOwnUpload = false;

    DrawVelocityAndRestore(cmd, s, g_sv.vs, indexCount, instanceCount, firstIndex, vertexOffset, firstInstance);

    hist.frame = g_frameId;
    memcpy(hist.l2w, p.curL2W, 64);
    memcpy(hist.bones, bones, 3600);
    ++g_sv.draws;
}

// ---------------------------------------------------------------- moving rigid meshes

// Every instance drawn this frame / last frame, grouped by geometry. A draw is "static" when last frame
// had an instance of the same geometry at exactly the same transform; otherwise the nearest one (if
// close enough) is taken as its previous position. Independent of draw order and culling changes.
struct RigidInstance {
    float l2w[16];   // translated LocalToWorld of that frame (pairs with that frame's view projection)
    float world[3];  // world-space translation, camera independent
};
// Instances of one geometry this frame and last frame: one map lookup per draw (~1600 per frame). The
// vectors keep their memory from frame to frame.
struct RigidGeometry {
    uint32_t frame = 0;  // frame `cur` belongs to
    std::vector<RigidInstance> cur, prev;
};
static std::unordered_map<uint64_t, RigidGeometry> g_rigid;
static uint32_t g_rigidFrame = 0;
static unsigned g_rigidDraws = 0, g_rigidChecked = 0;
static bool g_rigidVelocityEnabled = true;

// Squared distance between two instance transforms (translation in world units, rotation/scale scaled up).
static float InstanceDistance2(const float* l2wA, const float* worldA, const RigidInstance& b)
{
    float d2 = 0.0f;
    for (int i = 0; i < 3; ++i)
        d2 += (worldA[i] - b.world[i]) * (worldA[i] - b.world[i]);
    for (int i = 0; i < 12; ++i) {
        float d = (l2wA[i] - b.l2w[i]) * 100.0f;
        d2 += d * d;
    }
    return d2;
}

static void DrawRigidVelocity(command_list* cmd, uint32_t indexCount, uint32_t instanceCount, uint32_t firstIndex,
                              int32_t vertexOffset, uint32_t firstInstance)
{
    if (!g_rigidVelocityEnabled || !g_skinVelocityEnabled || instanceCount != 1)
        return;
    BindState& s = GetBind(cmd);
    if (!s.vsInfo.rigid || !s.dsv.handle || s.rtvCount == 0)
        return;
    if (!g_mainDepth.handle || s.dsvRes != g_mainDepth.handle || !g_camPrev.valid)
        return;  // (g_camPrev only changes at present, on this thread)
    device* dev = cmd->get_device();

    SkinVelocityParams p = {};
    float world[3];
    {
        std::lock_guard<std::mutex> lock(g_camMutex);
        const uint32_t l2w = s.vsInfo.rigidL2W;
        const std::vector<uint8_t>* g0 = Shadow(dev, s.vsCb[0]);
        const std::vector<uint8_t>* b1 = Shadow(dev, s.vsCb[1]);
        if (!g0 || !b1 || g0->size() < l2w + 64 || b1->size() < 96)
            return;
        memcpy(p.curL2W, g0->data() + l2w, 64);
        memcpy(p.curVP, b1->data(), 64);
        const float* pvt = reinterpret_cast<const float*>(b1->data() + 80);
        for (int i = 0; i < 3; ++i)
            world[i] = p.curL2W[12 + i] - pvt[i];
    }

    if (g_rigidFrame != g_frameId) {
        if (g_rigid.size() > 4096)  // level change: drop geometry that no longer exists
            g_rigid.clear();
        g_rigidFrame = g_frameId;
    }
    uint64_t base = SkinKey(cmd, indexCount, firstIndex, vertexOffset);
    RigidInstance inst;
    memcpy(inst.l2w, p.curL2W, 64);
    memcpy(inst.world, world, 12);
    ++g_rigidChecked;
    const RigidInstance* best = nullptr;
    float bestD2 = 1e30f;
    RigidGeometry& geo = g_rigid[base];
    if (geo.frame != g_frameId) {  // first draw of this geometry this frame
        if (geo.frame + 1 == g_frameId)
            geo.prev.swap(geo.cur);
        else
            geo.prev.clear();  // not drawn last frame: no history
        geo.cur.clear();
        geo.frame = g_frameId;
    }
    auto& curList = geo.cur;
    const auto& prevList = geo.prev;
    // Fast path: the game draws in a stable order, so the same slot last frame is usually this instance.
    // An exact match there means static; only otherwise search all candidates for the nearest.
    if (curList.size() < prevList.size() && memcmp(prevList[curList.size()].l2w, inst.l2w, sizeof(inst.l2w)) == 0) {
        curList.push_back(inst);
        return;
    }
    {
        for (const RigidInstance& cand : prevList) {
            float d2 = InstanceDistance2(inst.l2w, inst.world, cand);
            if (d2 < bestD2) {
                bestD2 = d2;
                best = &cand;
            }
        }
    }
    float prevL2W[16];
    bool moved = best && bestD2 > 1e-4f && bestD2 < 50.0f * 50.0f;  // exact match = static, far = unrelated
    if (moved)
        memcpy(prevL2W, best->l2w, 64);
    curList.push_back(inst);
    if (!moved)
        return;  // static this frame (or new): camera motion vectors are exact

    memcpy(p.prevL2W, prevL2W, 64);
    memcpy(p.prevVP, g_camPrev.viewProj, 64);
    memcpy(p.jitVP, p.curVP, 64);
    for (int r = 0; r < 4; ++r) {
        p.jitVP[r * 4 + 0] += g_jitterNdc[0] * p.jitVP[r * 4 + 3];
        p.jitVP[r * 4 + 1] += g_jitterNdc[1] * p.jitVP[r * 4 + 3];
    }
    resource_desc dd = dev->get_resource_desc(g_mainDepth);
    if (!EnsureSkinVelocity(dev, dd.texture.width, dd.texture.height))
        return;
    BeginVelocityFrame(cmd);
    g_inOwnUpload = true;
    dev->update_buffer_region(&p, g_sv.params, 0, sizeof(p));
    g_inOwnUpload = false;
    DrawVelocityAndRestore(cmd, s, g_sv.rigidVs, indexCount, instanceCount, firstIndex, vertexOffset, firstInstance);
    ++g_rigidDraws;
}

static void LogUpscalerStats();

static void LogSkinVelocityStats()
{
    LogUpscalerStats();
    LOGI("rigid velocity: %u moving draws of %u checked, %zu geometries", g_rigidDraws, g_rigidChecked, g_rigid.size());
    g_rigidDraws = g_rigidChecked = 0;
    LOGI("skin velocity: %u draws (%u with previous-frame data), %zu tracked sections", g_sv.draws, g_sv.withHistory,
         g_skinHistory.size());
    g_sv.draws = g_sv.withHistory = 0;
}

// ---------------------------------------------------------------- motion vector pass
// Runs right before UE3's final pass (HDR scene color -> back buffer, HUD follows). Produces
// camera motion vectors at render resolution; F10 toggles a debug overlay of them.

#include "motion_vectors_cs.h"

struct MotionParams {
    float invViewProj[16];
    float prevViewProj[16];
    float translationDelta[4];
    float renderSize[2];
    float pixelsForFullColor;
    float debugMode;
    float jitterNdc[2];
    float useObjectMotion;
    float pad;
};

struct MotionPass {
    pipeline_layout layout = {0};
    pipeline pipe = {0};
    resource motion = {0}, debug = {0}, history = {0};
    resource_view motionUav = {0}, debugUav = {0}, historySrv = {0};
    resource depthSrc = {0};
    resource_view depthSrv = {0};
    uint32_t width = 0, height = 0;
    format debugFormat = format::unknown;
    bool failed = false;
};
static MotionPass g_mv;
static int g_mvDebug = 0;  // 0 off, 1 vector colors, 2 reprojection (model), 3 reprojection bands
static bool g_mvDoneThisFrame = false;

// UE3 clears the scene depth after the main pass (before the foreground pass), so by the time the
// final pass runs it holds no scene depth. Copy it right before that clear.
struct DepthCopy {
    resource res = {0}, src = {0};
    bool copiedThisFrame = false;
    unsigned copies = 0;
};
static DepthCopy g_depthCopy;

static void PreserveSceneDepth(command_list* cmd, resource_view dsv)
{
    if (g_depthCopy.copiedThisFrame)
        return;
    resource depth;
    {
        std::lock_guard<std::mutex> lock(g_camMutex);
        if (!g_camCur.valid)
            return;  // main pass has not run yet this frame: this is the frame-start clear
        depth = g_camCur.depth;
    }
    device* dev = cmd->get_device();
    if (!depth.handle || dev->get_resource_from_view(dsv) != depth)
        return;
    if (g_depthCopy.src != depth || !g_depthCopy.res.handle) {
        if (g_depthCopy.res.handle)
            dev->destroy_resource(g_depthCopy.res);
        g_depthCopy.res = {0};
        resource_desc d = dev->get_resource_desc(depth);
        if (!dev->create_resource(d, nullptr, resource_usage::copy_dest, &g_depthCopy.res)) {
            LOGW("depth copy: failed to create %ux%u copy", d.texture.width, d.texture.height);
            return;
        }
        g_depthCopy.src = depth;
    }
    cmd->copy_resource(depth, g_depthCopy.res);
    g_depthCopy.copiedThisFrame = true;
    if (g_depthCopy.copies++ == 0)
        LOGI("depth copy: preserved scene depth before its mid-frame clear");
}
static unsigned g_mvRuns = 0;

static bool Invert4x4(const float* m, float* out)
{
    float inv[16];
    inv[0] = m[5] * m[10] * m[15] - m[5] * m[11] * m[14] - m[9] * m[6] * m[15] + m[9] * m[7] * m[14] + m[13] * m[6] * m[11] - m[13] * m[7] * m[10];
    inv[4] = -m[4] * m[10] * m[15] + m[4] * m[11] * m[14] + m[8] * m[6] * m[15] - m[8] * m[7] * m[14] - m[12] * m[6] * m[11] + m[12] * m[7] * m[10];
    inv[8] = m[4] * m[9] * m[15] - m[4] * m[11] * m[13] - m[8] * m[5] * m[15] + m[8] * m[7] * m[13] + m[12] * m[5] * m[11] - m[12] * m[7] * m[9];
    inv[12] = -m[4] * m[9] * m[14] + m[4] * m[10] * m[13] + m[8] * m[5] * m[14] - m[8] * m[6] * m[13] - m[12] * m[5] * m[10] + m[12] * m[6] * m[9];
    inv[1] = -m[1] * m[10] * m[15] + m[1] * m[11] * m[14] + m[9] * m[2] * m[15] - m[9] * m[3] * m[14] - m[13] * m[2] * m[11] + m[13] * m[3] * m[10];
    inv[5] = m[0] * m[10] * m[15] - m[0] * m[11] * m[14] - m[8] * m[2] * m[15] + m[8] * m[3] * m[14] + m[12] * m[2] * m[11] - m[12] * m[3] * m[10];
    inv[9] = -m[0] * m[9] * m[15] + m[0] * m[11] * m[13] + m[8] * m[1] * m[15] - m[8] * m[3] * m[13] - m[12] * m[1] * m[11] + m[12] * m[3] * m[9];
    inv[13] = m[0] * m[9] * m[14] - m[0] * m[10] * m[13] - m[8] * m[1] * m[14] + m[8] * m[2] * m[13] + m[12] * m[1] * m[10] - m[12] * m[2] * m[9];
    inv[2] = m[1] * m[6] * m[15] - m[1] * m[7] * m[14] - m[5] * m[2] * m[15] + m[5] * m[3] * m[14] + m[13] * m[2] * m[7] - m[13] * m[3] * m[6];
    inv[6] = -m[0] * m[6] * m[15] + m[0] * m[7] * m[14] + m[4] * m[2] * m[15] - m[4] * m[3] * m[14] - m[12] * m[2] * m[7] + m[12] * m[3] * m[6];
    inv[10] = m[0] * m[5] * m[15] - m[0] * m[7] * m[13] - m[4] * m[1] * m[15] + m[4] * m[3] * m[13] + m[12] * m[1] * m[7] - m[12] * m[3] * m[5];
    inv[14] = -m[0] * m[5] * m[14] + m[0] * m[6] * m[13] + m[4] * m[1] * m[14] - m[4] * m[2] * m[13] - m[12] * m[1] * m[6] + m[12] * m[2] * m[5];
    inv[3] = -m[1] * m[6] * m[11] + m[1] * m[7] * m[10] + m[5] * m[2] * m[11] - m[5] * m[3] * m[10] - m[9] * m[2] * m[7] + m[9] * m[3] * m[6];
    inv[7] = m[0] * m[6] * m[11] - m[0] * m[7] * m[10] - m[4] * m[2] * m[11] + m[4] * m[3] * m[10] + m[8] * m[2] * m[7] - m[8] * m[3] * m[6];
    inv[11] = -m[0] * m[5] * m[11] + m[0] * m[7] * m[9] + m[4] * m[1] * m[11] - m[4] * m[3] * m[9] - m[8] * m[1] * m[7] + m[8] * m[3] * m[5];
    inv[15] = m[0] * m[5] * m[10] - m[0] * m[6] * m[9] - m[4] * m[1] * m[10] + m[4] * m[2] * m[9] + m[8] * m[1] * m[6] - m[8] * m[2] * m[5];
    float det = m[0] * inv[0] + m[1] * inv[4] + m[2] * inv[8] + m[3] * inv[12];
    if (det == 0.0f)
        return false;
    for (int i = 0; i < 16; ++i)
        out[i] = inv[i] / det;
    return true;
}

static void DestroyMotionResources(device* dev)
{
    if (g_mv.motionUav.handle) dev->destroy_resource_view(g_mv.motionUav);
    if (g_mv.debugUav.handle) dev->destroy_resource_view(g_mv.debugUav);
    if (g_mv.depthSrv.handle) dev->destroy_resource_view(g_mv.depthSrv);
    if (g_mv.historySrv.handle) dev->destroy_resource_view(g_mv.historySrv);
    if (g_mv.history.handle) dev->destroy_resource(g_mv.history);
    g_mv.historySrv = {0};
    g_mv.history = {0};
    if (g_mv.motion.handle) dev->destroy_resource(g_mv.motion);
    if (g_mv.debug.handle) dev->destroy_resource(g_mv.debug);
    g_mv.motionUav = g_mv.debugUav = g_mv.depthSrv = {0};
    g_mv.motion = g_mv.debug = g_mv.depthSrc = {0};
    g_mv.width = g_mv.height = 0;
}

static void DestroyMotionPass(device* dev)
{
    DestroyMotionResources(dev);
    if (g_depthCopy.res.handle)
        dev->destroy_resource(g_depthCopy.res);
    g_depthCopy = {};
    if (g_mv.pipe.handle) dev->destroy_pipeline(g_mv.pipe);
    if (g_mv.layout.handle) dev->destroy_pipeline_layout(g_mv.layout);
    g_mv = {};
}

static bool EnsureMotionPass(device* dev, resource depth, resource sceneColor)
{
    if (g_mv.failed)
        return false;
    if (!g_mv.pipe.handle) {
        descriptor_range srvs = {};
        srvs.count = 4;
        srvs.visibility = shader_stage::compute;
        srvs.type = descriptor_type::shader_resource_view;
        descriptor_range uavs = {};
        uavs.count = 2;
        uavs.visibility = shader_stage::compute;
        uavs.type = descriptor_type::unordered_access_view;
        constant_range consts = {};
        consts.count = sizeof(MotionParams) / 4;
        consts.visibility = shader_stage::compute;
        const pipeline_layout_param params[3] = {pipeline_layout_param(srvs), pipeline_layout_param(uavs),
                                                 pipeline_layout_param(consts)};
        shader_desc cs = {};
        cs.code = g_MotionVectorsCS;
        cs.code_size = sizeof(g_MotionVectorsCS);
        pipeline_subobject sub = {pipeline_subobject_type::compute_shader, 1, &cs};
        if (!dev->create_pipeline_layout(3, params, &g_mv.layout) || !dev->create_pipeline(g_mv.layout, 1, &sub, &g_mv.pipe)) {
            LOGW("motion vectors: failed to create compute pipeline");
            g_mv.failed = true;
            return false;
        }
        LOGI("motion vectors: compute pipeline ready");
    }

    resource_desc dd = dev->get_resource_desc(depth);
    resource_desc cd = dev->get_resource_desc(sceneColor);
    if (g_mv.width != dd.texture.width || g_mv.height != dd.texture.height || g_mv.debugFormat != cd.texture.format) {
        DestroyMotionResources(dev);
        resource_desc md(dd.texture.width, dd.texture.height, 1, 1, format::r16g16_float, 1, memory_heap::default_,
                         resource_usage::unordered_access | resource_usage::shader_resource);
        resource_desc bd(cd.texture.width, cd.texture.height, 1, 1, cd.texture.format, 1, memory_heap::default_,
                         resource_usage::unordered_access | resource_usage::copy_source);
        if (!dev->create_resource(md, nullptr, resource_usage::unordered_access, &g_mv.motion) ||
            !dev->create_resource_view(g_mv.motion, resource_usage::unordered_access, resource_view_desc(format::r16g16_float), &g_mv.motionUav) ||
            !dev->create_resource(bd, nullptr, resource_usage::unordered_access, &g_mv.debug) ||
            !dev->create_resource_view(g_mv.debug, resource_usage::unordered_access, resource_view_desc(cd.texture.format), &g_mv.debugUav) ||
            !dev->create_resource(resource_desc(cd.texture.width, cd.texture.height, 1, 1, cd.texture.format, 1, memory_heap::default_,
                                                resource_usage::shader_resource | resource_usage::copy_dest),
                                  nullptr, resource_usage::shader_resource, &g_mv.history) ||
            !dev->create_resource_view(g_mv.history, resource_usage::shader_resource, resource_view_desc(cd.texture.format), &g_mv.historySrv)) {
            LOGW("motion vectors: failed to create textures (%ux%u)", dd.texture.width, dd.texture.height);
            DestroyMotionResources(dev);
            g_mv.failed = true;
            return false;
        }
        g_mv.width = dd.texture.width;
        g_mv.height = dd.texture.height;
        g_mv.debugFormat = cd.texture.format;
        LOGI("motion vectors: textures %ux%u", g_mv.width, g_mv.height);
    }
    if (g_mv.depthSrc != depth) {
        if (g_mv.depthSrv.handle)
            dev->destroy_resource_view(g_mv.depthSrv);
        g_mv.depthSrv = {0};
        if (!dev->create_resource_view(depth, resource_usage::shader_resource, resource_view_desc(format::r24_unorm_x8_uint), &g_mv.depthSrv)) {
            resource_desc dd2 = dev->get_resource_desc(depth);
            LOGW("motion vectors: cannot create depth SRV (format %u, %ux%u, samples %u, usage 0x%x)",
                 (unsigned)dd2.texture.format, dd2.texture.width, dd2.texture.height, dd2.texture.samples,
                 (unsigned)dd2.usage);
            g_mv.failed = true;
            return false;
        }
        g_mv.depthSrc = depth;
    }
    return true;
}

// ---------------------------------------------------------------- motion probe (F11)
// Reads back one frame and measures the real image shift between history and current color per
// tile (block matching), then compares it with the shift predicted by several camera hypotheses.

static bool g_probeRequested = false;

static float HalfToFloat(uint16_t h)
{
    uint32_t sign = (h >> 15) & 1, exp = (h >> 10) & 0x1f, mant = h & 0x3ff;
    float v;
    if (exp == 0)
        v = std::ldexp(static_cast<float>(mant), -24);
    else if (exp == 31)
        v = mant ? NAN : INFINITY;
    else
        v = std::ldexp(static_cast<float>(mant | 0x400), static_cast<int>(exp) - 25);
    return sign ? -v : v;
}

static bool Readback(command_list* cmd, device* dev, resource src, std::vector<uint8_t>& out, uint32_t& pitch)
{
    resource_desc sd = dev->get_resource_desc(src);
    resource_desc st(sd.texture.width, sd.texture.height, 1, 1, sd.texture.format, 1, memory_heap::readback,
                     resource_usage::copy_dest);
    resource staging = {0};
    if (!dev->create_resource(st, nullptr, resource_usage::copy_dest, &staging))
        return false;
    cmd->copy_resource(src, staging);
    subresource_data data = {};
    bool ok = dev->map_texture_region(staging, 0, nullptr, map_access::read_only, &data);
    if (ok) {
        pitch = data.row_pitch;
        out.assign(static_cast<const uint8_t*>(data.data), static_cast<const uint8_t*>(data.data) + size_t(pitch) * sd.texture.height);
        dev->unmap_texture_region(staging, 0);
    }
    dev->destroy_resource(staging);
    return ok;
}

// CPU version of the shader's reprojection (double precision). Returns motion in pixels.
static void PredictMotion(const Camera& cur, const Camera& prev, bool useTranslation, double px, double py, double depth,
                          double* mx, double* my)
{
    float inv[16];
    Invert4x4(cur.viewProj, inv);
    double W = cur.renderWidth, H = cur.renderHeight;
    double u = (px + 0.5) / W, v = (py + 0.5) / H;
    double c[4] = {u * 2 - 1, 1 - v * 2, depth, 1};
    double h[4] = {};
    for (int j = 0; j < 4; ++j)
        for (int i = 0; i < 4; ++i)
            h[j] += c[i] * inv[i * 4 + j];
    double p[4];
    if (std::fabs(h[3]) < 1e-12) {
        p[0] = h[0]; p[1] = h[1]; p[2] = h[2]; p[3] = 0;
    } else {
        for (int i = 0; i < 3; ++i)
            p[i] = h[i] / h[3] + (useTranslation ? double(prev.preViewTranslation[i]) - cur.preViewTranslation[i] : 0.0);
        p[3] = 1;
    }
    double q[4] = {};
    for (int j = 0; j < 4; ++j)
        for (int i = 0; i < 4; ++i)
            q[j] += p[i] * prev.viewProj[i * 4 + j];
    double pu = (q[0] / q[3]) * 0.5 + 0.5, pv = 0.5 - (q[1] / q[3]) * 0.5;
    *mx = (pu - u) * W;
    *my = (pv - v) * H;
}


static void ProbeMotion(command_list* cmd, device* dev, resource sceneColor, resource history, resource depthRes, const Camera& cur,
                        const Camera& prev, const Camera& prev2)
{
    std::vector<uint8_t> cb, hb, db;
    uint32_t cp = 0, hp = 0, dp = 0;
    if (!Readback(cmd, dev, sceneColor, cb, cp) || !Readback(cmd, dev, history, hb, hp) || !Readback(cmd, dev, depthRes, db, dp)) {
        LOGW("probe: readback failed");
        return;
    }
    const int W = int(cur.renderWidth), H = int(cur.renderHeight);
    std::vector<float> lc(size_t(W) * H), lh(size_t(W) * H), dz(size_t(W) * H);
    for (int y = 0; y < H; ++y)
        for (int x = 0; x < W; ++x) {
            const uint16_t* a = reinterpret_cast<const uint16_t*>(cb.data() + size_t(y) * cp) + x * 4;
            const uint16_t* b = reinterpret_cast<const uint16_t*>(hb.data() + size_t(y) * hp) + x * 4;
            auto lum = [](const uint16_t* t) {
                float l = 0.299f * HalfToFloat(t[0]) + 0.587f * HalfToFloat(t[1]) + 0.114f * HalfToFloat(t[2]);
                return std::sqrt(std::max(l, 0.0f));  // perceptual-ish, tames bright lights
            };
            lc[size_t(y) * W + x] = lum(a);
            lh[size_t(y) * W + x] = lum(b);
            uint32_t d = *reinterpret_cast<const uint32_t*>(db.data() + size_t(y) * dp + size_t(x) * 4);
            dz[size_t(y) * W + x] = float(d & 0xffffff) / 16777215.0f;
        }

    const int B = 12, R = 40, step = 160;
    struct Hyp { const char* name; const Camera* c; const Camera* p; bool trans; double err = 0; int n = 0; };
    Hyp hyps[] = {{"cur->prev (model)", &cur, &prev, true}, {"cur->prev no translation", &cur, &prev, false},
                  {"prev->prev2 (camera 1 frame late)", &prev, &prev2, true}, {"cur->prev2 (history 2 frames)", &cur, &prev2, true},
                  {"zero motion", &cur, &cur, true}};
    int logged = 0;
    for (int ty = R + B; ty < H - R - B; ty += step)
        for (int tx = R + B; tx < W - R - B; tx += step) {
            double mean = 0, var = 0;
            for (int y = -B; y < B; ++y)
                for (int x = -B; x < B; ++x)
                    mean += lc[size_t(ty + y) * W + tx + x];
            mean /= 4.0 * B * B;
            for (int y = -B; y < B; ++y)
                for (int x = -B; x < B; ++x) {
                    double d = lc[size_t(ty + y) * W + tx + x] - mean;
                    var += d * d;
                }
            if (var / (4.0 * B * B) < 1e-4)
                continue;  // flat tile, no reliable match
            double best = 1e30;
            int bx = 0, by = 0;
            for (int sy = -R; sy <= R; ++sy)
                for (int sx = -R; sx <= R; ++sx) {
                    double sad = 0;
                    for (int y = -B; y < B && sad < best; y += 2)
                        for (int x = -B; x < B; x += 2)
                            sad += std::fabs(lc[size_t(ty + y) * W + tx + x] - lh[size_t(ty + y + sy) * W + tx + x + sx]);
                    if (sad < best) {
                        best = sad;
                        bx = sx;
                        by = sy;
                    }
                }
            double depth = dz[size_t(ty) * W + tx];
            for (Hyp& hy : hyps) {
                double mx, my;
                PredictMotion(*hy.c, *hy.p, hy.trans, tx, ty, depth, &mx, &my);
                hy.err += std::hypot(mx - bx, my - by);
                ++hy.n;
                if (&hy == &hyps[0] && hy.n % 8 == 1 && logged < 12) {
                    ++logged;
                    LOGI("probe tile (%d,%d) depth %.6f: measured (%d,%d) predicted (%.1f,%.1f)", tx, ty, depth, bx, by, mx, my);
                }
            }
        }
    for (const Hyp& hy : hyps)
        LOGI("probe: %-36s mean error %.2f px over %d tiles", hy.name, hy.n ? hy.err / hy.n : -1.0, hy.n);
    LOGI("probe: PVT cur %.1f %.1f %.1f prev %.1f %.1f %.1f prev2 %.1f %.1f %.1f", cur.preViewTranslation[0],
         cur.preViewTranslation[1], cur.preViewTranslation[2], prev.preViewTranslation[0], prev.preViewTranslation[1],
         prev.preViewTranslation[2], prev2.preViewTranslation[0], prev2.preViewTranslation[1], prev2.preViewTranslation[2]);
}

#include "upscaler.inl"

static void LogUpscalerStats()
{
    LOGI("upscaler: %s, %u evaluations, %u failures, render %ux%u -> %ux%u", g_upscalerEnabled ? "on" : "off",
         g_up.evaluations, g_up.failures, g_up.renderWidth, g_up.renderHeight, g_up.outWidth, g_up.outHeight);
    g_up.evaluations = 0;
}

static bool RunMotionPass(command_list* cmd, BindState& s, resource_view sceneColorSrv)
{
    Camera cur, prev, prev2;
    {
        std::lock_guard<std::mutex> lock(g_camMutex);
        cur = g_camCur;
        prev = g_camPrev;
        prev2 = g_camPrev2;
    }
    if (!cur.valid || !cur.depth.handle || !cur.renderWidth || !sceneColorSrv.handle)
        return false;
    if (!prev.valid)
        prev = cur;
    device* dev = cmd->get_device();
    resource sceneColor = dev->get_resource_from_view(sceneColorSrv);
    resource depth = g_depthCopy.copiedThisFrame ? g_depthCopy.res : cur.depth;
    if (!EnsureMotionPass(dev, depth, sceneColor))
        return false;

    const bool haveVelocity = g_sv.srv.handle && g_sv.clearedFrame == g_frameId;
    MotionParams p = {};
    if (!Invert4x4(cur.viewProj, p.invViewProj))
        return false;
    memcpy(p.prevViewProj, prev.viewProj, sizeof(p.prevViewProj));
    for (int i = 0; i < 3; ++i)
        p.translationDelta[i] = prev.preViewTranslation[i] - cur.preViewTranslation[i];
    p.renderSize[0] = static_cast<float>(cur.renderWidth);
    p.renderSize[1] = static_cast<float>(cur.renderHeight);
    p.pixelsForFullColor = 32.0f;
    p.jitterNdc[0] = cur.jitterNdc[0];
    p.jitterNdc[1] = cur.jitterNdc[1];
    p.useObjectMotion = haveVelocity ? 1.0f : 0.0f;
    p.debugMode = static_cast<float>(g_mvDebug);

    cmd->bind_pipeline(pipeline_stage::compute_shader, g_mv.pipe);
    const resource_view srvs[4] = {g_mv.depthSrv, sceneColorSrv, g_mv.historySrv, haveVelocity ? g_sv.srv : resource_view{0}};
    const resource_view uavs[2] = {g_mv.motionUav, g_mv.debugUav};
    cmd->push_descriptors(shader_stage::compute, g_mv.layout, 0, descriptor_table_update{{}, 0, 0, 4, descriptor_type::shader_resource_view, srvs});
    cmd->push_descriptors(shader_stage::compute, g_mv.layout, 1, descriptor_table_update{{}, 0, 0, 2, descriptor_type::unordered_access_view, uavs});
    cmd->push_constants(shader_stage::compute, g_mv.layout, 2, 0, sizeof(MotionParams) / 4, &p);
    cmd->dispatch((cur.renderWidth + 7) / 8, (cur.renderHeight + 7) / 8, 1);

    // Unbind so the game can use these resources as render targets again without hazards.
    const resource_view nullViews[4] = {{0}, {0}, {0}, {0}};
    cmd->push_descriptors(shader_stage::compute, g_mv.layout, 0, descriptor_table_update{{}, 0, 0, 4, descriptor_type::shader_resource_view, nullViews});
    cmd->push_descriptors(shader_stage::compute, g_mv.layout, 1, descriptor_table_update{{}, 0, 0, 2, descriptor_type::unordered_access_view, nullViews});

    if (g_probeRequested) {
        g_probeRequested = false;
        ProbeMotion(cmd, dev, sceneColor, g_mv.history, depth, cur, prev, prev2.valid ? prev2 : prev);
    }

    // Keep this frame's untouched scene color for next frame's reprojection check.
    subresource_box box = {0, 0, 0, cur.renderWidth, cur.renderHeight, 1};
    if (g_mvDebug || g_debugKeys)  // only the debug views and the probe read it
        cmd->copy_texture_region(sceneColor, 0, &box, g_mv.history, 0, &box);
    if (g_mvDebug)
        cmd->copy_texture_region(g_mv.debug, 0, &box, sceneColor, 0, &box);
    ++g_mvRuns;

    if (!RunUpscalerAndFinalPass(cmd, s, sceneColor, depth, cur))
        return false;
    g_upscalerActive = true;
    return true;
}

// Returns true when the game's draw (its final pass) was replaced by ours.
static bool MaybeRunMotionPass(command_list* cmd)
{
    if (g_mvDoneThisFrame)
        return false;
    BindState& s = GetBind(cmd);
    if (!s.psInfo.isFinal)
        return false;
    g_mvDoneThisFrame = true;
    return RunMotionPass(cmd, s, s.psSrv0);
}

// ---------------------------------------------------------------- config (StyxUpscaler.ini next to the exe)

static int g_toggleKey = VK_HOME;    // Ctrl+Shift+<key> toggles the upscaler
static int g_qualityKey = VK_PRIOR;  // Ctrl+Shift+<key> cycles the quality presets

// Key names accepted in the ini: Home, End, Insert, Delete, PageUp, PageDown, F1..F12, A..Z, 0..9.
static int ParseKey(const char* name, int fallback)
{
    static const struct { const char* name; int vk; } named[] = {
        {"Home", VK_HOME}, {"End", VK_END}, {"Insert", VK_INSERT}, {"Delete", VK_DELETE},
        {"PageUp", VK_PRIOR}, {"PageDown", VK_NEXT}};
    for (const auto& n : named)
        if (_stricmp(name, n.name) == 0)
            return n.vk;
    if ((name[0] == 'F' || name[0] == 'f') && name[1] >= '1' && name[1] <= '9') {
        int f = atoi(name + 1);
        if (f >= 1 && f <= 12)
            return VK_F1 + f - 1;
    }
    if (name[0] && !name[1] && isalnum(static_cast<unsigned char>(name[0])))
        return toupper(static_cast<unsigned char>(name[0]));
    return fallback;
}

static void LoadConfig()
{
    const std::string ini = GameDirFile("StyxUpscaler.ini");
    char key[32];
    g_upscalerEnabled = GetPrivateProfileIntA("Upscaler", "Enabled", 1, ini.c_str()) != 0;
    GetPrivateProfileStringA("Upscaler", "ToggleKey", "Home", key, sizeof(key), ini.c_str());
    g_toggleKey = ParseKey(key, VK_HOME);
    GetPrivateProfileStringA("Upscaler", "QualityKey", "PageUp", key, sizeof(key), ini.c_str());
    g_qualityKey = ParseKey(key, VK_PRIOR);
    g_antiLag2Wanted = GetPrivateProfileIntA("Latency", "AntiLag2", 1, ini.c_str()) != 0;
    g_maxFrameLatency = GetPrivateProfileIntA("Latency", "MaxFrameLatency", 1, ini.c_str());
    g_debugKeys = GetPrivateProfileIntA("Debug", "DeveloperMode", 0, ini.c_str()) != 0;
    LOGI("config: upscaler %s, toggle Ctrl+Shift+%s, developer mode %s", g_upscalerEnabled ? "on" : "off", key,
         g_debugKeys ? "on" : "off");
}

// Mod hotkeys are Ctrl+Shift+<key> so they never collide with the game's own keys.
static bool ModHotkey(int vk)
{
    return (GetAsyncKeyState(VK_CONTROL) & 0x8000) && (GetAsyncKeyState(VK_SHIFT) & 0x8000) &&
           (GetAsyncKeyState(vk) & 0x8000);
}

// Edge-triggered hotkey: true once per press.
static bool HotkeyPressed(int vk, bool& wasDown)
{
    const bool down = ModHotkey(vk);
    const bool pressed = down && !wasDown;
    wasDown = down;
    return pressed;
}

// ---------------------------------------------------------------- quality presets
// The render resolution is the engine's GSystemSettings.ScreenPercentage, read every frame. It lives at a
// fixed place in StyxGame.exe's .data section (Steam build 1.0.10499); writing it switches the resolution
// live, and the upscaler feature follows. The choice is also saved to StyxSystemSettings.ini.

struct QualityPreset {
    const char* name;
    float percent;
};
static const QualityPreset kPresets[] = {
    {"Native AA", 100.0f}, {"Quality", 67.0f}, {"Balanced", 59.0f}, {"Performance", 50.0f}};
static float* g_screenPercentage = nullptr;

static float* FindScreenPercentage()
{
    static bool searched = false;
    if (searched)
        return g_screenPercentage;
    searched = true;
    const uintptr_t base = reinterpret_cast<uintptr_t>(GetModuleHandleA(nullptr));
    auto* p = reinterpret_cast<float*>(base + 0x3553390);
    // Sanity check against an unknown game build: a plausible percentage followed by bUpscaleScreenPercentage.
    const uint32_t next = *reinterpret_cast<const uint32_t*>(p + 1);
    if (*p >= 10.0f && *p <= 200.0f && next <= 1) {
        LOGI("quality: ScreenPercentage found (%.0f%%)", *p);
        return p;
    }
    LOGW("quality: ScreenPercentage not found (unknown game build), presets disabled");
    return nullptr;
}

// %USERPROFILE%\Documents (or its redirected location) via the registry.
static std::string GameConfigFile(const char* name)
{
    char raw[MAX_PATH] = {}, expanded[MAX_PATH] = {};
    DWORD size = sizeof(raw);
    if (RegGetValueA(HKEY_CURRENT_USER, "Software\\Microsoft\\Windows\\CurrentVersion\\Explorer\\User Shell Folders",
                     "Personal", RRF_RT_REG_SZ | RRF_RT_REG_EXPAND_SZ | RRF_NOEXPAND, nullptr, raw, &size) != ERROR_SUCCESS)
        return "";
    ExpandEnvironmentStringsA(raw, expanded, MAX_PATH);
    return std::string(expanded) + "\\My Games\\Styx\\StyxGame\\Config\\" + name;
}

static void SetQualityPreset(int index)
{
    if (!g_screenPercentage)
        return;
    const QualityPreset& q = kPresets[index];
    *g_screenPercentage = q.percent;
    const std::string ini = GameConfigFile("StyxSystemSettings.ini");
    char value[32];
    snprintf(value, sizeof(value), "%f", q.percent);
    if (!ini.empty())
        WritePrivateProfileStringA("SystemSettings", "ScreenPercentage", value, ini.c_str());
    LOGI("quality: %s (%.0f%% render resolution)", q.name, q.percent);
}

static int CurrentQualityPreset()
{
    if (!g_screenPercentage)
        return -1;
    for (int i = 0; i < static_cast<int>(std::size(kPresets)); ++i)
        if (fabsf(*g_screenPercentage - kPresets[i].percent) < 0.5f)
            return i;
    return -1;  // custom value from the ini
}

static void MotionFrameEnd()
{
    static bool upWasDown = false;
    if (HotkeyPressed(g_toggleKey, upWasDown)) {
        g_upscalerEnabled = !g_upscalerEnabled;
        g_up.resetNext = true;
        LOGI("upscaler %s", g_upscalerEnabled ? "ON" : "OFF");
    }
    static bool qualityWasDown = false;
    if (HotkeyPressed(g_qualityKey, qualityWasDown)) {
        g_screenPercentage = FindScreenPercentage();
        SetQualityPreset((CurrentQualityPreset() + 1) % static_cast<int>(std::size(kPresets)));
    }
    if (g_debugKeys) {  // developer mode only: Ctrl+Shift+M/J/O/P/B
        static bool debugWasDown = false, jitterWasDown = false, objectWasDown = false, probeWasDown = false,
                    bypassWasDown = false;
        if (HotkeyPressed('M', debugWasDown)) {
            g_mvDebug = (g_mvDebug + 1) % 5;
            static const char* names[5] = {"off", "vector colors", "reprojection (full model)", "reprojection bands",
                                           "character velocity coverage"};
            LOGI("motion vector debug view: %s (passes run so far: %u)", names[g_mvDebug], g_mvRuns);
        }
        if (HotkeyPressed('J', jitterWasDown)) {
            g_jitterEnabled = !g_jitterEnabled;
            LOGI("jitter %s", g_jitterEnabled ? "ON" : "OFF");
        }
        if (HotkeyPressed('O', objectWasDown)) {
            g_skinVelocityEnabled = !g_skinVelocityEnabled;
            LOGI("character motion %s", g_skinVelocityEnabled ? "ON" : "OFF");
        }
        if (HotkeyPressed('P', probeWasDown)) {
            g_probeRequested = true;
            LOGI("probe requested");
        }
        if (HotkeyPressed('B', bypassWasDown)) {
            g_bypass = !g_bypass;
            g_up.resetNext = true;
            LOGI("bypass %s", g_bypass ? "ON (add-on idle)" : "OFF");
        }
    }
    g_mvDoneThisFrame = false;
    g_depthCopy.copiedThisFrame = false;
}

// ---------------------------------------------------------------- trace-only hooks

static bool OnDraw(command_list* cmd, uint32_t n, uint32_t, uint32_t, uint32_t)
{
    if (g_bypass)
        return false;
    const int64_t t0 = g_debugKeys ? Now() : 0;
    ApplyJitter(cmd);
    SnapshotCamera(cmd);
    const bool replaced = MaybeRunMotionPass(cmd);
    Trace("DRAW n=%u", n);
    if (g_debugKeys)
        g_addonTicks += Now() - t0;
    return replaced;
}

static bool OnDrawIndexed(command_list* cmd, uint32_t n, uint32_t instances, uint32_t firstIndex, int32_t vertexOffset,
                          uint32_t firstInstance)
{
    if (g_bypass)
        return false;
    const int64_t t0 = g_debugKeys ? Now() : 0;
    ApplyJitter(cmd);
    DrawSkinVelocity(cmd, n, instances, firstIndex, vertexOffset, firstInstance);
    DrawRigidVelocity(cmd, n, instances, firstIndex, vertexOffset, firstInstance);
    SnapshotCamera(cmd);
    const bool replaced = MaybeRunMotionPass(cmd);
    Trace("DRAWIDX n=%u", n);
    if (g_debugKeys)
        g_addonTicks += Now() - t0;
    return replaced;
}

static bool OnClearRtv(command_list* cmd, resource_view rtv, const float*, uint32_t, const rect*)
{
    if (g_tracing) {
        char a[64];
        ViewDesc(cmd->get_device(), rtv, a, sizeof(a));
        Trace("CLEAR %s", a);
    }
    return false;
}

// ---------------------------------------------------------------- benchmark (Ctrl+Shift+F)
// Stand still: cycles full mod / upscaler off / bypass every 4 s and logs the average frame time of each.

static int g_benchPhase = -1;  // -1 = off
static void BenchFrameEnd()
{
    static bool wasDown = false;
    static int64_t lastPresent = 0, phaseStart = 0;
    static double sum[3] = {}, frames[3] = {};
    static int rounds = 0;
    static const char* names[3] = {"full mod (FSR on)", "upscaler off", "bypass (mod idle)"};
    const int64_t now = Now();
    bool down = g_debugKeys && ModHotkey(VK_END);
    if (down && !wasDown) {
        g_benchPhase = g_benchPhase < 0 ? 0 : -1;
        phaseStart = now;
        rounds = 0;
        for (int i = 0; i < 3; ++i)
            sum[i] = frames[i] = 0;
        LOGI("benchmark %s", g_benchPhase < 0 ? "stopped" : "started: stand still");
        if (g_benchPhase < 0) {
            g_upscalerEnabled = true;
            g_bypass = false;
        }
    }
    wasDown = down;
    if (g_benchPhase >= 0) {
        // Skip the first 0.5 s of each phase (feature re-creation, pipeline settle).
        if (lastPresent && TicksToMs(now - phaseStart) > 500.0) {
            sum[g_benchPhase] += TicksToMs(now - lastPresent);
            frames[g_benchPhase] += 1;
        }
        if (TicksToMs(now - phaseStart) > 4000.0) {
            g_benchPhase = (g_benchPhase + 1) % 3;
            phaseStart = now;
            if (g_benchPhase == 0 && ++rounds % 2 == 0)
                for (int i = 0; i < 3; ++i)
                    if (frames[i])
                        LOGI("benchmark: %-20s %.2f ms (%.0f fps)", names[i], sum[i] / frames[i], 1000.0 * frames[i] / sum[i]);
        }
        const bool wantUpscaler = g_benchPhase == 0, wantBypass = g_benchPhase == 2;
        if (g_upscalerEnabled != wantUpscaler) {
            g_upscalerEnabled = wantUpscaler;
            g_up.resetNext = true;
        }
        g_bypass = wantBypass;
    }
    lastPresent = now;
}

static void OnPresent(command_queue*, swapchain*, const rect*, const rect*, uint32_t, const rect*)
{
    ProfileFrameEnd();
    MotionFrameEnd();
    CameraFrameEnd();
    TraceFrameEnd();
    BenchFrameEnd();
    LatencyFrameEnd();
}

static void OnInitSwapchain(swapchain* sc, bool resize)
{
    resource_desc d = sc->get_device()->get_resource_desc(sc->get_current_back_buffer());
    LOGI("swapchain %s: back buffer %ux%u", resize ? "resized" : "created", d.texture.width, d.texture.height);
    // Anti-Lag 2 / frame latency follow the device that presents (the game and OptiScaler also create
    // short-lived helper devices).
    if (sc->get_device()->get_api() == device_api::d3d11)
        g_latencyDevice = reinterpret_cast<ID3D11Device*>(sc->get_device()->get_native());
}

// ---------------------------------------------------------------- settings overlay
// Shown in the ReShade overlay (Home) under Add-ons > Styx Upscaler. Changes apply live and are saved.

static void SaveSetting(const char* section, const char* key, int value)
{
    char text[16];
    snprintf(text, sizeof(text), "%d", value);
    WritePrivateProfileStringA(section, key, text, GameDirFile("StyxUpscaler.ini").c_str());
}

static const char* KeyName(int vk)
{
    static char buf[8];
    switch (vk) {
    case VK_HOME: return "Home";
    case VK_END: return "End";
    case VK_INSERT: return "Insert";
    case VK_DELETE: return "Delete";
    case VK_PRIOR: return "PageUp";
    case VK_NEXT: return "PageDown";
    }
    if (vk >= VK_F1 && vk <= VK_F12)
        snprintf(buf, sizeof(buf), "F%d", vk - VK_F1 + 1);
    else
        snprintf(buf, sizeof(buf), "%c", vk);
    return buf;
}

static void DrawSettings(effect_runtime*)
{
    if (ImGui::Checkbox("Upscaler (FSR / DLSS / XeSS via OptiScaler)", &g_upscalerEnabled)) {
        g_up.resetNext = true;
        SaveSetting("Upscaler", "Enabled", g_upscalerEnabled);
    }
    if (g_upscalerEnabled && g_up.renderWidth)
        ImGui::TextDisabled("Rendering %ux%u, output %ux%u", g_up.renderWidth, g_up.renderHeight, g_up.outWidth,
                            g_up.outHeight);
    else if (g_upscalerEnabled)
        ImGui::TextDisabled("Waiting for the first game frame...");

    g_screenPercentage = FindScreenPercentage();
    if (g_screenPercentage) {
        const int current = CurrentQualityPreset();
        char preview[48];
        if (current >= 0)
            snprintf(preview, sizeof(preview), "%s (%.0f%%)", kPresets[current].name, kPresets[current].percent);
        else
            snprintf(preview, sizeof(preview), "Custom (%.0f%%)", *g_screenPercentage);
        if (ImGui::BeginCombo("Quality", preview)) {
            for (int i = 0; i < static_cast<int>(std::size(kPresets)); ++i) {
                char label[48];
                snprintf(label, sizeof(label), "%s (%.0f%%)", kPresets[i].name, kPresets[i].percent);
                if (ImGui::Selectable(label, i == current))
                    SetQualityPreset(i);
            }
            ImGui::EndCombo();
        }
        ImGui::TextDisabled("Native AA = best image. Lower presets help when the GPU is the bottleneck.");
    } else {
        ImGui::TextDisabled("Quality presets unavailable (unknown game version).");
    }

    ImGui::Separator();
    if (ImGui::Checkbox("AMD Anti-Lag 2", &g_antiLag2Wanted))
        SaveSetting("Latency", "AntiLag2", g_antiLag2Wanted);
    if (g_latencyMode == kLatencyAntiLag2)
        ImGui::TextDisabled(g_antiLag2Wanted ? "Active (Radeon driver)" : "Off");
    else if (g_latencyMode == kLatencyFallback)
        ImGui::TextDisabled("Not available on this GPU/driver: frame queue limited to %d instead", g_maxFrameLatency);

    ImGui::Separator();
    ImGui::TextDisabled("Hotkeys: Ctrl+Shift+%s upscaler on/off", KeyName(g_toggleKey));
    ImGui::TextDisabled("         Ctrl+Shift+%s next quality preset", KeyName(g_qualityKey));
}

// ---------------------------------------------------------------- entry

BOOL APIENTRY DllMain(HMODULE module, DWORD reason, LPVOID)
{
    switch (reason) {
    case DLL_PROCESS_ATTACH:
        if (!reshade::register_addon(module))
            return FALSE;
        g_traceEnabled = GetFileAttributesA(GameDirFile("StyxUpscaler_trace.flag").c_str()) != INVALID_FILE_ATTRIBUTES;
        LoadConfig();
        reshade::register_overlay(nullptr, &DrawSettings);
        reshade::register_event<reshade::addon_event::bind_render_targets_and_depth_stencil>(OnBindRenderTargets);
        reshade::register_event<reshade::addon_event::clear_depth_stencil_view>(OnClearDsv);
        reshade::register_event<reshade::addon_event::destroy_resource_view>(OnDestroyResourceView);
        reshade::register_event<reshade::addon_event::destroy_resource>(OnDestroyResource);
        reshade::register_event<reshade::addon_event::init_device>(OnInitDevice);
        reshade::register_event<reshade::addon_event::destroy_device>(OnDestroyDevice);
        reshade::register_event<reshade::addon_event::init_swapchain>(OnInitSwapchain);
        reshade::register_event<reshade::addon_event::present>(OnPresent);
        reshade::register_event<reshade::addon_event::init_pipeline>(OnInitPipeline);
        reshade::register_event<reshade::addon_event::map_buffer_region>(OnMapBuffer);
        reshade::register_event<reshade::addon_event::unmap_buffer_region>(OnUnmapBuffer);
        reshade::register_event<reshade::addon_event::update_buffer_region>(OnUpdateBuffer);
        reshade::register_event<reshade::addon_event::bind_pipeline>(OnBindPipeline);
        reshade::register_event<reshade::addon_event::push_descriptors>(OnPushDescriptors);
        reshade::register_event<reshade::addon_event::draw>(OnDraw);
        reshade::register_event<reshade::addon_event::draw_indexed>(OnDrawIndexed);
        if (g_traceEnabled)
            reshade::register_event<reshade::addon_event::clear_render_target_view>(OnClearRtv);
        LOGI("StyxUpscaler stage 1 loaded (trace %s)", g_traceEnabled ? "on" : "off");
        break;
    case DLL_PROCESS_DETACH:
        UninstallInputLayoutCache();
        ReleaseInputLayoutCache();
        reshade::unregister_addon(module);
        break;
    }
    return TRUE;
}
