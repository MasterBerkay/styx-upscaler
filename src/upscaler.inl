// ---------------------------------------------------------------- upscaler (NGX / DLSS interface)
// Included from addon.cpp. At UE3's final pass we evaluate "DLSS" through NVIDIA's NGX API. On this
// AMD system OptiScaler (installed as winmm.dll) answers those calls and runs FSR 4 (or FSR 3 / XeSS /
// real DLSS, selectable in its overlay). The upscaled image then goes through our own copy of UE3's
// gamma pass; the game's HUD is drawn on top afterwards as usual.

#include <nvsdk_ngx.h>
#include <nvsdk_ngx_helpers.h>
#include <nvsdk_ngx_helpers_d3d.h>
#include "final_pass_vs.h"
#include "final_pass_ps.h"

struct Upscaler {
    bool initTried = false, available = false;
    NVSDK_NGX_Parameter* params = nullptr;
    NVSDK_NGX_Handle* feature = nullptr;
    uint32_t renderWidth = 0, renderHeight = 0, outWidth = 0, outHeight = 0;
    resource output = {0};
    resource_view outputSrv = {0};
    pipeline vs = {0}, ps = {0};
    resource constants = {0};
    bool resetNext = true;
    unsigned evaluations = 0, failures = 0;
};
static Upscaler g_up;
static bool g_upscalerEnabled = true;
// RCAS in the final pass (live, from the overlay): on/off and strength 0..2 (1 = full RCAS).
static bool g_sharpenEnabled = true;
static float g_sharpness = 1.0f;

static void ShutdownUpscaler(device* dev)
{
    if (g_up.feature)
        NVSDK_NGX_D3D11_ReleaseFeature(g_up.feature);
    if (g_up.params)
        NVSDK_NGX_D3D11_DestroyParameters(g_up.params);
    if (g_up.available)
        NVSDK_NGX_D3D11_Shutdown1(reinterpret_cast<ID3D11Device*>(dev->get_native()));
    if (g_up.outputSrv.handle) dev->destroy_resource_view(g_up.outputSrv);
    if (g_up.output.handle) dev->destroy_resource(g_up.output);
    if (g_up.constants.handle) dev->destroy_resource(g_up.constants);
    if (g_up.vs.handle) dev->destroy_pipeline(g_up.vs);
    if (g_up.ps.handle) dev->destroy_pipeline(g_up.ps);
    g_up = {};
}

static bool InitUpscaler(device* dev)
{
    if (g_up.initTried)
        return g_up.available;
    g_up.initTried = true;

    wchar_t dir[MAX_PATH];
    GetModuleFileNameW(nullptr, dir, MAX_PATH);
    if (wchar_t* slash = wcsrchr(dir, L'\\'))
        slash[0] = L'\0';
    auto* d3d = reinterpret_cast<ID3D11Device*>(dev->get_native());
    NVSDK_NGX_Result r = NVSDK_NGX_D3D11_Init(0x5354595855ull, dir, d3d);
    if (NVSDK_NGX_FAILED(r)) {
        LOGW("upscaler: NVSDK_NGX_D3D11_Init failed (0x%08x) - is OptiScaler installed as winmm.dll?", (unsigned)r);
        return false;
    }
    r = NVSDK_NGX_D3D11_GetCapabilityParameters(&g_up.params);
    if (NVSDK_NGX_FAILED(r) || !g_up.params) {
        LOGW("upscaler: GetCapabilityParameters failed (0x%08x)", (unsigned)r);
        return false;
    }
    int available = 0;
    NVSDK_NGX_Parameter_GetI(g_up.params, NVSDK_NGX_Parameter_SuperSampling_Available, &available);
    if (!available) {
        LOGW("upscaler: super sampling reported unavailable");
        return false;
    }

    shader_desc vsd = {};
    vsd.code = g_FinalPassVS;
    vsd.code_size = sizeof(g_FinalPassVS);
    shader_desc psd = {};
    psd.code = g_FinalPassPS;
    psd.code_size = sizeof(g_FinalPassPS);
    pipeline_subobject vso = {pipeline_subobject_type::vertex_shader, 1, &vsd};
    pipeline_subobject pso = {pipeline_subobject_type::pixel_shader, 1, &psd};
    if (!dev->create_pipeline({0}, 1, &vso, &g_up.vs) || !dev->create_pipeline({0}, 1, &pso, &g_up.ps) ||
        !dev->create_resource(resource_desc(64, memory_heap::default_, resource_usage::constant_buffer), nullptr,
                              resource_usage::constant_buffer, &g_up.constants)) {
        LOGW("upscaler: failed to create final pass objects");
        return false;
    }
    g_up.available = true;
    LOGI("upscaler: NGX initialised, super sampling available");
    return true;
}

static bool EnsureUpscalerFeature(command_list* cmd, uint32_t rw, uint32_t rh, uint32_t ow, uint32_t oh)
{
    if (g_up.feature && g_up.renderWidth == rw && g_up.renderHeight == rh && g_up.outWidth == ow && g_up.outHeight == oh)
        return true;
    device* dev = cmd->get_device();
    if (g_up.feature) {
        NVSDK_NGX_D3D11_ReleaseFeature(g_up.feature);
        g_up.feature = nullptr;
    }
    if (g_up.outputSrv.handle) dev->destroy_resource_view(g_up.outputSrv);
    if (g_up.output.handle) dev->destroy_resource(g_up.output);
    g_up.outputSrv = {0};
    g_up.output = {0};

    resource_desc od(ow, oh, 1, 1, format::r16g16b16a16_float, 1, memory_heap::default_,
                     resource_usage::unordered_access | resource_usage::shader_resource);
    if (!dev->create_resource(od, nullptr, resource_usage::unordered_access, &g_up.output) ||
        !dev->create_resource_view(g_up.output, resource_usage::shader_resource, resource_view_desc(format::r16g16b16a16_float), &g_up.outputSrv)) {
        LOGW("upscaler: failed to create %ux%u output", ow, oh);
        return false;
    }

    NVSDK_NGX_DLSS_Create_Params cp = {};
    cp.Feature.InWidth = rw;
    cp.Feature.InHeight = rh;
    cp.Feature.InTargetWidth = ow;
    cp.Feature.InTargetHeight = oh;
    cp.Feature.InPerfQualityValue = (rw >= ow) ? NVSDK_NGX_PerfQuality_Value_DLAA : NVSDK_NGX_PerfQuality_Value_MaxQuality;
    cp.InFeatureCreateFlags = NVSDK_NGX_DLSS_Feature_Flags_IsHDR | NVSDK_NGX_DLSS_Feature_Flags_MVLowRes |
                              NVSDK_NGX_DLSS_Feature_Flags_AutoExposure;
    auto* ctx = reinterpret_cast<ID3D11DeviceContext*>(cmd->get_native());
    NVSDK_NGX_Result r = NGX_D3D11_CREATE_DLSS_EXT(ctx, &g_up.feature, g_up.params, &cp);
    if (NVSDK_NGX_FAILED(r) || !g_up.feature) {
        LOGW("upscaler: create feature %ux%u -> %ux%u failed (0x%08x)", rw, rh, ow, oh, (unsigned)r);
        g_up.feature = nullptr;
        return false;
    }
    g_up.renderWidth = rw;
    g_up.renderHeight = rh;
    g_up.outWidth = ow;
    g_up.outHeight = oh;
    g_up.resetNext = true;
    // More jitter phases for larger upscale ratios (8 * ratio^2, as recommended for DLSS / FSR).
    float ratio = static_cast<float>(ow) / static_cast<float>(rw);
    g_jitterPhases = static_cast<uint32_t>(std::ceil(8.0f * ratio * ratio));
    LOGI("upscaler: feature created %ux%u -> %ux%u, %u jitter phases", rw, rh, ow, oh, g_jitterPhases);
    return true;
}

// Runs at the game's final pass draw. Returns true when the game's draw must be skipped (we drew it).
static bool RunUpscalerAndFinalPass(command_list* cmd, BindState& s, resource sceneColor, resource depth,
                                    const Camera& cur)
{
    if (!g_upscalerEnabled)
        return false;
    device* dev = cmd->get_device();
    if (!InitUpscaler(dev))
        return false;

    // The final pass reads ColorScale / OverlayColor / InverseGamma from its $Globals. Pixel shader state is
    // read from the context here (once per frame) instead of being tracked on every bind.
    const FinalPsInfo info = s.psInfo.fpi;
    if (!s.psInfo.haveFinalInfo || info.slot >= 14)
        return false;
    auto* ctx = reinterpret_cast<ID3D11DeviceContext*>(cmd->get_native());
    std::vector<uint8_t> globals;
    {
        ID3D11Buffer* cb = nullptr;
        ctx->PSGetConstantBuffers(info.slot, 1, &cb);
        if (cb)
            cb->Release();
        std::lock_guard<std::mutex> lock(g_camMutex);
        const std::vector<uint8_t>* sh = Shadow(dev, reinterpret_cast<uintptr_t>(cb));
        if (!sh)
            return false;
        globals = *sh;
    }
    if (globals.size() < info.inverseGamma + 4 || globals.size() < info.overlayColor + 16 || globals.size() < info.colorScale + 12)
        return false;

    D3D11_VIEWPORT vp = {};
    UINT numVp = 1;
    ctx->RSGetViewports(&numVp, &vp);
    const uint32_t rw = cur.renderWidth, rh = cur.renderHeight;
    const uint32_t ow = static_cast<uint32_t>(vp.Width + 0.5f), oh = static_cast<uint32_t>(vp.Height + 0.5f);
    if (!rw || !rh || !ow || !oh || !g_mv.motion.handle || !EnsureUpscalerFeature(cmd, rw, rh, ow, oh))
        return false;

    NVSDK_NGX_D3D11_DLSS_Eval_Params ep = {};
    ep.Feature.pInColor = reinterpret_cast<ID3D11Resource*>(sceneColor.handle);
    ep.Feature.pInOutput = reinterpret_cast<ID3D11Resource*>(g_up.output.handle);
    ep.pInDepth = reinterpret_cast<ID3D11Resource*>(depth.handle);
    ep.pInMotionVectors = reinterpret_cast<ID3D11Resource*>(g_mv.motion.handle);
    // Jitter in render pixels: the projection got clip.xy += (2*jx/W, -2*jy/H) * clip.w.
    ep.InJitterOffsetX = cur.jitterNdc[0] * 0.5f * static_cast<float>(rw);
    ep.InJitterOffsetY = -cur.jitterNdc[1] * 0.5f * static_cast<float>(rh);
    ep.InRenderSubrectDimensions.Width = rw;
    ep.InRenderSubrectDimensions.Height = rh;
    ep.InReset = g_up.resetNext ? 1 : 0;
    ep.InMVScaleX = static_cast<float>(rw);  // our vectors are in UV units of the render region
    ep.InMVScaleY = static_cast<float>(rh);
    const int64_t evalStart = Now();
    NVSDK_NGX_Result r = NGX_D3D11_EVALUATE_DLSS_EXT(ctx, g_up.feature, g_up.params, &ep);
    const int64_t evalTicks = Now() - evalStart;
    g_evalTicks += evalTicks;
    g_evalMaxTicks = std::max(g_evalMaxTicks, evalTicks);
    if (NVSDK_NGX_FAILED(r)) {
        if (g_up.failures++ < 5)
            LOGW("upscaler: evaluate failed (0x%08x)", (unsigned)r);
        return false;
    }
    g_up.resetNext = false;
    ++g_up.evaluations;

    // Our copy of UE3's gamma pass, reading the upscaled image.
    float c[16] = {};
    memcpy(c + 0, globals.data() + info.colorScale, 12);
    memcpy(c + 4, globals.data() + info.overlayColor, 16);
    memcpy(c + 8, globals.data() + info.inverseGamma, 4);
    c[10] = vp.TopLeftX;
    c[11] = vp.TopLeftY;
    c[12] = g_sharpenEnabled ? g_sharpness : 0.0f;
    g_inOwnUpload = true;
    dev->update_buffer_region(c, g_up.constants, 0, sizeof(c));
    g_inOwnUpload = false;

    // Game state our draw changes, put back afterwards.
    ID3D11Buffer* gameCb = nullptr;
    ID3D11InputLayout* gameLayout = nullptr;
    D3D11_PRIMITIVE_TOPOLOGY gameTopology = D3D11_PRIMITIVE_TOPOLOGY_UNDEFINED;
    ctx->PSGetConstantBuffers(13, 1, &gameCb);
    ctx->IAGetInputLayout(&gameLayout);
    ctx->IAGetPrimitiveTopology(&gameTopology);

    const buffer_range ourCb = {g_up.constants, 0, UINT64_MAX};
    cmd->bind_pipeline(pipeline_stage::vertex_shader, g_up.vs);
    cmd->bind_pipeline(pipeline_stage::pixel_shader, g_up.ps);
    ctx->IASetInputLayout(nullptr);
    ctx->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
    cmd->push_descriptors(shader_stage::pixel, {0}, 0, descriptor_table_update{{}, 0, 0, 1, descriptor_type::shader_resource_view, &g_up.outputSrv});
    cmd->push_descriptors(shader_stage::pixel, {0}, 0, descriptor_table_update{{}, 13, 0, 1, descriptor_type::constant_buffer, &ourCb});
    cmd->draw(3, 1, 0, 0);

    ctx->PSSetConstantBuffers(13, 1, &gameCb);
    cmd->push_descriptors(shader_stage::pixel, {0}, 0, descriptor_table_update{{}, 0, 0, 1, descriptor_type::shader_resource_view, &s.psSrv0});
    ctx->IASetInputLayout(gameLayout);
    ctx->IASetPrimitiveTopology(gameTopology);
    cmd->bind_pipeline(pipeline_stage::vertex_shader, pipeline{s.vs});
    cmd->bind_pipeline(pipeline_stage::pixel_shader, pipeline{s.ps});
    if (gameCb)
        gameCb->Release();
    if (gameLayout)
        gameLayout->Release();
    return true;
}
