// Replacement for UE3's GammaCorrectionPixelShader when the upscaler is active: same math, but it reads
// the upscaled full-resolution scene color instead of the low-resolution render region.
//   OutColor = pow(saturate(lerp(SceneColor * ColorScale, OverlayColor.rgb, OverlayColor.a)), InverseGamma)
// Optional RCAS sharpening (the robust contrast-adaptive sharpening of AMD FidelityFX FSR 1) on the
// gamma-corrected values, i.e. on what ends up on screen.

cbuffer FinalPassParams : register(b13)
{
    float4 ColorScale;    // xyz
    float4 OverlayColor;
    float4 GammaAndOrigin;  // x = InverseGamma, zw = viewport origin in pixels
    float4 SharpenParams;   // x = RCAS sharpness 0..2 (0 = off, 1 = full RCAS, above that stronger)
};

Texture2D<float4> UpscaledColor : register(t0);

void VSMain(uint id : SV_VertexID, out float4 pos : SV_Position)
{
    float2 uv = float2((id << 1) & 2, id & 2);
    pos = float4(uv * float2(2.0, -2.0) + float2(-1.0, 1.0), 0.0, 1.0);
}

// Scene color at pixel p, as the game's gamma pass would output it.
float3 Display(int2 p, int2 hi)
{
    float3 scene = UpscaledColor.Load(int3(clamp(p, int2(0, 0), hi), 0)).rgb;
    float3 linearColor = lerp(scene * ColorScale.rgb, OverlayColor.rgb, OverlayColor.a);
    return pow(saturate(linearColor), GammaAndOrigin.x);
}

// RCAS: sharpen with a negative lobe on the 4 neighbours, limited per pixel so the result never leaves the
// local min/max range (no halos, no clipping). Same math as FsrRcasF in ffx_fsr1.h, without the noise filter.
float3 Rcas(float3 b, float3 d, float3 e, float3 f, float3 h, float sharpness)
{
    const float kLimit = 0.25 - 1.0 / 16.0;
    float3 mn4 = min(min(b, d), min(f, h));
    float3 mx4 = max(max(b, d), max(f, h));
    float3 hitMin = mn4 / (4.0 * mx4 + 1e-5);
    float3 hitMax = (1.0 - mx4) / (4.0 * mn4 - 4.0 - 1e-5);
    float3 lobeRgb = max(-hitMin, hitMax);
    float lobe = max(-kLimit, min(max(lobeRgb.r, max(lobeRgb.g, lobeRgb.b)), 0.0)) * min(sharpness, 1.0);
    float3 color = (lobe * (b + d + f + h) + e) / (4.0 * lobe + 1.0);
    // Above 1 (RCAS is at its own limit): push the RCAS result further away from the neighbour average and allow
    // a controlled overshoot of up to half the local contrast at 2 - inside the plain local range nothing would
    // change at peaks and dips, which is where sharpening shows.
    if (sharpness > 1.0)
    {
        const float extra = sharpness - 1.0;
        const float3 lo = min(mn4, e), hi = max(mx4, e);
        const float3 overshoot = (hi - lo) * 0.5 * extra;
        color = clamp(color + (color - (b + d + f + h) * 0.25) * extra * 2.0, lo - overshoot, hi + overshoot);
        color = saturate(color);
    }
    return color;
}

float4 PSMain(float4 pos : SV_Position) : SV_Target0
{
    uint width, height;
    UpscaledColor.GetDimensions(width, height);
    const int2 hi = int2(width - 1, height - 1);
    const int2 p = int2(pos.xy - GammaAndOrigin.zw);
    float3 color = Display(p, hi);
    if (SharpenParams.x > 0.0)
        color = Rcas(Display(p + int2(0, -1), hi), Display(p + int2(-1, 0), hi), color, Display(p + int2(1, 0), hi),
                     Display(p + int2(0, 1), hi), SharpenParams.x);
    float alpha = UpscaledColor.Load(int3(clamp(p, int2(0, 0), hi), 0)).a;
    return float4(color, alpha);
}
