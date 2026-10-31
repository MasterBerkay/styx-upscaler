// Replacement for UE3's GammaCorrectionPixelShader when the upscaler is active: same math, but it reads
// the upscaled full-resolution scene color instead of the low-resolution render region.
//   OutColor = pow(saturate(lerp(SceneColor * ColorScale, OverlayColor.rgb, OverlayColor.a)), InverseGamma)

cbuffer FinalPassParams : register(b13)
{
    float4 ColorScale;    // xyz
    float4 OverlayColor;
    float4 GammaAndOrigin;  // x = InverseGamma, zw = viewport origin in pixels
};

Texture2D<float4> UpscaledColor : register(t0);

void VSMain(uint id : SV_VertexID, out float4 pos : SV_Position)
{
    float2 uv = float2((id << 1) & 2, id & 2);
    pos = float4(uv * float2(2.0, -2.0) + float2(-1.0, 1.0), 0.0, 1.0);
}

float4 PSMain(float4 pos : SV_Position) : SV_Target0
{
    float4 scene = UpscaledColor.Load(int3(int2(pos.xy - GammaAndOrigin.zw), 0));
    float3 linearColor = lerp(scene.rgb * ColorScale.rgb, OverlayColor.rgb, OverlayColor.a);
    return float4(pow(saturate(linearColor), GammaAndOrigin.x), scene.a);
}
