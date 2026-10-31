// Camera motion vectors from depth + current/previous UE3 view-projection.
//
// UE3 renders in "translated world" space (world + PreViewTranslation) with row-vector math:
//   clip = mul(float4(translatedWorld, 1), ViewProjection)
// For each pixel of the render region we rebuild the translated world position from depth,
// move it into the previous frame's translated space and project it with last frame's matrix.
// Output: motion in UV units pointing from the current pixel to its previous position
// (the FSR / DLSS convention), plus optional debug views.

cbuffer Params : register(b0)
{
    row_major float4x4 InvViewProj;   // current clip -> translated world
    row_major float4x4 PrevViewProj;  // previous translated world -> clip
    float4 TranslationDelta;          // prevPreViewTranslation - curPreViewTranslation (xyz)
    float2 RenderSize;                // size of the rendered region (top-left of the textures)
    float PixelsForFullColor;         // motion length (in pixels) that saturates the vector view
    float DebugMode;                  // 0 off, 1 vector colors, 2 reprojection (model), 3 reprojection bands
    float2 JitterNdc;                 // jitter added to this frame's projection (NDC units)
    float UseObjectMotion;            // 1 when ObjectVelocity holds this frame's character motion
};

Texture2D<float> SceneDepth : register(t0);
Texture2D<float4> SceneColor : register(t1);
Texture2D<float4> History : register(t2);    // previous frame's scene color (same layout)
Texture2D<float4> ObjectVelocity : register(t3);  // rg = motion, b = depth of the drawn surface, a = coverage
RWTexture2D<float2> MotionOut : register(u0);
RWTexture2D<float4> DebugOut : register(u1);

float3 LoadHistoryBilinear(float2 pixel)
{
    // pixel is in texel units, centers at +0.5
    float2 p = pixel - 0.5;
    int2 i = int2(floor(p));
    float2 f = p - i;
    int2 maxI = int2(RenderSize) - 1;
    float3 a = History.Load(int3(clamp(i, 0, maxI), 0)).rgb;
    float3 b = History.Load(int3(clamp(i + int2(1, 0), 0, maxI), 0)).rgb;
    float3 c = History.Load(int3(clamp(i + int2(0, 1), 0, maxI), 0)).rgb;
    float3 d = History.Load(int3(clamp(i + int2(1, 1), 0, maxI), 0)).rgb;
    return lerp(lerp(a, b, f.x), lerp(c, d, f.x), f.y);
}

float2 Reproject(float4 h, float2 uv, float3 delta)
{
    float4 prevClip;
    if (abs(h.w) < 1e-7)
        prevClip = mul(float4(h.xyz, 0.0), PrevViewProj);  // point at infinity (sky): direction only
    else
        prevClip = mul(float4(h.xyz / h.w + delta, 1.0), PrevViewProj);
    float2 prevNdc = prevClip.xy / prevClip.w;
    float2 prevUv = float2(prevNdc.x * 0.5 + 0.5, 0.5 - prevNdc.y * 0.5);
    return prevUv - uv;
}

[numthreads(8, 8, 1)]
void main(uint3 id : SV_DispatchThreadID)
{
    if (id.x >= (uint)RenderSize.x || id.y >= (uint)RenderSize.y)
        return;

    float2 uv = (float2(id.xy) + 0.5) / RenderSize;
    // The pixel was rendered with a jittered projection; remove the jitter so the vectors describe
    // unjittered motion (what FSR / DLSS expect).
    float2 ndc = float2(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0) - JitterNdc;
    uv = float2(ndc.x * 0.5 + 0.5, 0.5 - ndc.y * 0.5);
    float depth = SceneDepth.Load(int3(id.xy, 0));

    float4 h = mul(float4(ndc, depth, 1.0), InvViewProj);
    float2 motion = Reproject(h, uv, TranslationDelta.xyz);
    float4 ov = UseObjectMotion > 0.5 ? ObjectVelocity.Load(int3(id.xy, 0)) : float4(0, 0, 0, 0);
    // Character motion where the character is what ended up visible (its depth matches).
    bool objectVisible = ov.a > 0.5 && abs((1.0 - ov.b) - (1.0 - depth)) <= 0.02 * (1.0 - depth) + 1e-6;
    if (objectVisible)
        motion = ov.rg;
    MotionOut[id.xy] = motion;

    if (DebugMode > 3.5)
    {
        // Object velocity coverage: green = used, red = drawn but rejected by the depth test, else dim scene.
        float3 scene = SceneColor.Load(int3(id.xy, 0)).rgb * 0.25;
        float3 c = objectVisible ? float3(0.0, 0.8, 0.0) : (ov.a > 0.5 ? float3(0.9, 0.0, 0.0) : scene);
        DebugOut[id.xy] = float4(c, 1.0);
        return;
    }

    if (DebugMode < 0.5)
        return;

    float3 scene = SceneColor.Load(int3(id.xy, 0)).rgb;
    if (DebugMode < 1.5)
    {
        // Direction as hue-like color, magnitude in pixels; grey = no motion.
        float2 px = motion * RenderSize / PixelsForFullColor;
        float3 viz = float3(saturate(0.5 + px.x), saturate(0.5 + px.y), 0.5);
        float luma = dot(scene, float3(0.299, 0.587, 0.114));
        DebugOut[id.xy] = float4(lerp(viz, luma.xxx, 0.3), 1.0);
    }
    else
    {
        // Reproject last frame and show the error, amplified. Three bands for comparison:
        // left = full model, middle = rotation only (no translation), right = translation sign flipped.
        float third = DebugMode > 2.5 ? RenderSize.x / 3.0 : RenderSize.x + 10.0;  // mode 2: full screen model
        if (abs(float(id.x) - third) < 2.0 || abs(float(id.x) - 2.0 * third) < 2.0)
        {
            DebugOut[id.xy] = float4(1.0, 1.0, 1.0, 1.0);
            return;
        }
        float2 used = float(id.x) < third ? motion
                    : (float(id.x) < 2.0 * third ? Reproject(h, uv, float3(0.0, 0.0, 0.0))
                                                  : Reproject(h, uv, -TranslationDelta.xyz));
        float2 prevPixel = (uv + used) * RenderSize;
        bool offscreen = any(prevPixel < 0.0) || any(prevPixel > RenderSize);
        float3 prev = LoadHistoryBilinear(prevPixel);
        float3 diff = abs(scene - prev) * 4.0;
        DebugOut[id.xy] = offscreen ? float4(0.0, 0.0, 0.6, 1.0) : float4(diff, 1.0);
    }
}
