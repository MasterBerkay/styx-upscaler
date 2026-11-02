// Replacement for the game's Bink movie draw under DX11. The game binds the decoded Y / Cr / Cb planes and its
// conversion constants, but issues the draw without an input layout, so D3D11 drops it and cutscenes stay black.
// This pass draws the same image as a full-screen triangle that needs no vertex input.

cbuffer Globals : register(b0)  // the game's $Globals of its Bink pixel shader
{
    float4 Unused[5];
    float4 tor;     // @80: Y, Cr, Cb, 1 -> red
    float4 tog;     // @96: -> green
    float4 tob;     // @112: -> blue
    float4 consts;  // @128
};

Texture2D<float> Luma : register(t0);
Texture2D<float> ChromaR : register(t1);
Texture2D<float> ChromaB : register(t2);
SamplerState Linear : register(s0);

void VSMain(uint id : SV_VertexID, out float4 pos : SV_Position, out float2 uv : TEXCOORD0)
{
    uv = float2((id << 1) & 2, id & 2);
    pos = float4(uv * float2(2.0, -2.0) + float2(-1.0, 1.0), 0.0, 1.0);
}

float4 PSMain(float4 pos : SV_Position, float2 uv : TEXCOORD0) : SV_Target0
{
    const float4 ycc = float4(Luma.Sample(Linear, uv), ChromaR.Sample(Linear, uv), ChromaB.Sample(Linear, uv), 1.0);
    return float4(dot(ycc, tor), dot(ycc, tog), dot(ycc, tob), 1.0);
}
