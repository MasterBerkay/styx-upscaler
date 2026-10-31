// Per-pixel motion of GPU-skinned meshes (characters), drawn by the add-on right before the game's
// own base-pass draw of the same mesh, with the game's input assembler state still bound.
//
// Mirrors UE3's GpuSkinVertexFactory: position unpacked with MeshExtension/MeshOrigin, skinned by
// the weighted sum of four 3x4 bone matrices (rows of float4, translation in .w), then transformed
// by the translated LocalToWorld (row-vector convention) and the view projection.

cbuffer VSBoneConstants : register(b3)  // the game's current bones, left bound by the game
{
    float4 BoneMatrices[75 * 3];
};

cbuffer SkinVelocityParams : register(b5)
{
    row_major float4x4 CurLocalToWorld;   // translated world, this frame
    row_major float4x4 PrevLocalToWorld;  // translated world, previous frame
    row_major float4x4 CurViewProj;       // unjittered
    row_major float4x4 PrevViewProj;      // unjittered, previous frame
    row_major float4x4 JitteredViewProj;  // what the game rasterizes with this frame
    float4 MeshOrigin;
    float4 MeshExtension;
};

cbuffer PrevBoneConstants : register(b6)
{
    float4 PrevBoneMatrices[75 * 3];
};

// Must declare the same inputs in the same order as the game's skinned vertex shaders: the game's
// input layout feeds registers v0..v4, so leaving out TANGENT/NORMAL would shift the bone data.
struct VSInput
{
    float4 Position : POSITION0;      // v0
    float3 TangentX : TANGENT0;       // v1 (unused)
    float4 TangentZ : NORMAL0;        // v2 (unused)
    uint4 BlendIndices : BLENDINDICES0;  // v3
    float4 BlendWeights : BLENDWEIGHT0;  // v4
};

struct VSOutput
{
    float4 Position : SV_Position;
    float4 CurClip : TEXCOORD0;
    float4 PrevClip : TEXCOORD1;
};

float3 SkinCurrent(float3 p, uint4 idx, float4 w)
{
    float4 r0 = 0, r1 = 0, r2 = 0;
    [unroll] for (int k = 0; k < 4; ++k)
    {
        r0 += w[k] * BoneMatrices[idx[k] * 3 + 0];
        r1 += w[k] * BoneMatrices[idx[k] * 3 + 1];
        r2 += w[k] * BoneMatrices[idx[k] * 3 + 2];
    }
    float4 p1 = float4(p, 1.0);
    return float3(dot(r0, p1), dot(r1, p1), dot(r2, p1));
}

float3 SkinPrevious(float3 p, uint4 idx, float4 w)
{
    float4 r0 = 0, r1 = 0, r2 = 0;
    [unroll] for (int k = 0; k < 4; ++k)
    {
        r0 += w[k] * PrevBoneMatrices[idx[k] * 3 + 0];
        r1 += w[k] * PrevBoneMatrices[idx[k] * 3 + 1];
        r2 += w[k] * PrevBoneMatrices[idx[k] * 3 + 2];
    }
    float4 p1 = float4(p, 1.0);
    return float3(dot(r0, p1), dot(r1, p1), dot(r2, p1));
}

VSOutput VSMain(VSInput input)
{
    float3 local = input.Position.xyz * MeshExtension.xyz + MeshOrigin.xyz;
    float3 curWorld = mul(float4(SkinCurrent(local, input.BlendIndices, input.BlendWeights), 1.0), CurLocalToWorld).xyz;
    float3 prevWorld = mul(float4(SkinPrevious(local, input.BlendIndices, input.BlendWeights), 1.0), PrevLocalToWorld).xyz;

    VSOutput o;
    o.Position = mul(float4(curWorld, 1.0), JitteredViewProj);
    o.Position.w += 0.0 * (input.TangentX.x + input.TangentZ.x);  // keep v1/v2 in the signature
    o.CurClip = mul(float4(curWorld, 1.0), CurViewProj);
    o.PrevClip = mul(float4(prevWorld, 1.0), PrevViewProj);
    return o;
}

// rg = motion in UV (current -> previous), b = NDC depth of this surface (to test visibility later),
// a = 1 marks "object motion present".
float4 PSMain(VSOutput input) : SV_Target0
{
    float2 cur = input.CurClip.xy / input.CurClip.w;
    float2 prev = input.PrevClip.xy / input.PrevClip.w;
    float2 curUv = float2(cur.x * 0.5 + 0.5, 0.5 - cur.y * 0.5);
    float2 prevUv = float2(prev.x * 0.5 + 0.5, 0.5 - prev.y * 0.5);
    return float4(prevUv - curUv, input.Position.z, 1.0);
}

// Non-skinned meshes that moved since the last frame (weapons, attachments, doors, lifts):
// UE3's LocalVertexFactory transforms POSITION (v0) by LocalToWorld only.
struct RigidVSInput
{
    float4 Position : POSITION0;  // v0
};

VSOutput RigidVSMain(RigidVSInput input)
{
    float4 local = float4(input.Position.xyz, 1.0);
    float3 curWorld = mul(local, CurLocalToWorld).xyz;
    float3 prevWorld = mul(local, PrevLocalToWorld).xyz;

    VSOutput o;
    o.Position = mul(float4(curWorld, 1.0), JitteredViewProj);
    o.CurClip = mul(float4(curWorld, 1.0), CurViewProj);
    o.PrevClip = mul(float4(prevWorld, 1.0), PrevViewProj);
    return o;
}
