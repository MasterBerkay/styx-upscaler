# Styx Upscaler – FSR 4 / DLSS / XeSS for Styx: Master of Shadows

Real temporal upscaling and anti-aliasing for *Styx: Master of Shadows* (2014, Unreal Engine 3, 64-bit)
in its DirectX 11 mode: **FSR 4** (RX 9000), **FSR 3.1**, **XeSS** or **DLSS** through
[OptiScaler](https://github.com/optiscaler/OptiScaler).

The game has no motion vectors and no jitter, so this ReShade add-on builds them:

- Halton(2,3) sub-pixel jitter injected into the main-view `ViewProjectionMatrix`
- Per-pixel motion vectors for the camera, GPU-skinned characters and moving rigid meshes
- Upscaler call through the NVIDIA NGX API, which OptiScaler maps to FSR / XeSS / DLSS
- The upscaled image is fed through an equivalent of the game's gamma pass, HUD stays native
- Fix for the invisible menus of the DX11 mode (Scaleform binds RTV/DSV pairs of different size)
- Input-layout cache against camera-turn stutter, low-latency mode (`SetMaximumFrameLatency`)

Player downloads (bundled with ReShade and OptiScaler) are on Nexus Mods. Companion mod:
[styx-engine-fix](https://github.com/MasterBerkay/styx-engine-fix) – high FPS without flying ragdolls.

## How it works

The add-on uses ReShade 6.x API events on the immediate context:

1. **Constant-buffer shadowing** – UE3 fills its constant buffers with `UpdateSubresource`. The add-on keeps
   a copy of the bytes it needs: `VSOffsetConstants` (`ViewProjectionMatrix`, `PreViewTranslation`),
   `$Globals` (`LocalToWorld`, `MeshOrigin`, `MeshExtension`) and `VSBoneConstants` (75 bones × 3 rows).
   Offsets come from reflecting every vertex shader once (cached by the DXBC checksum).
2. **Jitter** – before main-view base-pass draws, `VSOffsetConstants` is re-uploaded with the jittered
   matrix (row-vector convention: `clip.xy += jitter * clip.w`).
3. **Object velocity** – before each skinned or moved rigid draw of the main view the mesh is drawn again
   with the game's input layout into an RGBA32F target (own D32 depth), using the previous frame's bones /
   transform. The vertex shader must declare `POSITION, TANGENT, NORMAL, BLENDINDICES, BLENDWEIGHT` in that
   order, because the layout is matched by register. Rigid meshes are matched to the nearest instance of
   the same geometry in the previous frame.
4. **Depth** – UE3 clears the scene depth mid-frame (before the foreground pass), so it is copied right
   before that clear. MSAA must be off.
5. **Motion vectors** – a compute shader reprojects depth with the current / previous unjittered matrices
   (near plane 10, infinite far) and takes the object velocity where its depth matches the scene.
6. **Upscale** – right before the final `SceneColorTexture` → back buffer pass: NGX evaluate (OptiScaler),
   then a replacement pass `pow(saturate(lerp(scene * ColorScale, Overlay.rgb, Overlay.a)), InverseGamma)`.

## Building

Visual Studio 2022 Build Tools (x64), then `build.bat` → `bin\StyxUpscaler.addon64`.

Put the dependencies in `third_party\` (not part of this repository):

| Path | From |
|---|---|
| `third_party\reshade\include` | `include` folder of [ReShade 6.8.0](https://github.com/crosire/reshade) (BSD-3) |
| `third_party\ngx\include` | `include` folder of the [NVIDIA DLSS SDK](https://github.com/NVIDIA/DLSS) |
| `third_party\ngx\lib\nvsdk_ngx_s.lib` | `lib\Windows_x86_64\x64\nvsdk_ngx_s.lib` of the DLSS SDK |

The NGX SDK is licensed by NVIDIA under the NVIDIA RTX SDKs License, which you accept by downloading it.

## Installing a build

Next to `Binaries\Win64\StyxGame.exe`: ReShade 6.x **with add-on support** as `dxgi.dll`, OptiScaler as
`winmm.dll` with `Dx11Upscaler=fsr31_12` (or `xess_12` / `dlss`) in `OptiScaler.ini`, the FidelityFX
DLLs from the OptiScaler release, `StyxUpscaler.addon64` and `dist\StyxUpscaler.ini`.
Launch the game with `-dx11` and set `MaxMultiSamples=1` in `StyxSystemSettings.ini`.

Settings (`StyxUpscaler.ini`): `[Upscaler] Enabled`, `ToggleKey` (with Ctrl+Shift, default Home),
`[Latency] MaxFrameLatency`, `[Debug] DeveloperMode` (debug views, bypass, benchmark, statistics).

## License

MIT (see `LICENSE`) for the code in this repository. Builds contain object code of the NVIDIA NGX SDK,
which stays under NVIDIA's license. Styx: Master of Shadows is © Cyanide / Focus Entertainment; no game
files are included. Not affiliated with Cyanide, Focus Entertainment, AMD, NVIDIA or Intel.
