@echo off
rem Builds bin\StyxUpscaler.addon64 (ReShade add-on, x64) with the VS 2022 Build Tools.
setlocal
set "VCVARS=%ProgramFiles(x86)%\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
call "%VCVARS%" >nul || exit /b 1
cd /d "%~dp0"
if not exist bin mkdir bin
if not exist obj mkdir obj
fxc /nologo /T cs_5_0 /E main /O3 /Vn g_MotionVectorsCS /Fh obj\motion_vectors_cs.h src\shaders\motion_vectors.hlsl >nul || exit /b 1
fxc /nologo /T vs_5_0 /E VSMain /O3 /Vn g_SkinVelocityVS /Fh obj\skin_velocity_vs.h src\shaders\skin_velocity.hlsl >nul || exit /b 1
fxc /nologo /T ps_5_0 /E PSMain /O3 /Vn g_SkinVelocityPS /Fh obj\skin_velocity_ps.h src\shaders\skin_velocity.hlsl >nul || exit /b 1
fxc /nologo /T vs_5_0 /E RigidVSMain /O3 /Vn g_RigidVelocityVS /Fh obj\rigid_velocity_vs.h src\shaders\skin_velocity.hlsl >nul || exit /b 1
fxc /nologo /T vs_5_0 /E VSMain /O3 /Vn g_FinalPassVS /Fh obj\final_pass_vs.h src\shaders\final_pass.hlsl >nul || exit /b 1
fxc /nologo /T ps_5_0 /E PSMain /O3 /Vn g_FinalPassPS /Fh obj\final_pass_ps.h src\shaders\final_pass.hlsl >nul || exit /b 1
cl /nologo /O2 /MT /EHsc /std:c++17 /W3 /DUNICODE /D_UNICODE /DNOMINMAX /D_CRT_SECURE_NO_WARNINGS /Ithird_party\reshade\include /Ithird_party\ngx\include /Isrc\imgui /Iobj ^
   /Foobj\ src\addon.cpp /LD /Febin\StyxUpscaler.addon64 /link /IMPLIB:obj\addon.lib dxguid.lib user32.lib advapi32.lib third_party\ngx\lib\nvsdk_ngx_s.lib || exit /b 1
echo Build OK: %~dp0bin\StyxUpscaler.addon64
