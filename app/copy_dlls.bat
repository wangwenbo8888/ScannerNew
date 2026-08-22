@echo off
REM 路径对齐 环境配置汇总.md（2026-08-22 修正：迁移后本机路径）
set QT5_BIN=C:\Qt\Qt5.15.2\5.15.2\msvc2019_64\bin
set OSG_BIN=F:\osg3.6.5\install\bin
REM 本机 F:/opencv4.13 的 Debug(d 后缀)与 Release DLL 同目录
set OPENCV_BIN=F:\opencv4.13\install\x64\vc17\bin
set DEST=%~1
set CFG=%~2

echo Copying DLLs to %DEST% (%CFG%)...

REM === Qt DLLs ===
for %%D in (Qt5Core Qt5Gui Qt5Widgets Qt5OpenGL Qt5Svg Qt5SerialPort) do (
    copy /Y "%QT5_BIN%\%%D.dll" "%DEST%\" >nul 2>&1
    copy /Y "%QT5_BIN%\%%Dd.dll" "%DEST%\" >nul 2>&1
)

REM === OSG DLLs ===
for %%D in (osg161-osg osg161-osgDB osg161-osgGA osg161-osgUtil osg161-osgViewer osg161-osgText ot21-OpenThreads) do (
    copy /Y "%OSG_BIN%\%%D.dll" "%DEST%\" >nul 2>&1
    copy /Y "%OSG_BIN%\%%Dd.dll" "%DEST%\" >nul 2>&1
)

REM === OpenCV DLLs ===
for %%D in (opencv_core4130 opencv_calib3d4130 opencv_imgproc4130 opencv_imgcodecs4130 opencv_features2d4130 opencv_flann4130 opencv_cudaarithm4130 opencv_cudawarping4130 opencv_cudaimgproc4130 opencv_cudafilters4130 opencv_cudev4130) do (
    copy /Y "%OPENCV_BIN%\%%D.dll" "%DEST%\" >nul 2>&1
    copy /Y "%OPENCV_BIN%\%%Dd.dll" "%DEST%\" >nul 2>&1
)

REM === CUDA runtime ===
if exist "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6\bin\cudart64_12.dll" (
    copy /Y "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6\bin\cudart64_12.dll" "%DEST%\" >nul 2>&1
)

REM === 大恒 Galaxy 相机 SDK 运行时（GxIAPICPPEx 直链依赖；目录缺则跳过）===
set GXSDK_BIN=F:\factory_calib\build_fc_all\gui_qt\Release
if exist "%GXSDK_BIN%\GxIAPICPPEx.dll" (
    for %%D in (GxIAPI GxIAPICPP GxIAPICPPEx DxImageProc GCBase_MD_VC120_v3_0 GenApi_MD_VC120_v3_0 GenCP_MD_VC120_v3_0 Log_MD_VC120_v3_0 log4cpp_MD_VC120_v3_0 MathParser_MD_VC120_v3_0 NodeMapData_MD_VC120_v3_0 msvcp100 msvcr100 mfc90) do (
        copy /Y "%GXSDK_BIN%\%%D.dll" "%DEST%\" >nul 2>&1
    )
)

REM === Qt platform plugin ===
if not exist "%DEST%\platforms" mkdir "%DEST%\platforms"
if /I "%CFG%"=="Debug" (
    copy /Y "%QT5_BIN%\..\plugins\platforms\qwindowsd.dll" "%DEST%\platforms\qwindows.dll" >nul 2>&1
) else (
    copy /Y "%QT5_BIN%\..\plugins\platforms\qwindows.dll" "%DEST%\platforms\" >nul 2>&1
)

echo Done.
exit /b 0
