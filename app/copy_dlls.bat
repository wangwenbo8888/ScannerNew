@echo off
REM 路径对齐 环境配置汇总.md（2026-08-13 修正：原 C:\Qt\/F:\ 为别机路径）
set QT5_BIN=C:\devlibs\Qt-5.15.2-msvc2019_64\5.15.2\msvc2019_64\bin
set OSG_BIN=C:\devlibs\osg-install\bin
if /I "%~2"=="Debug" (
    set OPENCV_BIN=C:\opencv-cuda-4.13.0-debug\x64\vc17\bin
) else (
    set OPENCV_BIN=C:\opencv-cuda-4.13.0\x64\vc17\bin
)
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
for %%D in (opencv_core4130 opencv_calib3d4130 opencv_imgproc4130 opencv_imgcodecs4130 opencv_features2d4130 opencv_flann4130 opencv_cudaarithm4130 opencv_cudawarping4130 opencv_cudaimgproc4130 opencv_cudev4130) do (
    copy /Y "%OPENCV_BIN%\%%D.dll" "%DEST%\" >nul 2>&1
    copy /Y "%OPENCV_BIN%\%%Dd.dll" "%DEST%\" >nul 2>&1
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
