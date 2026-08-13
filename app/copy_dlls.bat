@echo off
set QT5_BIN=C:\Qt\Qt5.15.2\5.15.2\msvc2019_64\bin
set OSG_BIN=F:\osg3.6.5\install\bin
set OPENCV_BIN=F:\opencv4.13\install\x64\vc17\bin
set DEST=%~1
set CFG=%~2

echo Copying DLLs to %DEST% (%CFG%)...

REM === Qt DLLs (always release names, except debug override below) ===
for %%D in (Qt5Core Qt5Gui Qt5Widgets Qt5OpenGL Qt5Svg Qt5SerialPort) do (
    copy /Y "%QT5_BIN%\%%D.dll" "%DEST%\" >nul 2>&1
    copy /Y "%QT5_BIN%\%%Dd.dll" "%DEST%\" >nul 2>&1
)

REM === OSG DLLs ===
for %%D in (osg161-osg osg161-osgDB osg161-osgGA osg161-osgUtil osg161-osgViewer osg161-osgText ot21-OpenThreads) do (
    copy /Y "%OSG_BIN%\%%D.dll" "%DEST%\" >nul 2>&1
    copy /Y "%OSG_BIN%\%%Dd.dll" "%DEST%\" >nul 2>&1
)

REM === OpenCV DLLs (core + calib3d deps + CUDA) ===
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
