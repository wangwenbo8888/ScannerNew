@echo off
rem nsys_profile.bat - GPU timeline capture template (2026-10-29 working recipe)
rem Usage: nsys_profile.bat <output-name-without-ext> <exe> [exe-args...]
rem PITFALL: this nsys build (2024.5.1) treats the "--" separator as an EMPTY
rem option -> "option is ambiguous". Correct form: options then exe+args with
rem NO separator. Graph tracing flag kept for graph-era binaries (harmless).
rem Steady-state analysis: skip first ~2000 trace rows in post-processing
rem (startup/calib-load dilution); overlap analysis via nsys_overlap.ps1.
setlocal
set NSYS=C:\Program Files\NVIDIA Corporation\Nsight Systems 2024.5.1\target-windows-x64\nsys.exe
set PATH=C:\opencv-cuda-4.13.0\x64\vc17\bin;%PATH%
set OUT=%~1
shift
set EXE=%~1
shift
"%NSYS%" profile -t cuda --cuda-graph-trace=node -o "C:\Users\DUUMM\AppData\Local\Temp\opencode\nsys\%OUT%" %EXE% %1 %2 %3 %4 %5 %6 %7 %8 %9
endlocal
