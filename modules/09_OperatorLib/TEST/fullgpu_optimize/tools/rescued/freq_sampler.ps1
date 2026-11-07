param([string]$file, [int]$seconds)
Add-Type -TypeDefinition @'
using System;
using System.Runtime.InteropServices;
using System.IO;
public class FreqSampler {
  [DllImport("pdh.dll")] static extern int PdhOpenQuery(IntPtr rc, IntPtr userdata, out IntPtr q);
  [DllImport("pdh.dll")] static extern int PdhAddEnglishCounter(IntPtr q, string path, IntPtr userdata, out IntPtr c);
  [DllImport("pdh.dll")] static extern int PdhCollectQueryData(IntPtr q);
  [DllImport("pdh.dll")] static extern int PdhGetFormattedCounterValue(IntPtr c, int fmt, out IntPtr type, out PDH_FMT_COUNTERVALUE val);
  [StructLayout(LayoutKind.Sequential)] struct PDH_FMT_COUNTERVALUE { public uint CSize, Type; public double DoubleValue; public int LongValue; public IntPtr pad1, pad2; }
  const int PDH_FMT_DOUBLE = 0x200;
  public static void Run(string file, int seconds) {
    string[] paths = new string[] {
      "\\Processor(4)\\% Processor Time",                      // 0: 1b 占用%（核2-LP0）
      "\\Processor(1)\\% Processor Time",                      // 1: ch2 占用%（核0-LP1）
      "\\Processor Information(0,_Total)\\% Processor Performance",  // 2: 整体主频%
      "\\Processor Information(0,4)\\% Processor Performance",       // 3: 核2(1b) 主频%
      "\\Processor Information(0,1)\\% Processor Performance"        // 4: 核0(ch2) 主频%
    };
    IntPtr q; PdhOpenQuery(IntPtr.Zero, IntPtr.Zero, out q);
    var counters = new IntPtr[paths.Length];
    for (int i = 0; i < paths.Length; i++) PdhAddEnglishCounter(q, paths[i], IntPtr.Zero, out counters[i]);
    var sw = new StreamWriter(file, false);
    var t0 = DateTime.UtcNow;
    PdhCollectQueryData(q);
    while ((DateTime.UtcNow - t0).TotalSeconds < seconds) {
      System.Threading.Thread.Sleep(100);
      if (PdhCollectQueryData(q) != 0) continue;
      var parts = new string[paths.Length];
      for (int i = 0; i < paths.Length; i++) {
        PDH_FMT_COUNTERVALUE v; IntPtr t;
        PdhGetFormattedCounterValue(counters[i], PDH_FMT_DOUBLE, out t, out v);
        parts[i] = v.DoubleValue.ToString("F1");
      }
      sw.WriteLine(string.Join(",", parts)); sw.Flush();
    }
    sw.Close();
  }
}
'@
[FreqSampler]::Run($file, $seconds)
