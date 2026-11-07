using System;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading;
using LibreHardwareMonitor.Hardware;

public class UpdateVisitor : IVisitor {
    public void VisitComputer(IComputer computer) { computer.Traverse(this); }
    public void VisitHardware(IHardware hardware) {
        hardware.Update();
        foreach (IHardware sub in hardware.SubHardware) sub.Traverse(this);
    }
    public void VisitSensor(ISensor sensor) { }
    public void VisitParameter(IParameter parameter) { }
}

public static class LhmDump {
    static string[] wantNameKeys = new string[] {
        "CPU Package", "CPU Cores", "CPU Graphics", "CPU DRAM", "CPU Soc",
        "Power Limit", "Core #1 Clock", "Core (Max)", "Bus Clock",
        "CPU Core", "CPU Total"
    };
    static bool Want(ISensor s) {
        if (s.Value == null || !s.Value.HasValue) return false;
        if (s.SensorType == SensorType.Power) return true;
        if (s.SensorType == SensorType.Clock || s.SensorType == SensorType.Temperature)
            return wantNameKeys.Any(k => s.Name.IndexOf(k, StringComparison.OrdinalIgnoreCase) >= 0);
        return false;
    }

    public static int Main(string[] args) {
        if (args.Length < 1) { Console.Error.WriteLine("usage: lhm_dump <out.csv> [secs=900] [intervalMs=1000] [list]"); return 2; }
        string outCsv = args[0];
        int secs = args.Length > 1 ? int.Parse(args[1]) : 900;
        int intervalMs = args.Length > 2 ? int.Parse(args[2]) : 1000;
        bool listOnly = args.Length > 3 && args[3] == "list";

        var computer = new Computer {
            IsCpuEnabled = true, IsMotherboardEnabled = true,
            IsGpuEnabled = false, IsMemoryEnabled = false,
            IsStorageEnabled = false, IsNetworkEnabled = false,
            IsControllerEnabled = true, IsPsuEnabled = false, IsBatteryEnabled = false
        };
        computer.Open();
        var visitor = new UpdateVisitor();
        computer.Accept(visitor);

        var cpu = computer.Hardware.FirstOrDefault(h => h.HardwareType == HardwareType.Cpu);
        if (cpu == null) { Console.Error.WriteLine("no CPU hardware"); computer.Close(); return 3; }
        var sensors = cpu.Sensors.Where(Want).OrderBy(s => s.SensorType).ThenBy(s => s.Name).ToArray();

        if (listOnly) {
            Console.WriteLine("== all CPU sensors ==");
            foreach (var s in cpu.Sensors.OrderBy(s => s.SensorType).ThenBy(s => s.Name))
                Console.WriteLine("{0}|{1}|now={2}", s.SensorType, s.Name, s.Value);
            foreach (IHardware mb in computer.Hardware) {
                Console.WriteLine("-- mb: {0} ({1})", mb.Name, mb.HardwareType);
                foreach (var s in mb.Sensors) Console.WriteLine("   {0}|{1}|now={2}", s.SensorType, s.Name, s.Value);
            }
            computer.Close();
            return 0;
        }

        var sb = new StringBuilder();
        sb.Append("time");
        foreach (var s in sensors) sb.Append(',').Append(s.SensorType).Append(':').Append(s.Name.Replace(',', ' '));
        string header = sb.ToString();
        using (var fs = new StreamWriter(outCsv, false, Encoding.UTF8)) {
            fs.WriteLine(header); fs.Flush();
            var sw = Stopwatch.StartNew();
            while (sw.Elapsed.TotalSeconds < secs) {
                computer.Accept(visitor);
                sb.Clear().Append(DateTime.Now.ToString("HH:mm:ss.fff"));
                foreach (var s in sensors) sb.Append(',').Append(s.Value.HasValue ? s.Value.Value.ToString("0.##", System.Globalization.CultureInfo.InvariantCulture) : "");
                fs.WriteLine(sb.ToString());
                fs.Flush();
                Thread.Sleep(intervalMs);
            }
        }
        computer.Close();
        return 0;
    }
}
