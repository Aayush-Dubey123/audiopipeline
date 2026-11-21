$ErrorActionPreference = "Stop"

$Root = $PSScriptRoot
if (-not $Root) { $Root = (Get-Location).Path }

$PythonExe = Join-Path $Root "backend\.venv\Scripts\python.exe"
if (-not (Test-Path $PythonExe)) { $PythonExe = "python" }

if (-not ("AudioPipeline.Launcher.NativeProcessJob" -as [type])) {
    Add-Type @"
using System;
using System.ComponentModel;
using System.Runtime.InteropServices;

namespace AudioPipeline.Launcher {
    public sealed class NativeProcessJob : IDisposable {
        [DllImport("kernel32.dll", CharSet = CharSet.Unicode)]
        private static extern IntPtr CreateJobObject(IntPtr attributes, string name);

        [DllImport("kernel32.dll", SetLastError = true)]
        private static extern bool SetInformationJobObject(
            IntPtr job, int informationClass, IntPtr information, uint length);

        [DllImport("kernel32.dll", SetLastError = true)]
        private static extern bool AssignProcessToJobObject(IntPtr job, IntPtr process);

        [DllImport("kernel32.dll", SetLastError = true)]
        private static extern bool CloseHandle(IntPtr handle);

        private const int JobObjectExtendedLimitInformation = 9;
        private const uint JobObjectLimitKillOnJobClose = 0x00002000;
        private IntPtr handle;

        [StructLayout(LayoutKind.Sequential)]
        private struct BasicLimitInformation {
            public long PerProcessUserTimeLimit;
            public long PerJobUserTimeLimit;
            public uint LimitFlags;
            public UIntPtr MinimumWorkingSetSize;
            public UIntPtr MaximumWorkingSetSize;
            public uint ActiveProcessLimit;
            public UIntPtr Affinity;
            public uint PriorityClass;
            public uint SchedulingClass;
        }

        [StructLayout(LayoutKind.Sequential)]
        private struct IoCounters {
            public ulong ReadOperationCount;
            public ulong WriteOperationCount;
            public ulong OtherOperationCount;
            public ulong ReadTransferCount;
            public ulong WriteTransferCount;
            public ulong OtherTransferCount;
        }

        [StructLayout(LayoutKind.Sequential)]
        private struct ExtendedLimitInformation {
            public BasicLimitInformation BasicLimitInformation;
            public IoCounters IoInfo;
            public UIntPtr ProcessMemoryLimit;
            public UIntPtr JobMemoryLimit;
            public UIntPtr PeakProcessMemoryUsed;
            public UIntPtr PeakJobMemoryUsed;
        }

        private NativeProcessJob(IntPtr handle) {
            this.handle = handle;
        }

        public static NativeProcessJob Create() {
            IntPtr handle = CreateJobObject(IntPtr.Zero, null);
            if (handle == IntPtr.Zero) throw new Win32Exception(Marshal.GetLastWin32Error());

            var limits = new ExtendedLimitInformation();
            limits.BasicLimitInformation.LimitFlags = JobObjectLimitKillOnJobClose;
            int size = Marshal.SizeOf(limits);
            IntPtr pointer = Marshal.AllocHGlobal(size);
            try {
                Marshal.StructureToPtr(limits, pointer, false);
                if (!SetInformationJobObject(handle, JobObjectExtendedLimitInformation, pointer, (uint)size)) {
                    throw new Win32Exception(Marshal.GetLastWin32Error());
                }
            } finally {
                Marshal.FreeHGlobal(pointer);
            }
            return new NativeProcessJob(handle);
        }

        public void Assign(System.Diagnostics.Process process) {
            if (!AssignProcessToJobObject(handle, process.Handle)) {
                throw new Win32Exception(Marshal.GetLastWin32Error());
            }
        }

        public void Dispose() {
            if (handle != IntPtr.Zero) {
                CloseHandle(handle);
                handle = IntPtr.Zero;
            }
        }
    }
}
"@
}

$Job = [AudioPipeline.Launcher.NativeProcessJob]::Create()
try {
    $BackendInfo = [System.Diagnostics.ProcessStartInfo]::new()
    $BackendInfo.FileName = $PythonExe
    $BackendInfo.Arguments = "-m uvicorn core.apis.api:app --app-dir backend --host 127.0.0.1 --port 8000"
    $BackendInfo.WorkingDirectory = $Root
    $BackendInfo.UseShellExecute = $false
    $BackendInfo.CreateNoWindow = $true

    $FrontendInfo = [System.Diagnostics.ProcessStartInfo]::new()
    $FrontendInfo.FileName = $env:ComSpec
    $FrontendInfo.Arguments = '/d /s /c "npm.cmd run dev"'
    $FrontendInfo.WorkingDirectory = Join-Path $Root "frontend"
    $FrontendInfo.UseShellExecute = $false
    $FrontendInfo.CreateNoWindow = $true

    $Backend = [System.Diagnostics.Process]::Start($BackendInfo)
    $Job.Assign($Backend)
    $Frontend = [System.Diagnostics.Process]::Start($FrontendInfo)
    $Job.Assign($Frontend)

    $Deadline = [DateTime]::UtcNow.AddSeconds(30)
    do {
        Start-Sleep -Milliseconds 250
        try {
            $Connection = [Net.Sockets.TcpClient]::new("127.0.0.1", 3000)
            $Connection.Dispose()
            $FrontendReady = $true
        } catch {
            $FrontendReady = $false
        }
    } until ($FrontendReady -or [DateTime]::UtcNow -ge $Deadline)

    if (-not $FrontendReady) {
        throw "The frontend did not start on http://localhost:3000 within 30 seconds."
    }

    Start-Process "http://localhost:3000"
    Wait-Process -Id $Backend.Id, $Frontend.Id
} finally {
    $Job.Dispose()
}

