using System;
using System.Diagnostics;
using System.IO;
using System.IO.Compression;
using System.Reflection;
using System.Text;
using System.Windows.Forms;

internal static class HierarchosStandaloneLauncher
{
    private static readonly byte[] Magic = Encoding.ASCII.GetBytes("HRCHSFX1");
    private const int FooterSize = 16;

    [STAThread]
    private static int Main(string[] args)
    {
        try
        {
            string selfPath = Assembly.GetExecutingAssembly().Location;
            PayloadInfo payload = ReadPayloadInfo(selfPath);

            if (args.Length > 0 && args[0] == "--sfx-verify")
            {
                return payload.Length > 0 ? 0 : 2;
            }

            string runtimeKey = payload.Length.ToString("X16")
                + "_"
                + File.GetLastWriteTimeUtc(selfPath).Ticks.ToString("X16");

            string runtimeRoot = Path.Combine(
                Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData),
                "Hierarchos",
                "StandaloneRuntime",
                runtimeKey
            );
            string guiExe = Path.Combine(runtimeRoot, "Hierarchos.exe");

            if (!File.Exists(guiExe) || !File.Exists(Path.Combine(runtimeRoot, "hierarchos-backend.exe")))
            {
                ExtractPayload(selfPath, payload, runtimeRoot);
            }

            ProcessStartInfo startInfo = new ProcessStartInfo(guiExe)
            {
                WorkingDirectory = runtimeRoot,
                UseShellExecute = false,
                Arguments = QuoteArgs(args),
            };
            Process.Start(startInfo);
            return 0;
        }
        catch (Exception ex)
        {
            MessageBox.Show(
                ex.ToString(),
                "Hierarchos launch failed",
                MessageBoxButtons.OK,
                MessageBoxIcon.Error
            );
            return 1;
        }
    }

    private static PayloadInfo ReadPayloadInfo(string selfPath)
    {
        using (FileStream stream = File.OpenRead(selfPath))
        {
            if (stream.Length < FooterSize)
            {
                throw new InvalidDataException("Standalone payload footer is missing.");
            }

            stream.Seek(-FooterSize, SeekOrigin.End);
            byte[] footer = new byte[FooterSize];
            ReadExactly(stream, footer, 0, footer.Length);

            for (int i = 0; i < Magic.Length; i++)
            {
                if (footer[i] != Magic[i])
                {
                    throw new InvalidDataException("Standalone payload marker is missing.");
                }
            }

            long payloadLength = BitConverter.ToInt64(footer, Magic.Length);
            long payloadOffset = stream.Length - FooterSize - payloadLength;
            if (payloadLength <= 0 || payloadOffset <= 0)
            {
                throw new InvalidDataException("Standalone payload length is invalid.");
            }

            return new PayloadInfo(payloadOffset, payloadLength);
        }
    }

    private static void ExtractPayload(string selfPath, PayloadInfo payload, string runtimeRoot)
    {
        if (Directory.Exists(runtimeRoot))
        {
            Directory.Delete(runtimeRoot, true);
        }
        Directory.CreateDirectory(runtimeRoot);
        string payloadZip = Path.Combine(runtimeRoot, "hierarchos_payload.zip");

        using (FileStream input = File.OpenRead(selfPath))
        using (FileStream output = File.Create(payloadZip))
        {
            input.Seek(payload.Offset, SeekOrigin.Begin);
            CopyBytes(input, output, payload.Length);
        }

        ZipFile.ExtractToDirectory(payloadZip, runtimeRoot);
        File.Delete(payloadZip);
    }

    private static void CopyBytes(Stream input, Stream output, long bytesToCopy)
    {
        byte[] buffer = new byte[1024 * 1024];
        long remaining = bytesToCopy;
        while (remaining > 0)
        {
            int read = input.Read(buffer, 0, (int)Math.Min(buffer.Length, remaining));
            if (read <= 0)
            {
                throw new EndOfStreamException("Unexpected end of standalone payload.");
            }
            output.Write(buffer, 0, read);
            remaining -= read;
        }
    }

    private static void ReadExactly(Stream stream, byte[] buffer, int offset, int count)
    {
        int total = 0;
        while (total < count)
        {
            int read = stream.Read(buffer, offset + total, count - total);
            if (read <= 0)
            {
                throw new EndOfStreamException();
            }
            total += read;
        }
    }

    private static string QuoteArgs(string[] args)
    {
        if (args == null || args.Length == 0)
        {
            return string.Empty;
        }

        string[] quoted = new string[args.Length];
        for (int i = 0; i < args.Length; i++)
        {
            quoted[i] = "\"" + args[i].Replace("\\", "\\\\").Replace("\"", "\\\"") + "\"";
        }
        return string.Join(" ", quoted);
    }

    private struct PayloadInfo
    {
        public readonly long Offset;
        public readonly long Length;

        public PayloadInfo(long offset, long length)
        {
            Offset = offset;
            Length = length;
        }
    }
}
