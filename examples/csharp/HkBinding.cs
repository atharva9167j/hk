/**
 * HK C# / .NET P/Invoke Binding
 * Cross-platform interface for Unity, Xamarin, and .NET applications.
 */

using System;
using System.Runtime.InteropServices;

namespace Hk
{
    public enum HkStorageType : byte
    {
        F32 = 0x00,
        F16 = 0x01,
        BF16 = 0x02,
        FP8_E4M3 = 0x03,
        FP8_E5M2 = 0x04,
        INT8 = 0x05,
        INT32 = 0x06,
        INT64 = 0x07,
        UINT8 = 0x08,
        BOOL = 0x09,
        DQ4 = 0x10,
        DQ8 = 0x11,
        DQ6 = 0x12,
        DQ12 = 0x13,
        DQT = 0x14,
        SPARSE_F16 = 0x20,
        SPARSE_DQ8 = 0x21,
        SPARSE_2_4 = 0x22,
        SPARSE_DQ4_2_4 = 0x23,
        NULL_REF = 0x30,
        SHARED_REF = 0x31,
        LORA_REF = 0x32
    }

    public enum HkTileLayout : byte
    {
        RowMajor = 0x00,
        ColMajor = 0x01,
        Tile16x16 = 0x02,
        Tile16x8 = 0x03,
        Tile32x16 = 0x04,
        BlockSparse2_4 = 0x05,
        Tile32x32 = 0x06,
        Tile64x64 = 0x07
    }

    public enum HkSparsityType : byte
    {
        None = 0x00,
        Bitmask = 0x01,
        CSR = 0x02,
        Structured2_4 = 0x03,
        PhysicalPruned = 0x04,
        BSR = 0x05
    }

    public enum HkAppendixType : byte
    {
        LoRAAdapter = 0x01,
        DeltaPatch = 0x02,
        NewLayer = 0x03,
        CodeEval = 0x04,
        KVCacheSink = 0x05,
        TopologyHead = 0x06
    }

    [StructLayout(LayoutKind.Sequential)]
    public struct HkTensorInfo
    {
        public IntPtr Name;
        public byte StorageType;
        public byte TileLayout;
        public byte SparsityType;
        public byte NDim;
        [MarshalAs(UnmanagedType.ByValArray, SizeConst = 8)]
        public ulong[] Shape;
        public ulong DataOffset;
        public ulong DataSize;
        public ulong ResidualOffset;
        public ulong ResidualSize;
        public ulong ScaleOffset;
        public ulong ScaleSize;
        public ushort BlockSize;
        public float SparsityRatio;
    }

    [StructLayout(LayoutKind.Sequential)]
    public struct HkAppendixEntry
    {
        public byte EntryType;
        public byte Flags;
        public uint Generation;
        public ulong Timestamp;
        [MarshalAs(UnmanagedType.ByValArray, SizeConst = 32)]
        public byte[] ParentHash;
        public float MetricLoss;
        public float MetricAcc;
        public float MetricPass;
        public float MetricCustom;
        public IntPtr Name;
        public IntPtr Target;
        public IntPtr Data;
        public ulong DataSize;
    }

    public static class HkNative
    {
        private const string LibName = "hk";

        [DllImport(LibName, CallingConvention = CallingConvention.Cdecl, CharSet = CharSet.Ansi)]
        public static extern IntPtr hk_open(string path);

        [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
        public static extern void hk_close(IntPtr reader);

        [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
        public static extern ulong hk_get_tensor_count(IntPtr reader);

        [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
        public static extern int hk_get_tensor_info(IntPtr reader, ulong index, ref HkTensorInfo outInfo);

        [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
        public static extern IntPtr hk_get_tensor_data(IntPtr reader, ulong index, ref ulong outSize);

        [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
        public static extern IntPtr hk_get_tensor_residual(IntPtr reader, ulong index, ref ulong outSize);

        [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
        public static extern IntPtr hk_get_tensor_scales(IntPtr reader, ulong index, ref ulong outSize);

        [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
        public static extern int hk_dequantize_f32(IntPtr reader, ulong index, int withResidual, [Out] float[] outBuf, ulong count);

        [DllImport(LibName, CallingConvention = CallingConvention.Cdecl, CharSet = CharSet.Ansi)]
        public static extern IntPtr hk_get_metadata_string(IntPtr reader, string key);

        [DllImport(LibName, CallingConvention = CallingConvention.Cdecl, CharSet = CharSet.Ansi)]
        public static extern int hk_get_metadata_int(IntPtr reader, string key, ref long outVal);

        [DllImport(LibName, CallingConvention = CallingConvention.Cdecl, CharSet = CharSet.Ansi)]
        public static extern int hk_get_metadata_float(IntPtr reader, string key, ref double outVal);

        [DllImport(LibName, CallingConvention = CallingConvention.Cdecl, CharSet = CharSet.Ansi)]
        public static extern int hk_get_metadata_bool(IntPtr reader, string key, ref int outVal);

        [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
        public static extern ulong hk_appendix_get_count(IntPtr reader);

        [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
        public static extern int hk_appendix_get_entry(IntPtr reader, ulong index, ref HkAppendixEntry outEntry);

        [DllImport(LibName, CallingConvention = CallingConvention.Cdecl, CharSet = CharSet.Ansi)]
        public static extern int hk_appendix_rollback(string filePath, uint targetGeneration);

        // Container Writer
        [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
        public static extern IntPtr hk_writer_create(ulong alignment);

        [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
        public static extern void hk_writer_destroy(IntPtr writer);

        [DllImport(LibName, CallingConvention = CallingConvention.Cdecl, CharSet = CharSet.Ansi)]
        public static extern int hk_writer_add_metadata_string(IntPtr writer, string key, string val);

        [DllImport(LibName, CallingConvention = CallingConvention.Cdecl, CharSet = CharSet.Ansi)]
        public static extern int hk_writer_add_metadata_int(IntPtr writer, string key, long val);

        [DllImport(LibName, CallingConvention = CallingConvention.Cdecl, CharSet = CharSet.Ansi)]
        public static extern int hk_writer_add_metadata_float(IntPtr writer, string key, double val);

        [DllImport(LibName, CallingConvention = CallingConvention.Cdecl, CharSet = CharSet.Ansi)]
        public static extern int hk_writer_add_metadata_bool(IntPtr writer, string key, int val);

        [DllImport(LibName, CallingConvention = CallingConvention.Cdecl, CharSet = CharSet.Ansi)]
        public static extern int hk_writer_add_tensor(IntPtr writer, string name, byte storageType, byte tileLayout, byte sparsityType, byte ndim, [In] ulong[] shape, [In] byte[] data, ulong dataLen, float sparsityRatio);

        [DllImport(LibName, CallingConvention = CallingConvention.Cdecl, CharSet = CharSet.Ansi)]
        public static extern int hk_writer_write_to_file(IntPtr writer, string path);
    }
}

