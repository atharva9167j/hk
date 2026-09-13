/**
 * HK Java / Android Native JNI Bridge
 * Exposes the high-performance HK binary loader to Java and Android applications.
 */

package com.hk;

public class HkBridge {
    static {
        System.loadLibrary("hk");
    }

    public static class TensorInfo {
        public String name;
        public int storageType;
        public int tileLayout;
        public int sparsityType;
        public int ndim;
        public long[] shape;
        public long dataOffset;
        public long dataSize;
        public long residualOffset;
        public long residualSize;
        public long scaleOffset;
        public long scaleSize;
        public int blockSize;
        public float sparsityRatio;
    }

    public static class AppendixEntry {
        public int entryType;
        public int flags;
        public int generation;
        public long timestamp;
        public byte[] parentHash;
        public float metricLoss;
        public float metricAccuracy;
        public float metricPassRate;
        public float metricCustom;
        public String name;
        public String target;
        public byte[] data;
    }

    // Native JNI functions
    public static native long open(String path);
    public static native void close(long readerPtr);
    public static native long getTensorCount(long readerPtr);
    public static native boolean getTensorInfo(long readerPtr, long index, TensorInfo outInfo);
    public static native int dequantizeF32(long readerPtr, long index, boolean withResidual, float[] outBuf);
    public static native String getMetadataString(long readerPtr, String key);
    public static native long getMetadataInt(long readerPtr, String key);
    public static native double getMetadataFloat(long readerPtr, String key);
    public static native boolean getMetadataBool(long readerPtr, String key);
    public static native long getAppendixCount(long readerPtr);
    public static native boolean getAppendixEntry(long readerPtr, long index, AppendixEntry outEntry);
    public static native int rollback(String path, int targetGeneration);
}

