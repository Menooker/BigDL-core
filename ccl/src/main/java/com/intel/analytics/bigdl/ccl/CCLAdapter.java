package com.intel.analytics.bigdl.ccl;

import java.io.File;
import java.io.FileOutputStream;
import java.io.InputStream;
import java.net.URL;
import java.nio.channels.Channels;
import java.nio.channels.FileChannel;
import java.nio.channels.ReadableByteChannel;

public class CCLAdapter{

    private static File extract(String path) {
        try {
            URL url = CCLAdapter.class.getResource("/" + path);
            if (url == null) {
                throw new Error("Can't find dynamic lib file in jar, path = " + path);
            } else {
                InputStream in = CCLAdapter.class.getResourceAsStream("/" + path);
                File file = null;
                if (System.getProperty("os.name").toLowerCase().contains("win")) {
                    file = new File(System.getProperty("java.io.tmpdir") + File.separator + path);
                } else {
                    file = File.createTempFile("cclNativeLoader", path);
                }

                ReadableByteChannel src = Channels.newChannel(in);
                FileChannel dest = (new FileOutputStream(file)).getChannel();
                dest.transferFrom(src, 0L, 9223372036854775807L);
                dest.close();
                src.close();
                return file;
            }
        } catch (Throwable var6) {
            throw new Error("Can't extract dynamic lib file to /tmp dir.\n" + var6);
        }
    }

    public static void load() {
        System.loadLibrary("ccl");
        File tmpFile = extract("libccl_java.so");
        try {
            System.load(tmpFile.getAbsolutePath());
        } finally {
            tmpFile.delete();
        }
    }

    public static void doInit()
    {
        initCCL();
        Runtime.getRuntime().addShutdownHook(new Thread(new Runnable() {
            public void run() {
                finalizeCCL();
            }
        }));
    }

    public static native void sayHello();
    public static native void initCCL();
    public static native void finalizeCCL();
    public static native void setEnv(String apiServer, int nNodes);
    private static native long createCommunicator(int color);
    private static native void releaseCommunicator(long ptr);
    private static native long allReduceFloat(long comm, float[] sendBuf, int sendOffset, float[] recvBuf, int recvOffset, int len);
    private static native long allReduceFloatAsync(long comm, String tensorName, float[] sendBuf, int sendOffset, int len);
    private static native long createTensorCache(long comm, String tensorName, int len);
    private static native long allReduceFloatCached(long comm, long cache, float[] sendBuf, int sendOffset, int priority);
    private static native long allReduceFP16Cached(long comm, long cache, float[] sendBuf, int sendOffset, int priority);
    private static native void waitRequest(long req);
    private static native boolean testRequest(long req);
    private static native void releaseRequest(long req);
    private static native void getResultFromRequest(long req, float[] buf, int offset);

    long ptrComm;
    public CCLAdapter(int color) {
        ptrComm = createCommunicator(color);
        System.out.println("Create COMM " + ptrComm );
    }
    public long allReduceFloat(float[] sendBuf, int sendOffset, float[] recvBuf, int recvOffset, int len) {
        return CCLAdapter.allReduceFloat(ptrComm, sendBuf, sendOffset, recvBuf, recvOffset, len);
    }

    public void release() {
        if (ptrComm != 0) {
            releaseCommunicator(ptrComm);
        }
        ptrComm = 0;
    }

    public static class RequestInfo {
        private long ptr;
        private RequestInfo(long ptr) {
            this.ptr = ptr;
        }

        public void await() {
            waitRequest(ptr);
        }

        public boolean test() {
            return testRequest(ptr);
        }

        public void release() {
            if (ptr != 0) {
                releaseRequest(ptr);
            }
            ptr = 0;
        }

        public void get(float[] buf, int offset) {
            getResultFromRequest(ptr, buf, offset);
            release();
        }
        protected void finalize() {
            release();
        }
    }

    public long createTensorCache(String tensorName, int len) {
        return CCLAdapter.createTensorCache(ptrComm, tensorName, len);
    }

    public RequestInfo allReduceFloatCached(long cacheId, float[] sendBuf, int sendOffset, int priority) {
        return new RequestInfo(
                CCLAdapter.allReduceFloatCached(ptrComm, cacheId, sendBuf, sendOffset, priority)
        );
    }

    public RequestInfo allReduceFP16Cached(long cacheId, float[] sendBuf, int sendOffset, int priority) {
        return new RequestInfo(
                CCLAdapter.allReduceFP16Cached(ptrComm, cacheId, sendBuf, sendOffset, priority)
        );
    }

    public RequestInfo allReduceFloatAsync(String name, float[] sendBuf, int sendOffset, int len) {
        return new RequestInfo(
                CCLAdapter.allReduceFloatAsync(ptrComm, name, sendBuf, sendOffset, len)
        );
    }
}
