package com.intel.analytics.bigdl.ccl;

import java.io.File;
import java.io.FileOutputStream;
import java.io.InputStream;
import java.net.URL;
import java.nio.channels.Channels;
import java.nio.channels.FileChannel;
import java.nio.channels.ReadableByteChannel;

public class CCLAdapter {

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

    static void load() {
        System.loadLibrary("ccl");
        File tmpFile = extract("libccl_java.so");
        try {
            System.load(tmpFile.getAbsolutePath());
        } finally {
            tmpFile.delete();
        }
    }

    static void doInit()
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
    private static native long allReduceFloat(long comm, float[] sendBuf, int sendOffset, float[] recvBuf, int recvOffset, int len);
    private static native long allReduceFloatAsync(long comm, float[] sendBuf, int sendOffset, int len);
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
        }

        public void get(float[] buf, int offset) {
            getResultFromRequest(ptr, buf, offset);
            release();
        }
        protected void finalize() {
            release();
        }
    }

    public RequestInfo allReduceFloatAsync(float[] sendBuf, int sendOffset, int len) {
        return new RequestInfo(
                CCLAdapter.allReduceFloatAsync(ptrComm, sendBuf, sendOffset, len)
        );
    }
}
