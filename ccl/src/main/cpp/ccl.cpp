#include "com_intel_analytics_bigdl_ccl_CCLAdapter.h"
#include <iostream>
#include <ccl.hpp>
#include <assert.h>
#include <string>
#include <vector>

size_t cclRank = -1, cclSize = -1;
using namespace ccl;

static inline uint16_t float2fp16(float* v)
{
#if __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__
    return *reinterpret_cast<uint16_t*>(v);
#else
    return *&(reinterpret_cast<uint16_t*>(v)[1]);
#endif
}

static inline float fp162float(uint16_t v)
{
    static_assert(sizeof(float) == sizeof(uint32_t), "Expecting sizeof(float) == sizeof(uint32_t)");
    uint32_t expanded = ((uint32_t)v) << 16;
    return *reinterpret_cast<float*>(&expanded);
}


// a base class for CCL related memory
// it contains a checker to prevent use-after-free and double-free
template <int ID>
struct __CCLObject {
    // FREE_CHECKING_MAGIC = hash(line_number)
    static constexpr int FREE_CHECKING_MAGIC = int(0x2a9d8c5bL * ID) ^ ID ^ (ID << 16);
    ~__CCLObject()
    {
        checkIsFreed();
        free_check = 0;
    }

    void checkIsFreed() {
        assert(FREE_CHECKING_MAGIC == free_check);
    }
private:
    int free_check = FREE_CHECKING_MAGIC;

};

#define CCLBase __CCLObject<__LINE__>

struct CCLRequest;

struct CCLTensorCache : CCLBase
{
    std::string name;
    jint len;
    std::unique_ptr<CCLRequest> cached_req;
    bool free = true;

    CCLTensorCache(std::string&& name, jint len);
    CCLRequest* getRequest();
};

struct CCLRequest: CCLBase
{
    communicator::coll_request_t req;
    std::unique_ptr<float[]> readBuf;
    std::unique_ptr<float[]> writeBuf;
    CCLTensorCache* owner;
    jint len;
    coll_attr attr;
    bool isFP16 = false;

    void finalize(JNIEnv *env, jfloatArray recvBuf, jint recvOff) {
        if (isFP16) {
            //convert fp16 from writeBuf to fp32
            auto fp16buffer = reinterpret_cast<uint16_t*>(writeBuf.get());
            for (int i=0; i<len; i++) {
               readBuf[i] = fp162float(fp16buffer[i]);
            }
            env->SetFloatArrayRegion(recvBuf, recvOff, len, readBuf.get());
        }
        else
        {
            env->SetFloatArrayRegion(recvBuf, recvOff, len, writeBuf.get());
        }
    }

    const std::string& getName() const 
    {
        if(owner)
            return owner->name;
        return name;
    }

    CCLRequest(std::string&& name, jint len) 
    : len(len), name(std::move(name)), owner(nullptr)
    {
        init();
    }

    CCLRequest(CCLTensorCache* owner, jint len) 
    : len(len), owner(owner)
    {
        init();
    }

private:
    std::string name;
    void init() {
        readBuf = std::unique_ptr<float[]>(new float[len]);
        writeBuf = std::unique_ptr<float[]>(new float[len]);
        attr.prologue_fn = NULL;
        attr.epilogue_fn = NULL;
        attr.reduction_fn = NULL;
        attr.priority = 0;
        attr.synchronous = 0;
        attr.match_id = getName().c_str();
        attr.to_cache = owner ? 1: 0;
    }
};

CCLTensorCache::CCLTensorCache(std::string&& name, jint len):
        name(std::move(name)), len(len)
{
            cached_req = std::make_unique<CCLRequest>(this, len);
}

CCLRequest* CCLTensorCache::getRequest()
{
    assert(this->free);
    this->free = false;
    return cached_req.get();
}

struct CCLCommunicator: CCLBase {
    communicator_t comm;
    std::vector<std::unique_ptr<CCLTensorCache>> tensorCache;
    
    CCLCommunicator(communicator_t&& comm): comm(std::move(comm))
    {}
};

std::vector<std::unique_ptr<CCLCommunicator>> comm;
environment* penv = nullptr;

static char cclErrorStr[][34] = {
    "ccl_status_success",
    "ccl_status_out_of_resource",
    "ccl_status_invalid_arguments",
    "ccl_status_unimplemented",
    "ccl_status_runtime_error",
    "ccl_status_blocked_due_to_resize",
};

ccl_resize_action_t simple_framework_func(size_t comm_size)
{

    std::cout<<"simple_framework_func "<<comm_size<<std::endl;
    // We have 2 or more ranks, so we can to start communication.
    if (comm_size >= cclSize)
    {
        return  ccl_ra_run;
    }
    else if (comm_size == 0)
    {
        return  ccl_ra_finalize;
    }
    // We have less that 1 rank, so we should to finalize.
    else
    {
        return ccl_ra_wait;
    }
}

static inline void cclCheckResult(JNIEnv *env, ccl_status_t status) {
    if(status == ccl_status_success) {
        return;
    }
    const char* str;
    if (status >= ccl_status_last_value) {
        std::string s = "CCL Unknown Error: " + std::to_string((int)status);
        str = s.c_str();
    }
    else {
        str = cclErrorStr[(int)status];
    }
    env->ThrowNew(env->FindClass("java/lang/Exception"), str);
}

#define ASSERTOK(s) cclCheckResult(env, (s))

JNIEXPORT void JNICALL Java_com_intel_analytics_bigdl_ccl_CCLAdapter_sayHello(JNIEnv *env, jclass cls) {
    std::cout<<"HI\n";
}

JNIEXPORT void JNICALL Java_com_intel_analytics_bigdl_ccl_CCLAdapter_initCCL(JNIEnv *env, jclass cls) {
    std::cout<<"CCL_K8S_API_ADDR"<<getenv("CCL_K8S_API_ADDR")<<std::endl;
    std::cout<<"CCL_ATL_TRANSPORT"<<getenv("CCL_ATL_TRANSPORT")<<std::endl;
    std::cout<<"CCL_PM_TYPE"<<getenv("CCL_PM_TYPE")<<std::endl;
    std::cout<<"CCL_WORLD_SIZE"<<getenv("CCL_WORLD_SIZE")<<std::endl;
    penv = &environment::instance();
    comm.emplace_back(std::make_unique<CCLCommunicator>(penv->create_communicator()));
    cclRank = comm.back()->comm->rank();
    cclSize = comm.back()->comm->size();
    std::cout<<"CCL SIZE "<< cclSize << std::endl;
    penv->set_resize_fn(simple_framework_func);
}

JNIEXPORT void JNICALL Java_com_intel_analytics_bigdl_ccl_CCLAdapter_finalizeCCL(JNIEnv *env, jclass cls) {
    comm.clear();
}

JNIEXPORT jlong JNICALL Java_com_intel_analytics_bigdl_ccl_CCLAdapter_allReduceFloat(JNIEnv *env, jclass cls, jlong ptrComm, jfloatArray sendBuf, jint sendOff, jfloatArray recvBuf, jint recvOff, jint len){
    jboolean isCopy = JNI_FALSE;
    float* sbuf = (float*) env->GetPrimitiveArrayCritical(sendBuf, &isCopy);
    assert(!isCopy);
    isCopy = JNI_FALSE;
    float* rbuf = (float*) env->GetPrimitiveArrayCritical(recvBuf, &isCopy);
    assert(!isCopy);
    ccl_request_t request;
    CCLCommunicator* comm = reinterpret_cast<CCLCommunicator*>(ptrComm);
    std::cout<<"ALL RED COMM "<< (void*)comm << std::endl;
    coll_attr cclCollAttr;
    cclCollAttr.prologue_fn = NULL;
    cclCollAttr.epilogue_fn = NULL;
    cclCollAttr.reduction_fn = NULL;
    cclCollAttr.priority = 0;
    cclCollAttr.synchronous = 1;
    cclCollAttr.match_id = nullptr;
    cclCollAttr.to_cache = 0;
    auto status = comm->comm->allreduce(sbuf + sendOff, rbuf + recvOff, len, reduction::sum, &cclCollAttr);
    static_assert(sizeof(jlong) >= sizeof(request), "expecting sizeof(jlong) >= sizeof(request)");
    env->ReleasePrimitiveArrayCritical(recvBuf, rbuf, 0);
    env->ReleasePrimitiveArrayCritical(sendBuf, sbuf, 0);
    return (jlong)request;
}

JNIEXPORT void JNICALL Java_com_intel_analytics_bigdl_ccl_CCLAdapter_setEnv(JNIEnv *env, jclass cls, jstring apiServer, jint nNodes){
    const char* server = env->GetStringUTFChars(apiServer, nullptr);
    setenv("CCL_K8S_API_ADDR", server, 1);
    env->ReleaseStringUTFChars(apiServer, server);
    setenv("CCL_ATL_TRANSPORT", "ofi", 1);
    setenv("CCL_PM_TYPE", "resizable", 1);
    setenv("CCL_WORLD_SIZE", std::to_string(nNodes).c_str(), 1);
    cclSize = nNodes;
}

JNIEXPORT jlong JNICALL Java_com_intel_analytics_bigdl_ccl_CCLAdapter_createCommunicator(JNIEnv *env, jclass cls, jint color){
    comm_attr attr = {color};
    auto ret = std::make_unique<CCLCommunicator>(penv->create_communicator(&attr));
    std::cout<<"COMM "<< (void*)ret.get() << std::endl;
    std::cout<<"Subsize "<< ret->comm->size() << std::endl;
    CCLCommunicator* ptr = ret.get();
    comm.emplace_back(std::move(ret));
    static_assert(sizeof(jlong) >= sizeof(void*), "expecting sizeof(jlong) >= sizeof(void*)");
    return (jlong)ptr;
}

JNIEXPORT void JNICALL Java_com_intel_analytics_bigdl_ccl_CCLAdapter_releaseCommunicator(JNIEnv *env, jclass cls, jlong ptr){
    CCLCommunicator* ths = reinterpret_cast<CCLCommunicator*>(ptr);
    for (auto& ptr: comm) {
        if (ptr.get() == ths) {
            ptr = nullptr; //release
            return;
        }
    }
    assert(0 && "Cannot find the communicator to release!");
}

JNIEXPORT jlong JNICALL Java_com_intel_analytics_bigdl_ccl_CCLAdapter_allReduceFloatCached(JNIEnv *env, jclass cls, jlong ptrComm, jlong ptrCache, jfloatArray sendBuf, jint sendOff, jint priority){
    CCLCommunicator* comm = reinterpret_cast<CCLCommunicator*>(ptrComm);
    CCLTensorCache* cache = reinterpret_cast<CCLTensorCache*>(ptrCache);
    auto ret = cache->getRequest();
    ret->isFP16 = false;
    env->GetFloatArrayRegion(sendBuf, sendOff, ret->len, ret->readBuf.get());
    coll_attr& attr = ret->attr;
    attr.priority = priority;
    auto req = comm->comm->allreduce(ret->readBuf.get(), ret->writeBuf.get(), ret->len, reduction::sum, &attr);
    ret->req = std::move(req);
    return (jlong)ret;
}

JNIEXPORT jlong JNICALL Java_com_intel_analytics_bigdl_ccl_CCLAdapter_allReduceFP16Cached(JNIEnv *env, jclass cls, jlong ptrComm, jlong ptrCache, jfloatArray sendBuf, jint sendOff, jint priority){
    CCLCommunicator* comm = reinterpret_cast<CCLCommunicator*>(ptrComm);
    CCLTensorCache* cache = reinterpret_cast<CCLTensorCache*>(ptrCache);
    auto ret = cache->getRequest();
    ret->isFP16 = true;
    auto buffer = ret->readBuf.get();
    auto fp16buffer = reinterpret_cast<uint16_t*>(buffer);
    //copy floats to the buffer
    env->GetFloatArrayRegion(sendBuf, sendOff, ret->len, buffer);
    //now in-place convert them to fp16
    for (int i = 0; i < ret->len; i++){
        uint16_t data = float2fp16(buffer + i);
        fp16buffer[i] = data;
    }
    coll_attr& attr = ret->attr;
    attr.priority = priority;
    auto req = comm->comm->allreduce(reinterpret_cast<void*>(buffer), reinterpret_cast<void*>(ret->writeBuf.get()),
        ret->len, ccl::data_type::dt_bfp16, reduction::sum, &attr);
    ret->req = std::move(req);
    return (jlong)ret;
}

JNIEXPORT jlong JNICALL Java_com_intel_analytics_bigdl_ccl_CCLAdapter_createTensorCache(JNIEnv *env, jclass cls, jlong ptrComm, jstring tensorName, jint len){
    CCLCommunicator* comm = reinterpret_cast<CCLCommunicator*>(ptrComm);
    const char* name = env->GetStringUTFChars(tensorName, nullptr);
    std::string tname = std::string(name);
    env->ReleaseStringUTFChars(tensorName, name);
    comm->tensorCache.emplace_back(std::make_unique<CCLTensorCache>(std::move(tname), len));
    return (jlong)comm->tensorCache.back().get();
}

JNIEXPORT jlong JNICALL Java_com_intel_analytics_bigdl_ccl_CCLAdapter_allReduceFloatAsync(JNIEnv *env, jclass cls, jlong ptrComm, jstring tensorName, jfloatArray sendBuf, jint sendOff, jint len){

    CCLCommunicator* comm = reinterpret_cast<CCLCommunicator*>(ptrComm);
    const char* name = env->GetStringUTFChars(tensorName, nullptr);
    std::string tname = std::string(name);
    env->ReleaseStringUTFChars(tensorName, name);
    auto ret = new CCLRequest(std::move(tname), len);
    env->GetFloatArrayRegion(sendBuf, sendOff, len, ret->readBuf.get());
    coll_attr& attr = ret->attr;
    auto req = comm->comm->allreduce(ret->readBuf.get(), ret->writeBuf.get(), len, reduction::sum, &attr);
    ret->req = std::move(req);
    return (jlong)ret;
}

JNIEXPORT void JNICALL Java_com_intel_analytics_bigdl_ccl_CCLAdapter_waitRequest(JNIEnv *env, jclass cls, jlong ptr) {
    CCLRequest* req = reinterpret_cast<CCLRequest*>(ptr);
    req->req->wait();
}

JNIEXPORT jboolean JNICALL Java_com_intel_analytics_bigdl_ccl_CCLAdapter_testRequest(JNIEnv *env, jclass cls, jlong ptr) {
    CCLRequest* req = reinterpret_cast<CCLRequest*>(ptr);
    bool ret = req->req->test();
    return ret;
}

JNIEXPORT void JNICALL Java_com_intel_analytics_bigdl_ccl_CCLAdapter_releaseRequest(JNIEnv *env, jclass cls, jlong ptr) {
    CCLRequest* req = reinterpret_cast<CCLRequest*>(ptr);
    if (req->owner) //if is cached
    {
        // check if this is freed to prevent "use-after-free"
        req->checkIsFreed();
        req->owner->checkIsFreed();
        req->req = nullptr; // release CCL request
        req->owner->free = true;
    }
    else
    {
        delete req;
    }
}

JNIEXPORT void JNICALL Java_com_intel_analytics_bigdl_ccl_CCLAdapter_getResultFromRequest(JNIEnv *env, jclass cls, jlong ptr, jfloatArray outArray, jint offset) {
    CCLRequest* req = reinterpret_cast<CCLRequest*>(ptr);
    req->finalize(env, outArray, offset);
}
