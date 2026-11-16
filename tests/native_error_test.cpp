// Exercise the real native callbacks with SDK-compatible NNRT/N-API mocks.
#include "../entry/src/main/cpp/permission_hiai.cpp"
#include <cassert>
#include <iostream>
#include <thread>

struct napi_value__ {
    napi_value__(std::string value, bool error) : text(std::move(value)), isError(error) {}
    std::string text;
    bool isError;
    bool typed = false, arrayBuffer = false;
    napi_typedarray_type type = napi_int32_array;
    uint32_t length = 64, byteLength = 256, number = 0;
    size_t reportedLength = 256, offset = 0;
    napi_value buffer = nullptr;
    void *wrapped = nullptr;
    std::vector<uint8_t> bytes;
};
struct napi_deferred__ {};
struct napi_async_work__ {};
struct NN_TensorDesc { size_t index; bool output; };
struct NN_Tensor { std::array<uint8_t, 256> bytes{}; };

namespace {
int buildMode = 0;
int compatibility = 0;
bool hasCannBackend = true;
bool failErrorCreation = false;
bool failStringCreation = false;
bool failArrayCreation = false;
bool failResolve = false;
bool pendingErrorCreation = false;
bool pending = false;
int rejected = 0, deleted = 0, destroyed = 0, buildCalls = 0;
int runCalls = 0;
std::string thrown;
std::vector<napi_value> callbackArgs;
void *queuedData = nullptr;
napi_async_execute_callback queuedExecute = nullptr;
napi_async_complete_callback queuedComplete = nullptr;
bool failPropertyRead = false;
std::string rejection;
bool rejectionIsError = false;
std::vector<std::unique_ptr<napi_value__>> jsValues;
napi_deferred__ deferred;
napi_async_work__ work;
int compilationToken;
int executorToken;
const char *faultAt = nullptr;
bool badTensorType = false;
bool nullShape = false;
bool badElementCount = false;
bool verifyTokenBuffers = false;
std::array<std::vector<int32_t>, 2> inputShapes{{{1, 16}, {1, 16}}};
const Logits expectedLogits{{0.5f, -0.25f, 1.5f, -2.0f, 0.0f}};
struct VendorFault {};
napi_value pendingReason = nullptr;

void MaybeThrow(const char *operation)
{
    if (faultAt && std::strcmp(faultAt, operation) == 0) throw VendorFault{};
}

napi_value Value(const std::string &text, bool error = false)
{
    jsValues.push_back(std::make_unique<napi_value__>(text, error));
    return jsValues.back().get();
}

void Reset()
{
    buildMode = 0;
    faultAt = nullptr;
    badTensorType = false;
    nullShape = badElementCount = verifyTokenBuffers = false;
    inputShapes = {{{1, 16}, {1, 16}}};
    compatibility = 0;
    hasCannBackend = true;
    failErrorCreation = failStringCreation = failArrayCreation = failResolve = pendingErrorCreation = pending = false;
    rejected = deleted = destroyed = buildCalls = 0;
    runCalls = 0;
    thrown.clear();
    callbackArgs.clear();
    queuedData = nullptr;
    queuedExecute = nullptr;
    queuedComplete = nullptr;
    failPropertyRead = false;
    rejection.clear();
    rejectionIsError = false;
    pendingReason = nullptr;
    jsValues.clear();
}

Job *NewJob(Operation operation)
{
    auto *job = new Job(operation);
    job->deferred = &deferred;
    job->work = &work;
    job->model = {1, 2, 3};
    return job;
}

void RunLoad(const std::string &expected)
{
    auto *job = NewJob(Operation::Load);
    std::thread worker([&] { Execute(nullptr, job); });
    worker.join();
    assert(std::string(job->error.data()).find(expected) != std::string::npos);
    assert(std::string(job->error.data()).find("exception_type=") != std::string::npos);
    Complete(nullptr, napi_ok, job);
    assert(rejected == 1 && deleted == 1);
    assert(rejection.find(expected) != std::string::npos);
}

void RunTensorRoundTrip(const std::vector<int32_t> &shape)
{
    Reset();
    buildMode = 3;
    inputShapes = {{shape, shape}};
    verifyTokenBuffers = true;
    auto *job = NewJob(Operation::Load);
    std::thread loader([&] { Execute(nullptr, job); }); loader.join();
    assert(job->session && job->error[0] == '\0');
    job->operation = Operation::Predict;
    for (size_t i = 0; i < job->ids.size(); ++i) {
        job->ids[i] = static_cast<int32_t>(100 + i);
        job->mask[i] = i % 2;
    }
    std::thread predictor([&] { Execute(nullptr, job); }); predictor.join();
    assert(job->error[0] == '\0' && job->logits == expectedLogits);
    Complete(nullptr, napi_ok, job);
    assert(rejected == 0 && deleted == 1);
}

napi_value TokenView(bool ids, size_t reportedLength, size_t offset)
{
    auto buffer = Value("ArrayBuffer");
    buffer->arrayBuffer = true;
    buffer->bytes.resize(offset + 256 + 16, 0x7f);
    for (int32_t i = 0; i < 64; ++i) {
        int32_t value = ids ? 100 + i : i % 2;
        std::memcpy(buffer->bytes.data() + offset + i * sizeof(value), &value, sizeof(value));
    }
    auto view = Value("Int32Array");
    view->typed = true;
    view->reportedLength = reportedLength;
    view->offset = offset;
    view->buffer = buffer;
    return view;
}

std::shared_ptr<Session> PrepareNapiPredict(int32_t length, size_t reportedLength, size_t offset)
{
    Reset(); buildMode = 3; verifyTokenBuffers = true;
    inputShapes = {{{1, length, 1, 1}, {1, length, 1, 1}}};
    auto *job = NewJob(Operation::Load);
    Execute(nullptr, job);
    assert(job->session && job->error[0] == '\0');
    auto session = job->session;
    delete job;
    callbackArgs = {Value("session"), TokenView(true, reportedLength, offset), TokenView(false, reportedLength, offset)};
    return session;
}

void RunNapiRoundTrip(int32_t length, size_t reportedLength, size_t offset)
{
    auto session = PrepareNapiPredict(length, reportedLength, offset);
    callbackArgs[0]->wrapped = &session;
    assert(Predict(nullptr, nullptr) && queuedData && queuedExecute && queuedComplete && thrown.empty());
    auto *job = static_cast<Job *>(queuedData);
    // The queued job must own copied values, independent of JS buffer lifetime.
    callbackArgs[1]->buffer->bytes.clear();
    callbackArgs[2]->buffer->bytes.clear();
    std::thread worker([&] { queuedExecute(nullptr, queuedData); }); worker.join();
    assert(job->error[0] == '\0' && job->logits == expectedLogits && runCalls == 1);
    queuedComplete(nullptr, napi_ok, queuedData);
    queuedData = nullptr;
    assert(rejected == 0 && deleted == 1);
}
} // namespace

extern "C" int PermissionTestCompatibility() { MaybeThrow("compatibility"); return compatibility; }
extern "C" void napi_module_register(napi_module *) {}
extern "C" napi_status napi_create_string_utf8(napi_env, const char *s, size_t n, napi_value *out)
{
    if (failStringCreation) return napi_generic_failure;
    *out = Value(n == NAPI_AUTO_LENGTH ? std::string(s) : std::string(s, n));
    return napi_ok;
}
extern "C" napi_status napi_create_error(napi_env, napi_value, napi_value text, napi_value *out)
{
    if (pendingErrorCreation) {
        pending = true;
        pendingReason = Value("pending JS error", true);
        return napi_pending_exception;
    }
    if (failErrorCreation) return napi_generic_failure;
    *out = Value(text->text, true);
    return napi_ok;
}
extern "C" napi_status napi_get_undefined(napi_env, napi_value *out) { *out = Value("undefined"); return napi_ok; }
extern "C" napi_status napi_is_exception_pending(napi_env, bool *out) { *out = pending; return napi_ok; }
extern "C" napi_status napi_get_and_clear_last_exception(napi_env, napi_value *out)
{ *out = pendingReason; pending = false; return napi_ok; }
extern "C" napi_status napi_reject_deferred(napi_env, napi_deferred, napi_value reason)
{
    assert(reason && !pending);
    rejected++;
    rejection = reason->text;
    rejectionIsError = reason->isError;
    return napi_ok;
}
extern "C" napi_status napi_resolve_deferred(napi_env, napi_deferred, napi_value)
{ return failResolve ? napi_generic_failure : napi_ok; }
extern "C" napi_status napi_delete_async_work(napi_env, napi_async_work) { deleted++; return napi_ok; }
extern "C" napi_status napi_create_object(napi_env, napi_value *out) { *out = Value("object"); return napi_ok; }
extern "C" napi_status napi_create_array_with_length(napi_env, size_t, napi_value *out)
{
    if (failArrayCreation) return napi_generic_failure;
    *out = Value("array");
    return napi_ok;
}
extern "C" napi_status napi_create_double(napi_env, double, napi_value *out) { *out = Value("number"); return napi_ok; }
extern "C" napi_status napi_set_element(napi_env, napi_value, uint32_t, napi_value) { return napi_ok; }
extern "C" napi_status napi_set_named_property(napi_env, napi_value, const char *, napi_value) { return napi_ok; }
extern "C" napi_status napi_wrap(napi_env, napi_value, void *, napi_finalize, void *, napi_ref *) { return napi_ok; }
extern "C" napi_status napi_throw_error(napi_env, const char *, const char *message)
{ thrown = message; return napi_ok; }
extern "C" napi_status napi_create_promise(napi_env, napi_deferred *out, napi_value *promise)
{ *out = &deferred; *promise = Value("promise"); return napi_ok; }
extern "C" napi_status napi_create_async_work(napi_env, napi_value, napi_value, napi_async_execute_callback execute,
    napi_async_complete_callback complete, void *data, napi_async_work *out)
{ queuedExecute = execute; queuedComplete = complete; queuedData = data; *out = &work; return napi_ok; }
extern "C" napi_status napi_queue_async_work(napi_env, napi_async_work) { return napi_ok; }
extern "C" napi_status napi_get_typedarray_info(napi_env, napi_value value, napi_typedarray_type *type, size_t *length,
    void **data, napi_value *buffer, size_t *offset)
{
    if (!value->typed) return napi_invalid_arg;
    *type = value->type; *length = value->reportedLength; *buffer = value->buffer; *offset = value->offset;
    if (data) *data = value->buffer->bytes.data() + value->offset;
    return napi_ok;
}
extern "C" napi_status napi_get_named_property(napi_env, napi_value object, const char *name, napi_value *out)
{
    if (failPropertyRead || !object->typed) return napi_generic_failure;
    if (std::strcmp(name, "length") != 0 && std::strcmp(name, "byteLength") != 0) return napi_generic_failure;
    *out = Value("uint32");
    (*out)->number = std::strcmp(name, "length") == 0 ? object->length : object->byteLength;
    return napi_ok;
}
extern "C" napi_status napi_get_value_uint32(napi_env, napi_value value, uint32_t *out)
{ *out = value->number; return napi_ok; }
extern "C" napi_status napi_get_cb_info(napi_env, napi_callback_info, size_t *count, napi_value *args, napi_value *, void **)
{
    if (callbackArgs.empty() || *count < callbackArgs.size()) return napi_generic_failure;
    *count = callbackArgs.size();
    for (size_t i = 0; i < *count; ++i) args[i] = callbackArgs[i];
    return napi_ok;
}
extern "C" napi_status napi_get_arraybuffer_info(napi_env, napi_value buffer, void **data, size_t *bytes)
{
    if (!buffer->arrayBuffer) return napi_arraybuffer_expected;
    *data = buffer->bytes.empty() ? nullptr : buffer->bytes.data(); *bytes = buffer->bytes.size(); return napi_ok;
}
extern "C" napi_status napi_unwrap(napi_env, napi_value value, void **out)
{ *out = value->wrapped; return napi_ok; }
extern "C" napi_status napi_define_properties(napi_env, napi_value, size_t, const napi_property_descriptor *) { return napi_ok; }

extern "C" OH_NN_ReturnCode OH_NNDevice_GetAllDevicesID(const size_t **out, uint32_t *count)
{ static const size_t id = 42; *out = &id; *count = 1; return OH_NN_SUCCESS; }
extern "C" OH_NN_ReturnCode OH_NNDevice_GetName(size_t, const char **name)
{ *name = hasCannBackend ? "HIAI_F" : "NPU_ohos.boot.hardware.Kirin9020_v2_0"; return OH_NN_SUCCESS; }
extern "C" OH_NN_ReturnCode OH_NNDevice_GetType(size_t, OH_NN_DeviceType *type)
{ *type = OH_NN_ACCELERATOR; return OH_NN_SUCCESS; }
extern "C" OH_NNCompilation *OH_NNCompilation_ConstructWithOfflineModelBuffer(const void *, size_t)
{ return reinterpret_cast<OH_NNCompilation *>(&compilationToken); }
extern "C" OH_NN_ReturnCode OH_NNCompilation_SetDevice(OH_NNCompilation *, size_t id)
{ assert(id == 42); return OH_NN_SUCCESS; }
extern "C" OH_NN_ReturnCode OH_NNCompilation_Build(OH_NNCompilation *)
{
    buildCalls++;
    if (buildMode == 1) throw std::runtime_error("injected vendor exception");
    if (buildMode == 2) throw 17;
    if (buildMode == 4) throw std::string("vendor string detail");
    if (buildMode == 5) throw "vendor C string detail";
    if (buildMode == 6) throw VendorFault{};
    return buildMode == 3 ? OH_NN_SUCCESS : OH_NN_FAILED;
}
extern "C" void OH_NNCompilation_Destroy(OH_NNCompilation **p) { destroyed++; *p = nullptr; }
extern "C" OH_NNExecutor *OH_NNExecutor_Construct(OH_NNCompilation *)
{ MaybeThrow("executor"); return reinterpret_cast<OH_NNExecutor *>(&executorToken); }
extern "C" void OH_NNExecutor_Destroy(OH_NNExecutor **) {}
extern "C" OH_NN_ReturnCode OH_NNExecutor_GetInputCount(const OH_NNExecutor *, size_t *count)
{ *count = 2; return OH_NN_SUCCESS; }
extern "C" OH_NN_ReturnCode OH_NNExecutor_GetOutputCount(const OH_NNExecutor *, size_t *count)
{ *count = 1; return OH_NN_SUCCESS; }
extern "C" NN_TensorDesc *OH_NNExecutor_CreateInputTensorDesc(const OH_NNExecutor *, size_t index)
{ MaybeThrow("input descriptor"); return new NN_TensorDesc{index, false}; }
extern "C" NN_TensorDesc *OH_NNExecutor_CreateOutputTensorDesc(const OH_NNExecutor *, size_t index)
{ return new NN_TensorDesc{index, true}; }
extern "C" OH_NN_ReturnCode OH_NNTensorDesc_GetDataType(const NN_TensorDesc *desc, OH_NN_DataType *type)
{ *type = desc->output ? OH_NN_FLOAT32 : (badTensorType ? OH_NN_INT64 : OH_NN_INT32); return OH_NN_SUCCESS; }
extern "C" OH_NN_ReturnCode OH_NNTensorDesc_GetElementCount(const NN_TensorDesc *desc, size_t *count)
{
    *count = 5;
    if (!desc->output) {
        *count = 1;
        for (auto dim : inputShapes[desc->index]) *count *= static_cast<size_t>(dim);
        if (badElementCount) ++*count;
    }
    return OH_NN_SUCCESS;
}
extern "C" OH_NN_ReturnCode OH_NNTensorDesc_GetShape(const NN_TensorDesc *desc, int32_t **shape, size_t *rank)
{ auto &dims = inputShapes[desc->index]; *shape = nullShape ? nullptr : dims.data(); *rank = dims.size(); return OH_NN_SUCCESS; }
extern "C" OH_NN_ReturnCode OH_NNTensorDesc_GetName(const NN_TensorDesc *desc, const char **name)
{ *name = desc->index == 0 ? "input_ids" : "attention_mask"; return OH_NN_SUCCESS; }
extern "C" OH_NN_ReturnCode OH_NNTensorDesc_Destroy(NN_TensorDesc **p)
{ delete *p; *p = nullptr; return OH_NN_SUCCESS; }
extern "C" NN_Tensor *OH_NNTensor_Create(size_t, NN_TensorDesc *)
{ MaybeThrow("tensor"); return new NN_Tensor; }
extern "C" OH_NN_ReturnCode OH_NNTensor_GetSize(const NN_Tensor *, size_t *size)
{ *size = 256; return OH_NN_SUCCESS; }
extern "C" void *OH_NNTensor_GetDataBuffer(const NN_Tensor *tensor)
{ return const_cast<uint8_t *>(tensor->bytes.data()); }
extern "C" OH_NN_ReturnCode OH_NNTensor_Destroy(NN_Tensor **p)
{ delete *p; *p = nullptr; return OH_NN_SUCCESS; }
extern "C" OH_NN_ReturnCode OH_NNExecutor_RunSync(OH_NNExecutor *, NN_Tensor *inputs[], size_t, NN_Tensor *outputs[], size_t)
{
    MaybeThrow("run");
    runCalls++;
    if (verifyTokenBuffers) {
        size_t length = 1;
        for (auto dim : inputShapes[0]) length *= static_cast<size_t>(dim);
        for (size_t i = 0; i < length; ++i) {
            int32_t id = 0, mask = 0;
            std::memcpy(&id, inputs[0]->bytes.data() + i * sizeof(int32_t), sizeof(int32_t));
            std::memcpy(&mask, inputs[1]->bytes.data() + i * sizeof(int32_t), sizeof(int32_t));
            assert(id == static_cast<int32_t>(100 + i) && mask == static_cast<int32_t>(i % 2));
        }
    }
    std::memcpy(outputs[0]->bytes.data(), expectedLogits.data(), sizeof(expectedLogits));
    return OH_NN_SUCCESS;
}

int main()
{
    Reset(); RunLoad("OH_NNCompilation_Build"); assert(buildCalls == 1 && destroyed == 1 && rejectionIsError);
    Reset(); hasCannBackend = false; RunLoad("generic NNRT accelerators cannot load OMC"); assert(buildCalls == 0);
    Reset(); compatibility = 1; RunLoad("compatibility check rejected"); assert(buildCalls == 0);
    Reset(); buildMode = 1; RunLoad("injected vendor exception"); assert(destroyed == 1);
    Reset(); buildMode = 2; RunLoad("Native exception int=17"); assert(destroyed == 1);
    assert(rejection.find("OH_NNCompilation_Build (not returned)") != std::string::npos);
    assert(rejection.find("exception_type=i;") != std::string::npos);
    Reset(); buildMode = 4; RunLoad("vendor string detail");
    Reset(); buildMode = 5; RunLoad("vendor C string detail");
    Reset(); buildMode = 6; RunLoad("Unknown native exception");
    assert(rejection.find("VendorFault") != std::string::npos);
    Reset(); faultAt = "compatibility"; RunLoad("HMS_HiAICompatibility_CheckFromBuffer (not returned)");
    assert(buildCalls == 0);
    Reset(); buildMode = 3; faultAt = "executor"; RunLoad("OH_NNExecutor_Construct (not returned)");
    Reset(); buildMode = 3; faultAt = "input descriptor";
    RunLoad("OH_NNExecutor_CreateInputTensorDesc[index=0] (not returned)");
    Reset(); buildMode = 3; faultAt = "tensor"; RunLoad("OH_NNTensor_Create input[index=0] (not returned)");
    Reset(); buildMode = 3; badTensorType = true; RunLoad("Offline model tensor mismatch: type=");
    Reset(); failErrorCreation = true; RunLoad("OH_NNCompilation_Build"); assert(!rejectionIsError);
    Reset(); pendingErrorCreation = true;
    auto *job = NewJob(Operation::Load); std::snprintf(job->error.data(), job->error.size(), "original failure");
    Complete(nullptr, napi_ok, job); assert(rejection == "pending JS error" && !pending && deleted == 1);
    Reset(); failStringCreation = true;
    job = NewJob(Operation::Load); std::snprintf(job->error.data(), job->error.size(), "original failure");
    Complete(nullptr, napi_ok, job); assert(rejection == "undefined" && deleted == 1);
    Reset(); failArrayCreation = true;
    Complete(nullptr, napi_ok, NewJob(Operation::Predict));
    assert(rejected == 1 && deleted == 1 && rejection.find("Cannot allocate logits array") != std::string::npos);
    Reset(); failResolve = true;
    Complete(nullptr, napi_ok, NewJob(Operation::Predict));
    assert(rejected == 1 && deleted == 1 && rejection.find("Cannot resolve native promise") != std::string::npos);
    Reset(); Complete(nullptr, napi_generic_failure, NewJob(Operation::Load));
    assert(rejection.find("cancelled") != std::string::npos && deleted == 1);
    Reset(); buildMode = 3;
    job = NewJob(Operation::Load);
    Execute(nullptr, job);
    assert(job->session && job->error[0] == '\0');
    auto session = job->session;
    delete job;
    job = NewJob(Operation::Predict); job->session = session;
    Execute(nullptr, job); assert(job->error[0] == '\0');
    Complete(nullptr, napi_ok, job); assert(rejected == 0 && deleted == 1);
    faultAt = "run";
    job = NewJob(Operation::Predict); job->session = session;
    std::thread worker([&] { Execute(nullptr, job); }); worker.join();
    Complete(nullptr, napi_ok, job);
    assert(rejected == 1 && deleted == 2);
    assert(rejection.find("Predict failed at OH_NNExecutor_RunSync (not returned)") != std::string::npos);
    assert(rejection.find("VendorFault") != std::string::npos);
    session.reset();
    for (int32_t length : {16, 32, 64}) {
        RunTensorRoundTrip({length});
        RunTensorRoundTrip({1, length});
        RunTensorRoundTrip({1, length, 1, 1});
        RunTensorRoundTrip({1, 1, 1, length});
    }
    for (const std::vector<int32_t> &shape : std::vector<std::vector<int32_t>>{
            {2, 8}, {1, 17}, {1, -1}, {0, 16}, {16, 16}, {1, 1}, {}, {1, 1, 1, 1, 1, 1, 1, 1, 16}}) {
        Reset(); buildMode = 3; inputShapes[0] = shape;
        RunLoad("Unsupported token input shape");
        assert(rejection.find("Validate token input shape") != std::string::npos);
        assert(rejection.find("BridgeError") != std::string::npos);
        assert(rejection.find("Unknown native exception") == std::string::npos);
    }
    Reset(); buildMode = 3; nullShape = true; RunLoad("Unsupported token input shape <null>");
    Reset(); buildMode = 3; inputShapes[1] = {1, 32}; RunLoad("NNRT token and mask shapes differ");
    Reset(); buildMode = 3; badElementCount = true; RunLoad("Offline model tensor mismatch: type=");
    for (int32_t length : {16, 32, 64}) {
        for (size_t reportedLength : {64, 256}) {
            RunNapiRoundTrip(length, reportedLength, 0);
            RunNapiRoundTrip(length, reportedLength, 16);
        }
    }
    for (int scenario = 0; scenario < 13; ++scenario) {
        auto current = PrepareNapiPredict(16, 256, 0);
        callbackArgs[0]->wrapped = &current;
        auto view = callbackArgs[1];
        switch (scenario) {
            case 0: view->type = napi_float32_array; break;
            case 1: view->length = 256; view->byteLength = 1024; view->buffer->bytes.resize(1024); break;
            case 2: view->length = 63; view->byteLength = 252; view->reportedLength = 252; break;
            case 3: view->byteLength = 255; break;
            case 4: view->reportedLength = 65; break;
            case 5: view->buffer->bytes.resize(255); break;
            case 6: view->offset = 4; view->buffer->bytes.resize(256); break;
            case 7: view->offset = static_cast<size_t>(-1); break;
            case 8: view->offset = 2; break;
            case 9: view->buffer->bytes.clear(); break;
            case 10: view->typed = false; break;
            case 11: failPropertyRead = true; break;
            case 12: view->buffer->arrayBuffer = false; break;
        }
        assert(!Predict(nullptr, nullptr) && !queuedData && runCalls == 0 && !thrown.empty());
        if (scenario < 10) assert(thrown.find("reportedLength=") != std::string::npos);
    }
    std::cout << "PASS: 69 native scenarios: prior 44 plus 12 N-API round trips and 13 invalid TypedArray cases\n";
}
