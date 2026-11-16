// Exercise the real native callbacks with SDK-compatible NNRT/N-API mocks.
#include "../entry/src/main/cpp/permission_hiai.cpp"
#include <cassert>
#include <iostream>
#include <thread>

struct napi_value__ { std::string text; bool isError = false; };
struct napi_deferred__ {};
struct napi_async_work__ {};

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
std::string rejection;
bool rejectionIsError = false;
std::vector<std::unique_ptr<napi_value__>> jsValues;
napi_deferred__ deferred;
napi_async_work__ work;
int compilationToken;
napi_value pendingReason = nullptr;

napi_value Value(const std::string &text, bool error = false)
{
    jsValues.push_back(std::make_unique<napi_value__>(napi_value__{text, error}));
    return jsValues.back().get();
}

void Reset()
{
    buildMode = 0;
    compatibility = 0;
    hasCannBackend = true;
    failErrorCreation = failStringCreation = failArrayCreation = failResolve = pendingErrorCreation = pending = false;
    rejected = deleted = destroyed = buildCalls = 0;
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
    Complete(nullptr, napi_ok, job);
    assert(rejected == 1 && deleted == 1);
    assert(rejection.find(expected) != std::string::npos);
}
} // namespace

extern "C" int PermissionTestCompatibility() { return compatibility; }
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
extern "C" napi_status napi_throw_error(napi_env, const char *, const char *) { return napi_ok; }
extern "C" napi_status napi_create_promise(napi_env, napi_deferred *, napi_value *) { return napi_ok; }
extern "C" napi_status napi_create_async_work(napi_env, napi_value, napi_value, napi_async_execute_callback,
    napi_async_complete_callback, void *, napi_async_work *) { return napi_ok; }
extern "C" napi_status napi_queue_async_work(napi_env, napi_async_work) { return napi_ok; }
extern "C" napi_status napi_get_typedarray_info(napi_env, napi_value, napi_typedarray_type *, size_t *, void **,
    napi_value *, size_t *) { return napi_generic_failure; }
extern "C" napi_status napi_get_cb_info(napi_env, napi_callback_info, size_t *, napi_value *, napi_value *, void **)
{ return napi_generic_failure; }
extern "C" napi_status napi_get_arraybuffer_info(napi_env, napi_value, void **, size_t *) { return napi_generic_failure; }
extern "C" napi_status napi_unwrap(napi_env, napi_value, void **) { return napi_generic_failure; }
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
    return OH_NN_FAILED;
}
extern "C" void OH_NNCompilation_Destroy(OH_NNCompilation **p) { destroyed++; *p = nullptr; }
extern "C" OH_NNExecutor *OH_NNExecutor_Construct(OH_NNCompilation *) { return nullptr; }
extern "C" void OH_NNExecutor_Destroy(OH_NNExecutor **) {}
extern "C" OH_NN_ReturnCode OH_NNExecutor_GetInputCount(const OH_NNExecutor *, size_t *) { return OH_NN_FAILED; }
extern "C" OH_NN_ReturnCode OH_NNExecutor_GetOutputCount(const OH_NNExecutor *, size_t *) { return OH_NN_FAILED; }
extern "C" NN_TensorDesc *OH_NNExecutor_CreateInputTensorDesc(const OH_NNExecutor *, size_t) { return nullptr; }
extern "C" NN_TensorDesc *OH_NNExecutor_CreateOutputTensorDesc(const OH_NNExecutor *, size_t) { return nullptr; }
extern "C" OH_NN_ReturnCode OH_NNTensorDesc_GetDataType(const NN_TensorDesc *, OH_NN_DataType *) { return OH_NN_FAILED; }
extern "C" OH_NN_ReturnCode OH_NNTensorDesc_GetElementCount(const NN_TensorDesc *, size_t *) { return OH_NN_FAILED; }
extern "C" OH_NN_ReturnCode OH_NNTensorDesc_GetShape(const NN_TensorDesc *, int32_t **, size_t *) { return OH_NN_FAILED; }
extern "C" OH_NN_ReturnCode OH_NNTensorDesc_GetName(const NN_TensorDesc *, const char **) { return OH_NN_FAILED; }
extern "C" OH_NN_ReturnCode OH_NNTensorDesc_Destroy(NN_TensorDesc **) { return OH_NN_SUCCESS; }
extern "C" NN_Tensor *OH_NNTensor_Create(size_t, NN_TensorDesc *) { return nullptr; }
extern "C" OH_NN_ReturnCode OH_NNTensor_GetSize(const NN_Tensor *, size_t *) { return OH_NN_FAILED; }
extern "C" void *OH_NNTensor_GetDataBuffer(const NN_Tensor *) { return nullptr; }
extern "C" OH_NN_ReturnCode OH_NNTensor_Destroy(NN_Tensor **) { return OH_NN_SUCCESS; }
extern "C" OH_NN_ReturnCode OH_NNExecutor_RunSync(OH_NNExecutor *, NN_Tensor *[], size_t, NN_Tensor *[], size_t)
{ return OH_NN_FAILED; }

int main()
{
    Reset(); RunLoad("OH_NNCompilation_Build"); assert(buildCalls == 1 && destroyed == 1 && rejectionIsError);
    Reset(); hasCannBackend = false; RunLoad("generic NNRT accelerators cannot load OMC"); assert(buildCalls == 0);
    Reset(); compatibility = 1; RunLoad("compatibility check rejected"); assert(buildCalls == 0);
    Reset(); buildMode = 1; RunLoad("injected vendor exception"); assert(destroyed == 1);
    Reset(); buildMode = 2; RunLoad("Unknown native exception"); assert(destroyed == 1);
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
    std::cout << "PASS: 11 native worker/completion fault-injection cases; no exceptions escaped callbacks\n";
}
