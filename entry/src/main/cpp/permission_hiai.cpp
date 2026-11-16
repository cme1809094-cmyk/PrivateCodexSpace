#include <napi/native_api.h>
#include <neural_network_runtime/neural_network_core.h>
#include "nnrt_device.h"
#include "cann_runtime.h"
#include "native_diagnostics.h"

#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#if !defined(__cpp_exceptions) && !defined(__EXCEPTIONS)
#error "permission_hiai requires C++ exception handling; remove exception-disabling compiler options"
#endif

namespace {
constexpr size_t kSequenceLength = 64;
constexpr size_t kLabelCount = 5;
using permission_nnrt::DescribeDevice;
using permission_nnrt::ListDevices;
using permission_nnrt::SelectOfflineAccelerator;
using permission_nnrt::NativeTrace;
using Tokens = std::array<int32_t, kSequenceLength>;
using Logits = std::array<float, kLabelCount>;

template <typename T, auto Destroy>
struct NNDeleter {
    void operator()(T *value) const noexcept
    {
        try {
            Destroy(&value);
        } catch (const std::exception &error) {
            PermissionLog(error.what());
        } catch (...) {
            PermissionLog("Unknown exception during NNRT cleanup");
        }
    }
};

template <typename T, auto Destroy>
using NNPointer = std::unique_ptr<T, NNDeleter<T, Destroy>>;

using Compilation = NNPointer<OH_NNCompilation, OH_NNCompilation_Destroy>;
using Executor = NNPointer<OH_NNExecutor, OH_NNExecutor_Destroy>;
using Descriptor = NNPointer<NN_TensorDesc, OH_NNTensorDesc_Destroy>;
using Tensor = NNPointer<NN_Tensor, OH_NNTensor_Destroy>;

void Check(OH_NN_ReturnCode code, const std::string &operation)
{
    if (code != OH_NN_SUCCESS) {
        const auto message = operation + " failed (NNRT code " + std::to_string(code) + ")";
        PermissionLog(message.c_str());
        throw std::runtime_error(message);
    }
}

permission_nnrt::Device SelectModelDevice()
{
    const auto &runtime = permission_nnrt::GetCannRuntime();
    if (!runtime.Available()) throw std::runtime_error(runtime.Error());
    return SelectOfflineAccelerator();
}

void CheckJs(napi_status status, const char *operation)
{
    if (status != napi_ok) {
        throw std::runtime_error(operation);
    }
}

void ValidateTensor(const Descriptor &descriptor, OH_NN_DataType expectedType, size_t expectedElements,
    NativeTrace &trace, int index)
{
    if (!descriptor) {
        throw std::runtime_error("NNRT returned no tensor descriptor");
    }
    OH_NN_DataType type;
    size_t elements = 0;
    Check(trace.Call("OH_NNTensorDesc_GetDataType", [&] {
        return OH_NNTensorDesc_GetDataType(descriptor.get(), &type);
    }, index), "OH_NNTensorDesc_GetDataType");
    Check(trace.Call("OH_NNTensorDesc_GetElementCount", [&] {
        return OH_NNTensorDesc_GetElementCount(descriptor.get(), &elements);
    }, index), "OH_NNTensorDesc_GetElementCount");
    if (type != expectedType || elements != expectedElements) {
        throw std::runtime_error("Offline model tensor mismatch: type=" + std::to_string(type)
            + ", elements=" + std::to_string(elements) + "; expected type=" + std::to_string(expectedType)
            + ", elements=" + std::to_string(expectedElements));
    }
}

void *TensorBuffer(const Tensor &tensor, size_t expectedSize, NativeTrace &trace, int index)
{
    size_t size = 0;
    Check(trace.Call("OH_NNTensor_GetSize", [&] { return OH_NNTensor_GetSize(tensor.get(), &size); }, index),
        "OH_NNTensor_GetSize");
    void *buffer = trace.Call("OH_NNTensor_GetDataBuffer", [&] {
        return OH_NNTensor_GetDataBuffer(tensor.get());
    }, index);
    if (!buffer || size < expectedSize) {
        throw std::runtime_error("NNRT tensor buffer is too small");
    }
    return buffer;
}

class Session {
public:
    explicit Session(const std::vector<uint8_t> &model, NativeTrace &trace)
    {
        const auto *runtime = trace.Call("Initialize CANN Foundation", [] {
            return &permission_nnrt::GetCannRuntime();
        });
        trace.SetRuntimeVersion(runtime->Version());
        device_ = trace.Call("Select CANN HIAI_F backend", [] { return SelectModelDevice(); });
        const size_t device = device_.id;
        const std::string backend = DescribeDevice(device_);
        PermissionLog(("Load OMC bytes=" + std::to_string(model.size()) + " on " + backend).c_str());
        if (runtime->HasCompatibilityCheck()) {
            const int compatibility = trace.Call("HMS_HiAICompatibility_CheckFromBuffer", [&] {
                return runtime->CheckCompatibility(model.data(), model.size());
            });
            if (compatibility != 0) {
                throw std::runtime_error("CANN compatibility check rejected the offline model on " + backend
                    + " (result " + std::to_string(compatibility) + ")");
            }
        }
        Compilation compilation(trace.Call("OH_NNCompilation_ConstructWithOfflineModelBuffer", [&] {
            return OH_NNCompilation_ConstructWithOfflineModelBuffer(model.data(), model.size());
        }));
        if (!compilation) {
            throw std::runtime_error("Cannot create NNRT compilation from the OM model");
        }
        Check(trace.Call("OH_NNCompilation_SetDevice", [&] {
            return OH_NNCompilation_SetDevice(compilation.get(), device);
        }), "OH_NNCompilation_SetDevice(" + backend + ")");
        Check(trace.Call("OH_NNCompilation_Build", [&] {
            return OH_NNCompilation_Build(compilation.get());
        }), "OH_NNCompilation_Build(" + backend + ")");
        executor_.reset(trace.Call("OH_NNExecutor_Construct", [&] { return OH_NNExecutor_Construct(compilation.get()); }));
        if (!executor_) {
            throw std::runtime_error("OH_NNExecutor_Construct failed");
        }
        size_t inputCount = 0;
        size_t outputCount = 0;
        Check(trace.Call("OH_NNExecutor_GetInputCount", [&] {
            return OH_NNExecutor_GetInputCount(executor_.get(), &inputCount);
        }), "OH_NNExecutor_GetInputCount");
        Check(trace.Call("OH_NNExecutor_GetOutputCount", [&] {
            return OH_NNExecutor_GetOutputCount(executor_.get(), &outputCount);
        }), "OH_NNExecutor_GetOutputCount");
        if (inputCount != inputs_.size() || outputCount != 1) {
            throw std::runtime_error("Offline model must expose two inputs and one output");
        }
        bool foundIds = false;
        bool foundMask = false;
        for (size_t index = 0; index < inputCount; ++index) {
            Descriptor descriptor(trace.Call("OH_NNExecutor_CreateInputTensorDesc", [&] {
                return OH_NNExecutor_CreateInputTensorDesc(executor_.get(), index);
            }, static_cast<int>(index)));
            if (!descriptor) {
                throw std::runtime_error("NNRT returned no input tensor descriptor");
            }
            int32_t *shape = nullptr;
            size_t rank = 0;
            Check(trace.Call("OH_NNTensorDesc_GetShape", [&] {
                return OH_NNTensorDesc_GetShape(descriptor.get(), &shape, &rank);
            }, static_cast<int>(index)), "OH_NNTensorDesc_GetShape");
            if (!shape || rank != 2 || shape[0] != 1 || (shape[1] != 16 && shape[1] != 32 && shape[1] != 64)) {
                throw std::runtime_error("Expected NNRT token shape [1,16], [1,32] or [1,64]");
            }
            const size_t length = static_cast<size_t>(shape[1]);
            if (sequenceLength_ && sequenceLength_ != length) {
                throw std::runtime_error("NNRT token and mask shapes differ");
            }
            sequenceLength_ = length;
            ValidateTensor(descriptor, OH_NN_INT32, sequenceLength_, trace, static_cast<int>(index));
            const char *name = nullptr;
            Check(trace.Call("OH_NNTensorDesc_GetName", [&] {
                return OH_NNTensorDesc_GetName(descriptor.get(), &name);
            }, static_cast<int>(index)), "OH_NNTensorDesc_GetName");
            const std::string inputName = name ? name : "";
            if (inputName == "input_ids" && !foundIds) {
                idsIndex_ = index;
                foundIds = true;
            } else if (inputName == "attention_mask" && !foundMask) {
                foundMask = true;
            } else {
                throw std::runtime_error("Unexpected OM input name: " + inputName);
            }
            inputs_[index].reset(trace.Call("OH_NNTensor_Create input", [&] {
                return OH_NNTensor_Create(device, descriptor.get());
            }, static_cast<int>(index)));
            if (!inputs_[index]) {
                throw std::runtime_error("OH_NNTensor_Create input failed");
            }
        }
        Descriptor descriptor(trace.Call("OH_NNExecutor_CreateOutputTensorDesc", [&] {
            return OH_NNExecutor_CreateOutputTensorDesc(executor_.get(), 0);
        }));
        ValidateTensor(descriptor, OH_NN_FLOAT32, kLabelCount, trace, 0);
        output_.reset(trace.Call("OH_NNTensor_Create output", [&] { return OH_NNTensor_Create(device, descriptor.get()); }));
        if (!output_) {
            throw std::runtime_error("OH_NNTensor_Create output failed");
        }
    }

    Logits Predict(const Tokens &ids, const Tokens &mask, NativeTrace &trace)
    {
        std::lock_guard<std::mutex> lock(mutex_);
        trace.SetRuntimeVersion(permission_nnrt::GetCannRuntime().Version());
        const size_t bytes = sequenceLength_ * sizeof(int32_t);
        std::memcpy(TensorBuffer(inputs_[idsIndex_], bytes, trace, static_cast<int>(idsIndex_)), ids.data(), bytes);
        std::memcpy(TensorBuffer(inputs_[1 - idsIndex_], bytes, trace, static_cast<int>(1 - idsIndex_)), mask.data(), bytes);
        NN_Tensor *inputTensors[] = {inputs_[0].get(), inputs_[1].get()};
        NN_Tensor *outputTensors[] = {output_.get()};
        Check(trace.Call("OH_NNExecutor_RunSync", [&] {
            return OH_NNExecutor_RunSync(executor_.get(), inputTensors, 2, outputTensors, 1);
        }),
            "OH_NNExecutor_RunSync(" + DescribeDevice(device_) + ")");
        Logits logits{};
        std::memcpy(logits.data(), TensorBuffer(output_, sizeof(logits), trace, 0), sizeof(logits));
        for (const float value : logits) {
            if (!std::isfinite(value)) {
                throw std::runtime_error("HiAI returned non-finite logits");
            }
        }
        return logits;
    }

    std::string DeviceName() const { return DescribeDevice(device_); }

private:
    permission_nnrt::Device device_;
    Executor executor_;
    std::array<Tensor, 2> inputs_;
    Tensor output_;
    size_t idsIndex_ = 0;
    size_t sequenceLength_ = 0;
    std::mutex mutex_;
};

enum class Operation { Load, Predict };

struct Job {
    explicit Job(Operation requested) : operation(requested), trace(requested == Operation::Load) {}
    Operation operation;
    napi_async_work work = nullptr;
    napi_deferred deferred = nullptr;
    std::shared_ptr<Session> session;
    std::vector<uint8_t> model;
    Tokens ids{};
    Tokens mask{};
    Logits logits{};
    std::array<char, 1024> error{};
    NativeTrace trace;
};

napi_value JsString(napi_env env, const std::string &text)
{
    napi_value result;
    CheckJs(napi_create_string_utf8(env, text.c_str(), text.size(), &result), "Cannot allocate string");
    return result;
}

void RejectDeferred(napi_env env, napi_deferred deferred, const char *message) noexcept
{
    // This function is used from catch handlers. Never call CheckJs here:
    // a second C++ exception while converting the first one aborts FFRT.
    PermissionLog(message);
    napi_value reason = nullptr;
    bool pending = false;
    if (napi_is_exception_pending(env, &pending) == napi_ok && pending) {
        (void)napi_get_and_clear_last_exception(env, &reason);
    }
    if (!reason) {
        napi_value text = nullptr;
        if (napi_create_string_utf8(env, message, NAPI_AUTO_LENGTH, &text) == napi_ok) {
            if (napi_create_error(env, nullptr, text, &reason) != napi_ok) reason = text;
        }
    }
    if (!reason) (void)napi_get_undefined(env, &reason);
    pending = false;
    if (napi_is_exception_pending(env, &pending) == napi_ok && pending) {
        napi_value pendingReason = nullptr;
        if (napi_get_and_clear_last_exception(env, &pendingReason) == napi_ok && pendingReason) reason = pendingReason;
    }
    if (!reason) {
        PermissionLog("Cannot allocate a JS rejection value; original failure logged above");
        return;
    }
    if (napi_reject_deferred(env, deferred, reason) != napi_ok) {
        PermissionLog("Cannot reject native promise; original failure logged above");
    }
}

void Execute(napi_env, void *data) noexcept
{
    auto &job = *static_cast<Job *>(data);
    const char *operation = job.operation == Operation::Load ? "Load OMC" : "Predict";
    try {
        if (job.operation == Operation::Load) {
            job.session = std::make_shared<Session>(job.model, job.trace);
        } else {
            job.logits = job.session->Predict(job.ids, job.mask, job.trace);
        }
    } catch (const std::exception &error) {
        job.trace.Failure(job.error.data(), job.error.size(), operation, error.what());
        PermissionLog(job.error.data());
    } catch (const std::string &error) {
        job.trace.Failure(job.error.data(), job.error.size(), operation, error.c_str());
        PermissionLog(job.error.data());
    } catch (const char *error) {
        job.trace.Failure(job.error.data(), job.error.size(), operation, error);
        PermissionLog(job.error.data());
    } catch (int error) {
        char detail[64];
        std::snprintf(detail, sizeof(detail), "Native exception int=%d", error);
        job.trace.Failure(job.error.data(), job.error.size(), operation, detail);
        PermissionLog(job.error.data());
    } catch (...) {
        job.trace.Failure(job.error.data(), job.error.size(), operation, "Unknown native exception");
        PermissionLog(job.error.data());
    }
}

napi_value JobResult(napi_env env, const Job &job)
{
    napi_value result;
    if (job.operation == Operation::Load) {
        CheckJs(napi_create_object(env, &result), "Cannot allocate session object");
        auto holder = std::make_unique<std::shared_ptr<Session>>(job.session);
        CheckJs(napi_wrap(env, result, holder.get(),
            [](napi_env, void *data, void *) { delete static_cast<std::shared_ptr<Session> *>(data); },
            nullptr, nullptr), "Cannot wrap session");
        holder.release();
        CheckJs(napi_set_named_property(env, result, "device", JsString(env, job.session->DeviceName())), "Cannot set device");
    } else {
        CheckJs(napi_create_array_with_length(env, job.logits.size(), &result), "Cannot allocate logits array");
        for (uint32_t i = 0; i < job.logits.size(); ++i) {
            napi_value value;
            CheckJs(napi_create_double(env, job.logits[i], &value), "Cannot allocate logit");
            CheckJs(napi_set_element(env, result, i, value), "Cannot set logit");
        }
    }
    return result;
}

void Complete(napi_env env, napi_status status, void *data) noexcept
{
    std::unique_ptr<Job> job(static_cast<Job *>(data));
    try {
        if (status != napi_ok) {
            RejectDeferred(env, job->deferred, "HiAI async operation cancelled");
        } else if (job->error[0] != '\0') {
            RejectDeferred(env, job->deferred, job->error.data());
        } else {
            CheckJs(napi_resolve_deferred(env, job->deferred, JobResult(env, *job)), "Cannot resolve native promise");
        }
    } catch (const std::exception &error) {
        RejectDeferred(env, job->deferred, error.what());
    } catch (...) {
        RejectDeferred(env, job->deferred, "Unknown native exception while completing NNRT work");
    }
    napi_delete_async_work(env, job->work);
}

napi_value Queue(napi_env env, std::unique_ptr<Job> job)
{
    napi_value promise;
    CheckJs(napi_create_promise(env, &job->deferred, &promise), "Cannot allocate HiAI promise");
    CheckJs(napi_create_async_work(env, nullptr, JsString(env, "PermissionHiAI"), Execute, Complete,
        job.get(), &job->work), "Cannot create HiAI work");
    if (napi_queue_async_work(env, job->work) != napi_ok) {
        napi_delete_async_work(env, job->work);
        throw std::runtime_error("Cannot queue HiAI work");
    }
    job.release();
    return promise;
}

Tokens ReadTokens(napi_env env, napi_value value)
{
    napi_typedarray_type type;
    size_t length = 0;
    void *data = nullptr;
    CheckJs(napi_get_typedarray_info(env, value, &type, &length, &data, nullptr, nullptr),
        "Expected an Int32Array");
    if (type != napi_int32_array || length != kSequenceLength || !data) {
        throw std::runtime_error("Expected an Int32Array of length 64");
    }
    Tokens tokens{};
    std::memcpy(tokens.data(), data, sizeof(tokens));
    return tokens;
}

napi_value Devices(napi_env env, napi_callback_info) noexcept
{
    try {
        PermissionLog("PermissionHiAI native bridge 1.0.6: enumerate backends");
        (void)permission_nnrt::GetCannRuntime();
        auto devices = ListDevices();
        napi_value result;
        CheckJs(napi_create_array_with_length(env, devices.size(), &result), "Cannot allocate device list");
        for (uint32_t i = 0; i < devices.size(); ++i) {
            PermissionLog(("NNRT device " + DescribeDevice(devices[i])).c_str());
            CheckJs(napi_set_element(env, result, i, JsString(env,
                DescribeDevice(devices[i]))), "Cannot set device");
        }
        return result;
    } catch (const std::exception &error) {
        napi_throw_error(env, nullptr, error.what());
        return nullptr;
    } catch (...) {
        napi_throw_error(env, nullptr, "Unknown native exception during device enumeration");
        return nullptr;
    }
}

napi_value Accelerator(napi_env env, napi_callback_info) noexcept
{
    try {
        const auto device = SelectModelDevice();
        return JsString(env, DescribeDevice(device) + " · CANN " + permission_nnrt::GetCannRuntime().Version());
    } catch (const std::exception &error) {
        napi_throw_error(env, nullptr, error.what());
        return nullptr;
    } catch (...) {
        napi_throw_error(env, nullptr, "Unknown native exception during CANN backend selection");
        return nullptr;
    }
}

napi_value Load(napi_env env, napi_callback_info info) noexcept
{
    try {
        size_t argc = 1;
        napi_value value;
        CheckJs(napi_get_cb_info(env, info, &argc, &value, nullptr, nullptr), "Cannot read OM argument");
        if (argc != 1) {
            throw std::runtime_error("Expected an OM ArrayBuffer");
        }
        size_t size = 0;
        void *data = nullptr;
        CheckJs(napi_get_arraybuffer_info(env, value, &data, &size), "Expected an OM ArrayBuffer");
        if (size == 0 || !data) {
            throw std::runtime_error("The OM model is empty");
        }
        auto job = std::make_unique<Job>(Operation::Load);
        const auto *bytes = static_cast<const uint8_t *>(data);
        job->model.assign(bytes, bytes + size);
        return Queue(env, std::move(job));
    } catch (const std::exception &error) {
        napi_throw_error(env, nullptr, error.what());
        return nullptr;
    } catch (...) {
        napi_throw_error(env, nullptr, "Unknown native exception while scheduling model load");
        return nullptr;
    }
}

napi_value Predict(napi_env env, napi_callback_info info) noexcept
{
    try {
        size_t argc = 3;
        napi_value values[3];
        CheckJs(napi_get_cb_info(env, info, &argc, values, nullptr, nullptr), "Cannot read inference arguments");
        if (argc != 3) {
            throw std::runtime_error("Expected HiAI session, input_ids and attention_mask");
        }
        void *holder = nullptr;
        CheckJs(napi_unwrap(env, values[0], &holder), "Invalid HiAI session");
        if (!holder) {
            throw std::runtime_error("Invalid HiAI session");
        }
        auto job = std::make_unique<Job>(Operation::Predict);
        job->session = *static_cast<std::shared_ptr<Session> *>(holder);
        job->ids = ReadTokens(env, values[1]);
        job->mask = ReadTokens(env, values[2]);
        return Queue(env, std::move(job));
    } catch (const std::exception &error) {
        napi_throw_error(env, nullptr, error.what());
        return nullptr;
    } catch (...) {
        napi_throw_error(env, nullptr, "Unknown native exception while scheduling inference");
        return nullptr;
    }
}

napi_value Init(napi_env env, napi_value exports)
{
    napi_property_descriptor properties[] = {
        {"devices", nullptr, Devices, nullptr, nullptr, nullptr, napi_default, nullptr},
        {"accelerator", nullptr, Accelerator, nullptr, nullptr, nullptr, napi_default, nullptr},
        {"load", nullptr, Load, nullptr, nullptr, nullptr, napi_default, nullptr},
        {"predict", nullptr, Predict, nullptr, nullptr, nullptr, napi_default, nullptr},
    };
    if (napi_define_properties(env, exports, sizeof(properties) / sizeof(properties[0]), properties) != napi_ok) {
        return nullptr;
    }
    return exports;
}
} // namespace

static napi_module module = {1, 0, nullptr, Init, "permission_hiai", nullptr, {0}};

extern "C" __attribute__((constructor)) void RegisterPermissionHiAI()
{
    napi_module_register(&module);
}
