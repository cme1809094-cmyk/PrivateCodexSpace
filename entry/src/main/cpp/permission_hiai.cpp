#include <napi/native_api.h>
#include <neural_network_runtime/neural_network_core.h>
#include "nnrt_device.h"

#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {
constexpr size_t kSequenceLength = 64;
constexpr size_t kLabelCount = 5;
using permission_nnrt::DescribeDevice;
using permission_nnrt::ListDevices;
using permission_nnrt::SelectAccelerator;
using Tokens = std::array<int32_t, kSequenceLength>;
using Logits = std::array<float, kLabelCount>;

template <typename T, auto Destroy>
struct NNDeleter {
    void operator()(T *value) const { Destroy(&value); }
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
        throw std::runtime_error(std::string(operation) + " failed (" + std::to_string(code) + ")");
    }
}

void CheckJs(napi_status status, const char *operation)
{
    if (status != napi_ok) {
        throw std::runtime_error(operation);
    }
}

void ValidateTensor(const Descriptor &descriptor, OH_NN_DataType expectedType, size_t expectedElements)
{
    if (!descriptor) {
        throw std::runtime_error("NNRT returned no tensor descriptor");
    }
    OH_NN_DataType type;
    size_t elements = 0;
    Check(OH_NNTensorDesc_GetDataType(descriptor.get(), &type), "OH_NNTensorDesc_GetDataType");
    Check(OH_NNTensorDesc_GetElementCount(descriptor.get(), &elements), "OH_NNTensorDesc_GetElementCount");
    if (type != expectedType || elements != expectedElements) {
        throw std::runtime_error("Offline model tensor type or element count does not match MiniRBT");
    }
}

void *TensorBuffer(const Tensor &tensor, size_t expectedSize)
{
    size_t size = 0;
    Check(OH_NNTensor_GetSize(tensor.get(), &size), "OH_NNTensor_GetSize");
    void *buffer = OH_NNTensor_GetDataBuffer(tensor.get());
    if (!buffer || size < expectedSize) {
        throw std::runtime_error("NNRT tensor buffer is too small");
    }
    return buffer;
}

class Session {
public:
    explicit Session(const std::vector<uint8_t> &model) : device_(SelectAccelerator())
    {
        const size_t device = device_.id;
        const std::string backend = DescribeDevice(device_);
        Compilation compilation(OH_NNCompilation_ConstructWithOfflineModelBuffer(model.data(), model.size()));
        if (!compilation) {
            throw std::runtime_error("Cannot create NNRT compilation from the OM model");
        }
        Check(OH_NNCompilation_SetDevice(compilation.get(), device), "OH_NNCompilation_SetDevice(" + backend + ")");
        Check(OH_NNCompilation_Build(compilation.get()), "OH_NNCompilation_Build(" + backend + ")");
        executor_.reset(OH_NNExecutor_Construct(compilation.get()));
        if (!executor_) {
            throw std::runtime_error("OH_NNExecutor_Construct failed");
        }
        size_t inputCount = 0;
        size_t outputCount = 0;
        Check(OH_NNExecutor_GetInputCount(executor_.get(), &inputCount), "OH_NNExecutor_GetInputCount");
        Check(OH_NNExecutor_GetOutputCount(executor_.get(), &outputCount), "OH_NNExecutor_GetOutputCount");
        if (inputCount != inputs_.size() || outputCount != 1) {
            throw std::runtime_error("Offline model must expose two inputs and one output");
        }
        bool foundIds = false;
        bool foundMask = false;
        for (size_t index = 0; index < inputCount; ++index) {
            Descriptor descriptor(OH_NNExecutor_CreateInputTensorDesc(executor_.get(), index));
            if (!descriptor) {
                throw std::runtime_error("NNRT returned no input tensor descriptor");
            }
            int32_t *shape = nullptr;
            size_t rank = 0;
            Check(OH_NNTensorDesc_GetShape(descriptor.get(), &shape, &rank), "OH_NNTensorDesc_GetShape");
            if (!shape || rank != 2 || shape[0] != 1 || (shape[1] != 16 && shape[1] != 32 && shape[1] != 64)) {
                throw std::runtime_error("Expected NNRT token shape [1,16], [1,32] or [1,64]");
            }
            const size_t length = static_cast<size_t>(shape[1]);
            if (sequenceLength_ && sequenceLength_ != length) {
                throw std::runtime_error("NNRT token and mask shapes differ");
            }
            sequenceLength_ = length;
            ValidateTensor(descriptor, OH_NN_INT32, sequenceLength_);
            const char *name = nullptr;
            Check(OH_NNTensorDesc_GetName(descriptor.get(), &name), "OH_NNTensorDesc_GetName");
            const std::string inputName = name ? name : "";
            if (inputName == "input_ids" && !foundIds) {
                idsIndex_ = index;
                foundIds = true;
            } else if (inputName == "attention_mask" && !foundMask) {
                foundMask = true;
            } else {
                throw std::runtime_error("Unexpected OM input name: " + inputName);
            }
            inputs_[index].reset(OH_NNTensor_Create(device, descriptor.get()));
            if (!inputs_[index]) {
                throw std::runtime_error("OH_NNTensor_Create input failed");
            }
        }
        Descriptor descriptor(OH_NNExecutor_CreateOutputTensorDesc(executor_.get(), 0));
        ValidateTensor(descriptor, OH_NN_FLOAT32, kLabelCount);
        output_.reset(OH_NNTensor_Create(device, descriptor.get()));
        if (!output_) {
            throw std::runtime_error("OH_NNTensor_Create output failed");
        }
    }

    Logits Predict(const Tokens &ids, const Tokens &mask)
    {
        std::lock_guard<std::mutex> lock(mutex_);
        const size_t bytes = sequenceLength_ * sizeof(int32_t);
        std::memcpy(TensorBuffer(inputs_[idsIndex_], bytes), ids.data(), bytes);
        std::memcpy(TensorBuffer(inputs_[1 - idsIndex_], bytes), mask.data(), bytes);
        NN_Tensor *inputTensors[] = {inputs_[0].get(), inputs_[1].get()};
        NN_Tensor *outputTensors[] = {output_.get()};
        Check(OH_NNExecutor_RunSync(executor_.get(), inputTensors, 2, outputTensors, 1),
            "OH_NNExecutor_RunSync(" + DescribeDevice(device_) + ")");
        Logits logits{};
        std::memcpy(logits.data(), TensorBuffer(output_, sizeof(logits)), sizeof(logits));
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
    explicit Job(Operation requested) : operation(requested) {}
    Operation operation;
    napi_async_work work = nullptr;
    napi_deferred deferred = nullptr;
    std::shared_ptr<Session> session;
    std::vector<uint8_t> model;
    Tokens ids{};
    Tokens mask{};
    Logits logits{};
    std::string error;
};

napi_value JsString(napi_env env, const std::string &text)
{
    napi_value result;
    CheckJs(napi_create_string_utf8(env, text.c_str(), text.size(), &result), "Cannot allocate string");
    return result;
}

napi_value JsError(napi_env env, const std::string &message)
{
    napi_value result;
    CheckJs(napi_create_error(env, nullptr, JsString(env, message), &result), "Cannot allocate error");
    return result;
}

void Execute(napi_env, void *data)
{
    auto &job = *static_cast<Job *>(data);
    try {
        if (job.operation == Operation::Load) {
            job.session = std::make_shared<Session>(job.model);
        } else {
            job.logits = job.session->Predict(job.ids, job.mask);
        }
    } catch (const std::exception &error) {
        job.error = error.what();
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

void Complete(napi_env env, napi_status status, void *data)
{
    std::unique_ptr<Job> job(static_cast<Job *>(data));
    try {
        if (status != napi_ok) {
            throw std::runtime_error("HiAI async operation cancelled");
        }
        if (!job->error.empty()) {
            throw std::runtime_error(job->error);
        }
        napi_resolve_deferred(env, job->deferred, JobResult(env, *job));
    } catch (const std::exception &error) {
        napi_reject_deferred(env, job->deferred, JsError(env, error.what()));
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

napi_value Devices(napi_env env, napi_callback_info)
{
    try {
        auto devices = ListDevices();
        napi_value result;
        CheckJs(napi_create_array_with_length(env, devices.size(), &result), "Cannot allocate device list");
        for (uint32_t i = 0; i < devices.size(); ++i) {
            CheckJs(napi_set_element(env, result, i, JsString(env,
                DescribeDevice(devices[i]))), "Cannot set device");
        }
        return result;
    } catch (const std::exception &error) {
        napi_throw_error(env, nullptr, error.what());
        return nullptr;
    }
}

napi_value Accelerator(napi_env env, napi_callback_info)
{
    try {
        return JsString(env, DescribeDevice(SelectAccelerator()));
    } catch (const std::exception &error) {
        napi_throw_error(env, nullptr, error.what());
        return nullptr;
    }
}

napi_value Load(napi_env env, napi_callback_info info)
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
    }
}

napi_value Predict(napi_env env, napi_callback_info info)
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
