#include "register/op_def_registry.h"
#include "register/op_impl_kernel_registry.h"
#include "exe_graph/runtime/tiling_context.h"
#include "platform/platform_infos_def.h"
#include <charconv>
#include <cstdio>
#include <string>

// Host ABI from the matching Kirin plugin. Used only by the tiling diagnostic.
namespace gert {
class OpImplRegistry {
public:
    static OpImplRegistry& GetInstance();
    const OpImplKernelRegistry::OpImplFunctions* GetOpImpl(const char* type) const;
};
}

namespace ops {
static gert::OpImplRegisterV2::TilingKernelFunc VendorTiling(const char* type)
{
    auto* impl = gert::OpImplRegistry::GetInstance().GetOpImpl(type);
    return impl == nullptr ? nullptr : impl->tiling;
}

static gert::OpImplRegisterV2::TilingKernelFunc softmax_tiling = nullptr;

static ge::graphStatus RunSoftmaxTiling(gert::TilingContext* context)
{
    if (context == nullptr || softmax_tiling == nullptr) {
        return ge::GRAPH_FAILED;
    }
    // The mobile Softmax host tiler uses the incoming blockDim as its core
    // budget. The standalone tool starts it at zero; seed that budget using
    // the loaded platform resources before calling the unmodified vendor tiler.
    // Preserve a budget that the compiler or caller has already supplied.
    if (context->GetBlockDim() == 0U) {
        auto* platform = context->GetPlatformInfo();
        std::string cores;
        if (platform == nullptr ||
            !platform->GetPlatformRes("SoCInfo", "vector_core_cnt", cores)) {
            return ge::GRAPH_FAILED;
        }
        uint32_t budget = 0;
        const auto result = std::from_chars(cores.data(), cores.data() + cores.size(), budget);
        if (result.ec != std::errc{} || result.ptr != cores.data() + cores.size() ||
            budget == 0U || context->SetBlockDim(budget) != ge::GRAPH_SUCCESS) {
            return ge::GRAPH_FAILED;
        }
        std::fprintf(stderr,
            "SoftmaxV2 standalone context: blockDim initialized from platform vector_core_cnt=%u\n",
            budget);
    }
    return softmax_tiling(context);
}

class BatchMatMulV2 : public OpDef {
public:
    explicit BatchMatMulV2(const char* name) : OpDef(name)
    {
        Input("x1").ParamType(REQUIRED).DataType({ge::DT_FLOAT16}).Format({ge::FORMAT_ND});
        Input("x2").ParamType(REQUIRED).DataType({ge::DT_FLOAT16}).Format({ge::FORMAT_ND});
        Input("bias").ParamType(OPTIONAL).DataType({ge::DT_FLOAT16}).Format({ge::FORMAT_ND});
        Input("offset_w").ParamType(OPTIONAL).DataType({ge::DT_INT32}).Format({ge::FORMAT_ND});
        Output("y").ParamType(REQUIRED).DataType({ge::DT_FLOAT16}).Format({ge::FORMAT_ND});
        Attr("adj_x1").AttrType(OPTIONAL).Bool(false);
        Attr("adj_x2").AttrType(OPTIONAL).Bool(false);
        AICore().SetTiling(VendorTiling(name));
        AICore().AddConfig("kirin9030");
    }
};
OP_ADD(BatchMatMulV2);

class SoftmaxV2 : public OpDef {
public:
    explicit SoftmaxV2(const char* name) : OpDef(name)
    {
        Input("x").ParamType(REQUIRED).DataType({ge::DT_FLOAT16}).Format({ge::FORMAT_ND});
        Output("y").ParamType(REQUIRED).DataType({ge::DT_FLOAT16}).Format({ge::FORMAT_ND});
        Attr("axes").AttrType(OPTIONAL).ListInt({-1});
        Attr("half_to_float").AttrType(OPTIONAL).Bool(false);
        softmax_tiling = VendorTiling(name);
        AICore().SetTiling(RunSoftmaxTiling);
        AICore().AddConfig("kirin9030");
    }
};
OP_ADD(SoftmaxV2);
}
