#include "register/op_def_registry.h"

namespace ops {
static ge::graphStatus InferMatmul(gert::InferShapeContext* context)
{
    const auto* a = context->GetInputShape(0);
    const auto* b = context->GetInputShape(1);
    auto* out = context->GetOutputShape(0);
    if (!a || !b || !out || a->GetDimNum() != 3 || b->GetDimNum() != 3
        || a->GetDim(0) != b->GetDim(0) || a->GetDim(2) != b->GetDim(1)) {
        return ge::GRAPH_FAILED;
    }
    *out = gert::Shape{a->GetDim(0), a->GetDim(1), b->GetDim(2)};
    return ge::GRAPH_SUCCESS;
}

class BatchMatMulV2 : public OpDef {
public:
    explicit BatchMatMulV2(const char* name) : OpDef(name)
    {
        Input("x1").ParamType(REQUIRED).DataType({ge::DT_FLOAT16}).Format({ge::FORMAT_ND});
        Input("x2").ParamType(REQUIRED).DataType({ge::DT_FLOAT16}).Format({ge::FORMAT_ND});
        Output("y").ParamType(REQUIRED).DataType({ge::DT_FLOAT16}).Format({ge::FORMAT_ND}).Follow("x1");
        Attr("adj_x1").AttrType(OPTIONAL).Bool(false);
        Attr("adj_x2").AttrType(OPTIONAL).Bool(false);
        SetInferShape(InferMatmul);
        AICore().AddConfig("kirin9030");
    }
};
OP_ADD(BatchMatMulV2);

class SoftmaxV2 : public OpDef {
public:
    explicit SoftmaxV2(const char* name) : OpDef(name)
    {
        Input("x").ParamType(REQUIRED).DataType({ge::DT_FLOAT16}).Format({ge::FORMAT_ND});
        Output("y").ParamType(REQUIRED).DataType({ge::DT_FLOAT16}).Format({ge::FORMAT_ND}).Follow("x");
        Attr("axes").AttrType(OPTIONAL).ListInt({-1});
        Attr("half_to_float").AttrType(OPTIONAL).Bool(false);
        AICore().AddConfig("kirin9030");
    }
};
OP_ADD(SoftmaxV2);
}
