#include "register/register.h"
#include "graph/attr_value.h"

namespace domi {
static Status ParseMatmul(const ge::Operator&, ge::Operator& op)
{
    op.SetAttr(std::string("adj_x1"), false);
    op.SetAttr(std::string("adj_x2"), false);
    return SUCCESS;
}

static Status ParseSoftmax(const ge::Operator&, ge::Operator& op)
{
    op.SetAttr(std::string("half_to_float"), false);
    return SUCCESS;
}

REGISTER_CUSTOM_OP("BatchMatMulV2")
    .FrameworkType(ONNX)
    .OriginOpType({"NativeAttentionMatmul", "ai.onnx::13::NativeAttentionMatmul", "ai.onnx::11::NativeAttentionMatmul"})
    .ParseParamsByOperatorFn(ParseMatmul);

REGISTER_CUSTOM_OP("SoftmaxV2")
    .FrameworkType(ONNX)
    .OriginOpType({"NativeAttentionSoftmax", "ai.onnx::13::NativeAttentionSoftmax", "ai.onnx::11::NativeAttentionSoftmax"})
    .ParseParamsByOperatorFn(ParseSoftmax);
}
