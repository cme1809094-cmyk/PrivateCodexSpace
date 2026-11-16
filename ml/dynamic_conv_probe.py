"""Try grouped convolution with runtime weights for the two attention products."""
from pathlib import Path
import numpy as np
import onnx
from onnx import TensorProto as T, helper as h, numpy_helper as nh


def export(path, rows, inner, cols):
    graph = h.make_graph([], "dynamic_attention", [
        h.make_tensor_value_info("a", T.FLOAT, [1, 8, rows, inner]),
        h.make_tensor_value_info("b", T.FLOAT, [1, 8, inner, cols]),
    ], [h.make_tensor_value_info("y", T.FLOAT, [1, 8, rows, cols])])

    def node(kind, inputs, output, **attrs):
        graph.node.append(h.make_node(kind, inputs, [output], name=output, **attrs))

    def reshape(x, shape, y):
        dims = y + "_shape"
        graph.initializer.append(nh.from_array(np.asarray(shape, np.int64), dims))
        node("Reshape", [x, dims], y)

    node("Transpose", ["a"], "a_permute", perm=[0, 1, 3, 2])
    reshape("a_permute", [1, 8 * inner, 1, rows], "x")
    node("Transpose", ["b"], "b_permute", perm=[0, 1, 3, 2])
    reshape("b_permute", [8 * cols, inner, 1, 1], "weights")
    node("Conv", ["x", "weights"], "conv", kernel_shape=[1, 1], group=8)
    reshape("conv", [1, 8, cols, rows], "packed")
    node("Transpose", ["packed"], "y", perm=[0, 1, 3, 2])
    model = h.make_model(graph, opset_imports=[h.make_opsetid("", 13)], ir_version=8)
    onnx.checker.check_model(model, full_check=True)
    onnx.save(model, path)


root = Path(__file__).parent
export(root / "dynamic_qk.onnx", 64, 32, 64)
export(root / "dynamic_pv.onnx", 64, 64, 32)
