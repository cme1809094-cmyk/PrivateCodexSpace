"""Read static BERT encoder dimensions from the exported ONNX graph."""
import argparse
import json
from pathlib import Path
import re

import onnx


def inspect_model(model):
    layers = sorted({int(m[1]) for n in model.graph.node
                     if (m := re.search(r"/encoder/layer\.(\d+)/", n.name))})
    if not layers or layers != list(range(len(layers))):
        raise ValueError(f"Expected contiguous BERT encoder layers, got {layers}")
    embeddings = [w for w in model.graph.initializer if w.name.endswith("word_embeddings.weight")]
    if len(embeddings) != 1 or len(embeddings[0].dims) != 2:
        raise ValueError("Expected one BERT word embedding matrix")
    hidden = embeddings[0].dims[1]
    inferred = onnx.shape_inference.infer_shapes(model)
    shapes = {v.name: [d.dim_value for d in v.type.tensor_type.shape.dim]
              for v in (*inferred.graph.input, *inferred.graph.value_info, *inferred.graph.output)}
    qk = [n for n in inferred.graph.node if n.name.endswith("/attention/self/MatMul")]
    if len(qk) != len(layers):
        raise ValueError("Expected one QK MatMul per encoder layer")
    dims = shapes[qk[0].input[0]]
    if len(dims) != 4 or dims[0] != 1 or min(dims) <= 0 or dims[1] * dims[3] != hidden:
        raise ValueError(f"Unexpected attention input shape {dims}")
    return {"num_hidden_layers": len(layers), "hidden_size": hidden,
            "num_attention_heads": dims[1], "head_size": dims[3],
            "input_length": dims[2], "vocab_size": embeddings[0].dims[0]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    args = parser.parse_args()
    print(json.dumps(inspect_model(onnx.load(args.model)), indent=2))
