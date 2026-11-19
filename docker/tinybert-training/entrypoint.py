"""Check the requested runtime before executing the training command."""
import json
import os
import sys

import torch


def main():
    requested = os.environ.get("TINYBERT_TRAINING_DEVICE", "cuda")
    if requested not in ("cuda", "cpu"):
        raise SystemExit("TINYBERT_TRAINING_DEVICE must be cuda or cpu")
    report = {"device": requested, "torch": str(torch.__version__), "compiled_cuda": torch.version.cuda}
    if requested == "cuda":
        try:
            if not torch.cuda.is_available():
                raise RuntimeError("torch.cuda.is_available() returned false")
            probe = torch.ones(4, device="cuda:0")
            if (probe @ probe).item() != 4.0:
                raise RuntimeError("CUDA tensor operation returned an incorrect result")
            report["gpus"] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
        except RuntimeError as error:
            raise SystemExit(
                f"CUDA preflight failed: {error}\n"
                "Use docker run --gpus all with NVIDIA Container Toolkit and a host driver compatible with CUDA 12.8. "
                "Set TINYBERT_TRAINING_DEVICE=cpu to opt into CPU mode."
            ) from None
    print(json.dumps(report, ensure_ascii=False), flush=True)

    command = sys.argv[1:]
    if command and command != ["--check"]:
        os.execvp(command[0], command)


if __name__ == "__main__":
    main()
