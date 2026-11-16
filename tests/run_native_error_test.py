"""Compile and run host-only native callback fault injection with real SDK headers."""
import argparse
import os
from pathlib import Path
import subprocess
import tempfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sdk-include", type=Path, required=True)
    parser.add_argument("--compiler", default="g++")
    args = parser.parse_args()
    tests = Path(__file__).resolve().parent
    with tempfile.TemporaryDirectory(prefix="permission-native-test-") as directory:
        work = Path(directory)
        subprocess.run([args.compiler, "-x", "c++", "-std=c++17", "-shared", "-fPIC",
                        str(tests / "fake_cann_runtime.cpp"), "-o", str(work / "libhiai_foundation.so")], check=True)
        executable = work / "native-error-test"
        subprocess.run([args.compiler, "-x", "c++", "-std=c++17", "-Wall", "-Wextra", "-fexceptions",
                        "-funwind-tables", "-pthread", "-rdynamic", "-I" + str(args.sdk_include.resolve()),
                        str(tests / "native_error_test.cpp"), "-lstdc++", "-ldl", "-o", str(executable)], check=True)
        env = os.environ.copy()
        env["LD_LIBRARY_PATH"] = str(work) + (":" + env["LD_LIBRARY_PATH"] if env.get("LD_LIBRARY_PATH") else "")
        subprocess.run([str(executable)], env=env, check=True)


if __name__ == "__main__":
    main()
