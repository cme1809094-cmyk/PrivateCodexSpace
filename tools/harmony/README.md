# HarmonyOS 云端构建

本项目在 Linux x64 的 Codex 云端使用以下固定版本完成1.1.1干净构建，包含 ArkTS、原生 C++、资源和未签名 HAP 打包：

| 组件 | 版本 |
| --- | --- |
| Command Line Tools | 5.0.3.906 |
| HarmonyOS SDK | 5.0.0.71 / API 12 Release |
| Hvigor / HarmonyOS plugin | 5.8.9 |
| OHPM | 5.0.8 |
| SDK 自带 Node | 18.20.1 |
| JDK | Temurin 17.0.21+9 |

下载地址和 SHA-256 固定在 toolchain.json，安装器会校验完整性。SDK 和缓存放在仓库之外，不修改用户 HOME 或 shell 配置，不关闭 TLS 验证。完整工具包约3.2 GB，SDK解压后约9.2 GiB，首次安装需预留约14 GB可用空间。安装成功后默认删除下载归档，保留工具和校验记录；可加 --keep-archives 保留归档。

## 安装与完整构建

在仓库根目录运行，工具目录必须在仓库之外：

~~~sh
git lfs pull
python3 tools/harmony/setup.py --root /workspace/work/harmony-build
bash tools/harmony/build.sh /workspace/work/harmony-build
~~~

已有安装且校验记录匹配时，setup 不会重新下载。构建脚本在自己的进程中使用 SDK 自带 Node 和 JDK 17；先核对模型资源，再执行 OHPM 安装、Hvigor clean 和 assembleHap，最后校验实际 HAP 的版本、字节码、AArch64 库、九个模型文件及配套元数据。

OHPM 仍使用官方的用户缓存 ~/.ohpm。在限制文件写入的沙箱中需允许写入该目录；不要通过改 HOME 绕开配置。Hvigor/npm 缓存已定位到工具根目录的 state 下面。Windows/macOS 开发者可继续使用 DevEco Studio，配置自己的调试签名后 Clean/Rebuild。

产物：

- entry/build/default/outputs/default/entry-default-unsigned.hap
- 工具根目录 logs/assemble-hap.log：完整编译日志
- 工具根目录 logs/hap-verification.json：实际打包内容与 SHA-256
- 工具根目录 activate.sh：需要手动调用 SDK 工具时可 source

HAP、SDK、编译缓存和证书不放入 Git。清理构建目录会重新生成 HAP，SDK环境可以继续复用。

## 这次修复与验证边界

1.1.0 的 PermissionModel 在 catch 中直接 throw error。ArkTS 将被捕获值视为任意类型，触发 arkts-limited-throw；普通 TypeScript 严格检查不会发现这个差异。

1.1.1 先把原始异常归一成错误消息，清理临时会话后抛出 new Error(reason)。如果清理本身也失败，以 [PNPU][CLEANUP] 记录，不覆盖原始推理错误。十九组控制流回归通过，包括该场景。此版本已通过实际 HarmonyOS SDK 的 CompileArkTS 和 ARM64 native 构建，完整日志见 tests/evidence/harmony-build-1.1.1.log。

当前构建为0错误，保留23条 MindSpore Lite API 的可选系统能力警告、SDK native 编译参数警告，以及预期的未配置签名提示。没有关闭 ArkTS 检查或屏蔽错误。CPU加载前已有 canIUse 系统能力检查，模型能力及手机执行仍须在设备上验证。

未签名 HAP 不能替代该设备的合法调试签名。当前没有手机连接，也没有真机 NPU RunSync、自检或性能验收结论。构建成功保证的是当前提交在固定 SDK 下可编译、可打包；后续修改 ArkTS/native/build 配置时，应重新执行完整构建。
