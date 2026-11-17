# 双模型切换与 TinyBERT 验收 · 1.1.2

仓库根目录现在构建同一个双模型应用，包名仍为 com.intentlab.permissiondemohiaperf。默认选择 TinyBERT 4L，可切换到 MiniRBT H256；两套模型均可选择自动、16、32、64 token。模型二进制没有重新训练或重新编译，本轮更改集中在集成、切换和验收入口。

双模型 rawfile 二进制合计250,708,535字节，约239 MiB（HAP压缩前）；这会增加安装包体积，不表示全部模型同时驻留内存。仅测试 TinyBERT 时仍可生成独立工程。

## 构建与操作

在 feat/tinybert-4l-kirin9030 分支执行：

~~~sh
git pull --ff-only
git lfs pull
python3 ml/verify_artifacts.py
~~~

校验应通过十个二进制、两个模型目录及配套元数据。不要使用 GitHub 源码 ZIP 中尚未展开的 LFS 指针文件直接构建。

用 DevEco Studio 打开仓库根目录，配置本机调试签名，Clean/Rebuild native 与 ArkTS 后安装到 Mate 80 Pro Max / HarmonyOS 5.0.0。确认界面版本为 1.1.2。此前的独立 TinyBERT 包名不同，可能与新应用并存。

1. 保持默认 TinyBERT，输入“录一段视频，别录声音”，分别选16、32、64识别。结果应显示当前模型、实际后端和档位；初次使用会加载并执行参考自检。
2. 切换 MiniRBT，重复三个档位，再切回 TinyBERT。核对模型名与实际后端，留意异常、残留旧结果和内存增长。
3. 自动模式会选择能容纳输入的最小 NPU 档位；失败后可继续较大档位。手动模式只尝试指定 NPU 档位，超长直接拒绝，不会截断。正常识别可退到同模型的64 token CPU，界面明确显示 CPU。
4. 点击“验收 TinyBERT 4L 三档 NPU”。等待逐档完成，再点“复制验收 JSON”，保留完整内容。输入框和模型选择在验收期间禁用。切换模型会清除界面上旧模型的验收结果。
5. HiLog 按 Tag PermissionNPU 过滤并包含 Info/Warn/Error；重点保留 [MODEL]、[LOAD]、[INPUT]、[SELFTEST]、[ACCEPT]，失败时保留 [ERROR] 和 [FALLBACK]。

## 本轮 NPU 验收范围与标准

验收会重新创建每档会话，不复用此前成功结果，也不调用应用 CPU 回退路径。每次最多保留一个 NPU 档位，会话切换、失败和验收结束时显式释放。原生在途任务有独立所有权，不会因句柄释放而访问已销毁的模型。

| 档位 | 可容纳的参考指令 | 通过条件 |
| --- | ---: | --- |
| 16 | 27 | 27/27 权限集合一致 |
| 32 | 33 | 33/33 权限集合一致 |
| 64 | 33 | 33/33 权限集合一致 |

两套模型均使用各自的33条参考数据。MiniRBT 的手机自检由原来8条补齐至已有测试向量的33条，没有借用 TinyBERT logits。token 数含 CLS/SEP。

每个输出必须是5个有限 logit；除了权限集合完全一致，最大绝对 logit 差还须 ≤0.1。这是当前暂定的 FP16 工程门槛，尚未由手机误差分布校准；需根据首轮完整结果评审，不能为了“通过”而直接放宽。参考覆盖数量不足、标签不一致、NaN、异常或任意档位失败，整体均不通过。

每档用同一条能装入该档的参考短指令，预热5次、记录50次，报告 P50/P95。时延包括原生桥接、调度、张量复制和推理，不包括加载、自检、分词和文本渲染。本轮未定义延迟 SLA，时延是测量项，不是性能达标结论。

JSON 包含版本、时间、模型 ID、运行库后端/版本、阈值、档位、模型字节数、清单 SHA-256、参考匹配数、最大误差、时延及失败消息。1.1.2 的 schemaVersion=2 还包含手机实际系统/API版本、逐样本logits、标签变化、阈值余量与失败样本token；详见 [TinyBERT数值调查](tinybert-npu-parity.md)。清单 SHA-256 是构建前校验过的身份；手机只核对模型字节数，没有进行手机端 SHA-256 测量，不能把 manifestSha256 当成设备实测值。

这个入口验证的是转换后输出一致性和基本执行稳定性。应用层禁止 CPU 回退、编译分区 CPUCL=0，均不能代替手机 profiler 对底层执行位置的观察。分词、输入准备、阈值判断和界面仍在 CPU 执行。

## 已完成的主机检查

2026-11-17 本轮重新执行：

- 19组实际 ArkTS 路由／切换／验收控制流测试通过；NPU 与 MindSpore Lite 接口为 mock。两个真实词表分别在33条 input_ids / attention_mask 上一致。
- 75组实际原生回调与故障注入通过，包括释放、在途推理所有权、重复释放、队列失败恢复；不是手机执行结果。
- 原生 CMake 主机测试继续保留；另已使用真实 HarmonyOS SDK 编译 AArch64 桥接库，验证实际 C++ catch 处理代码，并与 ArkTS 字节码一起打入1.1.1 HAP。早期主机占位链接库没有用于 HAP。
- 十个模型二进制 SHA-256、目录身份和配套元数据通过校验。
- TinyBERT 三档标准 ONNX 权限对照为27/27、33/33、33/33，最大 logit 差3.933906555175781e-6。
- TinyBERT MindSpore Lite CPU 对照33/33，最大 logit 差2.4332202e-5，使用 TinyBERT 自己的阈值。
- TinyBERT 既有编译产物再次核验：每档8个 BatchMatMulV2、4个 SoftmaxV2 的原生 tiling/NPU 任务及内核二进制齐全，NPUCL=1、CPUCL=0。
- 独立 TinyBERT 工程生成仍可使用，目录由单模型配置驱动，不再修改 ArkTS 源码字符串。

可复现命令：

~~~sh
node --experimental-strip-types tests/nnrt_startup_test.mjs
node --experimental-strip-types tests/model_switch_test.mjs
python3 tests/run_native_error_test.py --sdk-include /path/to/public-sdk-headers
python3 ml/verify_artifacts.py

# TinyBERT 的 Linux CPU 回退模型；不能误用根目录 MiniRBT 阈值。
python3 ml/verify_ms.py /path/to/mindspore-lite-2.1.0-linux-x64 \
  --model entry/src/main/resources/rawfile/models/tinybert4l/permission.ms \
  --vectors variants/tinybert4l/test_vectors.json \
  --config entry/src/main/resources/rawfile/models/tinybert4l/model_config.json
~~~

## 手机与模型质量待验收项

当前 Codex 已安装 HarmonyOS 5.0.0.71 / API 12、OHPM、Hvigor、JDK 17，并完成1.1.1完整干净构建及未签名 HAP 包内校验，详见 [云端构建](../tools/harmony/README.md)。当前仍没有签名配置及连接的手机，未产生手机验收 JSON。已有手机日志只证明 MiniRBT 1.0.8 完成三档会话创建，随后在 TypedArray 参数校验被阻断；1.0.9 的兼容性修复尚未获得新的手机 RunSync 结果。不能以本轮主机测试宣布真机问题已经解决。

首轮手机验收需确认三档 RunSync、全部参考输出与误差；随后测冷加载、真实指令的端到端耗时、峰值内存、连续推理温升与跨模型反复切换。CPU Model 的公开 JS 接口没有 dispose，切换时只能清掉应用引用，回收时机由运行库决定；需要在手机上观察实际内存。

当前仍是少量合成数据训练的 Demo：TinyBERT 测试完全匹配25/33，MiniRBT 为24/33；对应的33/33转换一致性不表示真实标签正确。任务质量验收还需要独立真实中文、否定／组合／歧义／域外样本，多标签 Precision/Recall/F1、完全匹配率和过度预测率，并约定上线门槛。
