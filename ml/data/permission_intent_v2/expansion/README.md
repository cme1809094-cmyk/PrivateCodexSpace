# 扩充语料与测试候选

原300条保持原文件和审核状态；本轮新增1200条，共240个场景族，每族5种自然表达。合计1500条。字典继续使用上层目录的32项能力和条件映射，不新增平台权限名称或执行接口。

新增部分的候选分配为训练750条/150族、验证150条/30族、测试300条/60族。原300条在合并视图中标为reference_only，不进入任何训练或调参候选。所有人工审核均为pending；这些是合成候选材料，尚未冻结为独立验收集，也未用于训练或测量模型准确率。

最通俗的解释见[测试集说明](../../../../docs/permission-intent-test-design.md)。

## 文件和复现

- shards/：四位作者的源语料；每个分片300条，人工修改应先改源分片。
- expanded_instructions.jsonl：合并后的1200条新增语料。
- all_instructions.jsonl：含旧参考300条的1500条完整视图。
- splits/：按候选用途拆分的四份JSONL；reference_only单列。
- family_manifest.json：240个场景的摘要、标准答案和成员ID。
- split_manifest.json：各候选文件的ID、场景ID与SHA256。
- instruction_review.csv：1500条人审视图，包含中文能力名称。
- family_review.csv：240个场景的人审视图，每行并排放五种说法，便于检查改写是否保持意思。
- test_review.csv / test_set_overview.json：单独的300条测试候选审核表及结构统计。
- token_length_report.json：用应用实际WordPiece和两份配套词表检查原话长度；1500条均可放入64 token，最长含特殊符号55 token。这不包含未来上下文拼接，也不代表模型语义答对。
- validation_report.json：数量、去重、结构、引用、覆盖与近似说法筛查结果。
- ai_review_report.json：独立AI复核的实际覆盖、修正记录与最终分片校验值。

在仓库根目录运行：

    python3 ml/prepare_permission_intent_expansion.py --render --require-cross-review

生成视图会被再次运行覆盖。请先另存人工审核副本再填写human_decision和human_notes；表格意见不会自动同步到源JSONL。修订源数据后要重新复核和生成候选文件。

## 使用约束

同一场景的五条改写不得跨集合。场景ID不相同不代表语义独立，近似说法筛查也不能代替语义复核。若未来把旧300条或外部数据并入训练，必须重新对测试场景做泄漏审查。

context_required表示标准答案需要额外背景，不能把没有输入这些背景的单句分类器直接拿来判卷。现有Demo仍为五标签输出，不能直接承担本字典32能力、否定、范围和追问等全部任务。

测试候选已暴露给数据作者和审核人员；它们适合后续人工筛选和内部阶段评估，不是天然盲测或生产准确率证据。冻结前应核对真实使用分布、输出协议和版本，冻结后不能用测试结果反复调参而继续称为独立考试。

三个候选集合均覆盖32项能力正例；测试候选中的明确禁止题尚未覆盖每一项能力，缺项在test_set_overview.json中列出。这是正式冻结前需要补齐的覆盖，不因结构检查通过就视为全面验收。
