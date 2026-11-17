# 首批能力字典与代表性指令

本版覆盖12个新增能力组、3个保留组，共32项能力、109条平台映射和300条种子指令。

条目由子agent生成，AI交叉复核状态见独立报告；人工审核仍为pending。结构校验通过不等于人工审核或平台实际可调用。

## 文件

- [capability_catalog.json](capability_catalog.json)：能力定义、平台版本条件、授权路径、限制与证据来源。
- [representative_instructions.jsonl](representative_instructions.jsonl)：指令、语义三态、范围条件与追问；以能力ID引用字典。
- [capability_rules_review.csv](capability_rules_review.csv)：逐条平台规则的人审表。
- [instruction_review.csv](instruction_review.csv)：逐条指令、平台映射条件的人审表。
- [validation_report.json](validation_report.json)：数量、去重、引用、标注一致性及待核实映射统计。
- [ai_review_report.json](ai_review_report.json)：独立子agent的语义/平台交叉复核记录。

## 语义和平台映射

required表示任务所需能力，不表示系统已授权。forbidden只记录用户明确禁止的完整能力；未提及保持unmentioned。模态、目标、范围、持续时间的部分限制放在constraints，避免扩大成整项能力禁止。

模型学习能力与语义，平台权限统一按字典的API、route、开放级别和交互条件解释。candidate_permissions不属于可执行请求方案。restricted/needs_verification及仅OpenHarmony参考的映射仍须核实，不能直接作为普通HarmonyOS应用的申请依据。

在线服务INTERNET属于实现依赖，不等于需要设备定位。系统选择器URI授权仍需要用户选择；取消、拒绝或认证失败不扩大其他访问范围。

## 审核

先审动作、范围、否定、歧义和上下文，再审API及实现route。CSV中的human_decision/human_notes留给人工填写；human_review保持pending，不能因一次格式确认批量标成人工审核通过。

CSV是生成视图。填写前请另存审核副本；再次运行--render会重新生成并覆盖这两份CSV，不会自动把表格意见写回JSON/JSONL。修改canonical数据后须重新交叉复核。

本批用于人工确认和后续扩写，尚未分配训练/验证/测试集合，也未重新训练Demo。后续应按语义场景族拆分，避免近似改写泄漏到独立验收集。

## 能力索引

| 能力ID | 名称 | 分组 |
| --- | --- | --- |
| media.photo.pick_one | 系统选图器选择一张照片 | photos |
| media.photo.read_library | 读取照片库 | photos |
| media.photo.add | 保存新照片到相册 | photos |
| files.pick_one | 选择并读取一个文件 | files |
| files.save_user_selected | 保存文件到用户选择的位置 | files |
| contacts.pick_one | 系统选择一位联系人 | contacts |
| contacts.read | 直接读取通讯录 | contacts |
| contacts.write | 创建或修改联系人 | contacts |
| calendar.read | 读取可访问日历日程 | calendar |
| calendar.write | 创建或修改日历日程 | calendar |
| notifications.publish | 发布本应用通知 | notifications |
| notifications.open_settings | 打开本应用通知设置 | notifications |
| reminders.schedule_local | 安排本地提醒 | reminders |
| reminders.schedule_exact | 安排要求准时触发的提醒 | reminders |
| bluetooth.scan | 发现附近蓝牙设备 | bluetooth |
| bluetooth.connect | 连接或配对蓝牙设备 | bluetooth |
| wlan.scan | 查看或扫描附近WLAN | wlan |
| wlan.connect | 请求连接WLAN | wlan |
| nfc.read_tag | 读取NFC标签 | nfc |
| nfc.write_tag | 写入NFC标签 | nfc |
| motion.read_steps | 读取步数传感器 | motion |
| motion.recognize_activity | 识别运动状态 | motion |
| biometric.authenticate | 系统生物认证 | biometric |
| clipboard.paste_user_initiated | 用户主动粘贴 | clipboard |
| clipboard.read | 直接读取当前剪贴板 | clipboard |
| clipboard.write | 写入剪贴板 | clipboard |
| camera.capture_photo | 相机拍照 | base_camera |
| camera.record_video | 相机录制视频 | base_camera |
| microphone.record_audio | 麦克风录音 | base_microphone |
| location.coarse.foreground_once | 前台单次粗略定位 | base_location |
| location.precise.foreground_once | 前台单次精确定位 | base_location |
| location.background | 后台持续定位 | base_location |

复现检查：

    python3 ml/validate_permission_intent_v2.py --render
