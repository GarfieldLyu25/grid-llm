# CLAUDE.md — Grid-LLM 项目 AI 助手指令

## 语言

中文回答，中文注释。代码注释用中文。

## 开发流程（强制执行）

本项目使用 mew-spec 流程，**任何功能、模块或代码改动，必须先走完四份文档，获得用户批准后才能写实现代码。**

```
spec.md → plan.md → task.md → checklist.md → 开发 → 验收
```

### HARD GATE

> 四份文档全部生成并获得用户批准之前，**禁止编写任何实现代码。**
> 无论改动看起来多简单，一律走完流程。

### 文档职责

| 文档 | 回答 | 存放位置 |
|------|------|---------|
| spec.md | 做什么 | `docs/iterations/<编号>-<名称>/spec.md` |
| plan.md | 怎么做 | `docs/iterations/<编号>-<名称>/plan.md` |
| task.md | 按什么顺序做 | `docs/iterations/<编号>-<名称>/task.md` |
| checklist.md | 做对了没 | `docs/iterations/<编号>-<名称>/checklist.md` |

### 流程规则

1. **一次一个问题** — 需求澄清阶段，每次只问一个选择题
2. **逐段审批** — 每段确认后再展示下一段
3. **首选选择题** — 比开放题更容易回答
4. **先有证据再下结论** — 先跑命令看输出，再报状态
5. **YAGNI** — 只设计和实现 spec 提到的内容
6. **每个任务完成后必须跑验证** —「应该没问题」不算证据
7. **被阻塞时停下来问** — 不要猜

### 审批流程

每个阶段完成后，呈现给用户审批：
- spec.md → 用户审批功能需求
- plan.md → 用户审批架构设计
- task.md → 用户审批任务拆解
- checklist.md → 用户审批验收方案
- 开发阶段 → 按 task.md 逐任务执行
- 验收阶段 → 按 checklist.md 逐项验证

## 项目技术约定

- **框架语言**：Python 3.12（AutoDL 镜像：PyTorch 2.8 / CUDA 12.8 / Ubuntu 22.04）
- **包管理器**：uv（`uv sync` 创建隔离环境，`uv run python` 执行脚本，不污染系统 Python）
- **代码路径**：所有 Python 代码在 `src/grid_llm/` 下，包名为 `grid_llm`
- **模型**：Qwen2.5-7B-Instruct
- **日志**：所有日志写 `logs/YYYY-MM-DD/` 目录，文件级别 DEBUG，控制台 INFO
- **共享常量**：`utils/common.py` 提供 `SYSTEM_PROMPT`、`DEFAULT_MAX_NEW_TOKENS`、`MIN_INSTRUCTION_LEN` 等，不要在各文件里硬编码
- **训练框架**：LLaMA-Factory（首选）/ transformers + peft（备选）
- **RL 框架**：TRL GRPOTrainer
- **量化**：autoawq
- **部署**：vllm
- **GPU**：单卡 RTX 5090（32GB），峰值显存不超过 28GB
- **数据集**：ape210k + CMATH（中文数学）

## 代码风格

- 遵循 mew-spec 核心原则：spec 聚焦行为描述，plan 定义接口，task 写具体步骤
- 所有脚本独立可运行，参数通过配置文件和命令行传入；默认优先读取 `config/config.local.yaml`，不存在时读取 `config/config.yaml`
- 每阶段产出物（adapter / 合并模型 / 量化模型）独立保存，路径写进配置文件
- 训练脚本带 checkpoint 续训逻辑
- 评估脚本输出可对比的指标（准确率、loss、吞吐）
