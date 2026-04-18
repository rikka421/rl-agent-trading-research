# 实施决策记录（2026-04-18）

## 你的选择 -> 当前落地

1. 框架选择：采用 Stable-Baselines3 + Gymnasium
- 原因：入门资料多、生态稳定、业界使用广。

2. 算法选择：交易任务默认 DQN，同时保留 PPO 对比配置
- 原因：当前简历主线强调离散动作交易决策，DQN更贴合最小可讲清链路。
- 兼容：保留 PPO 作为第二条 RL baseline，用于比较不同算法在同一奖励设计下的行为差异。

3. 编排选择：先稳定版，再 LLM版
- 稳定版：`orchestrator.py`（train -> eval）
- LLM版：`orchestrator_llm.py`（DeepSeek 先给实验建议，再执行 train -> eval）

4. 设备策略：GPU可用但轻量训练
- 使用 `device: auto` 自动识别 GPU。
- 默认训练步数保持轻量级，避免笔记本过热或训练过久。

5. 任务选择：从通用控制切到 trading-first
- 已实现 `env_trading.py`，将 OHLCV 映射为短/空/多离散动作任务。
- 奖励函数采用收益 - 交易成本 - 回撤增量惩罚。
- 评估时固定输出 RL、Buy-and-Hold、均线策略三类结果，方便简历与面试复述。

## DeepSeek 接入说明

- 环境变量：`DEEPSEEK_API_KEY`
- 默认模型：`deepseek-chat`
- 默认接口：`https://api.deepseek.com/chat/completions`

如果未设置 API Key，LLM编排器会自动跳过建议生成并提示你设置。