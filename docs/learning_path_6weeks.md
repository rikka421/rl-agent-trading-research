# RL-Agent 六周学习与实战路径

## Week 1: RL基础与工具链
- 目标：搞清楚MDP、value/policy、exploration、off-policy。
- 必做：
  - 看 Spinning Up 的 RL Intro
  - 跑通 Gymnasium + SB3 CartPole
- 输出：一页笔记 + 第一个可复现实验日志

## Week 2: DQN/PPO核心机制
- 目标：理解DQN与PPO训练差异，能解释超参数作用。
- 必做：
  - 读 CleanRL 的 dqn.py / ppo.py
  - 改 2-3 个关键超参观察曲线变化
- 输出：超参影响小结表

## Week 3: 交易任务建模
- 目标：把OHLCV映射为状态、动作、奖励。
- 必做：
  - 下载SPY历史数据
  - 设计最小交易环境（买/持/卖）
- 输出：环境定义文档 + baseline回测

## Week 4: 风险约束与评测
- 目标：建立可解释指标体系。
- 必做：
  - 指标：累计收益、最大回撤、胜率、换手率
  - 对比：Buy-and-Hold、均线策略
- 输出：对比图 + 失败案例分析

## Week 5: Agent编排
- 目标：自动执行 train->eval->report。
- 必做：
  - 编写 orchestrator
  - 自动汇总指标与失败case
- 输出：一键执行脚本 + 自动报告模板

## Week 6: 研究化与面试化
- 目标：把工程成果转为“可讲述”的研究闭环。
- 必做：
  - 多seed + walk-forward验证
  - 写一页实验结论和局限性
- 输出：简历项目定稿 + 5个高频问答

## 核心资料
- Spinning Up: https://spinningup.openai.com/en/latest/
- Gymnasium: https://gymnasium.farama.org/
- SB3 Docs: https://stable-baselines3.readthedocs.io/
- CleanRL Docs: https://docs.cleanrl.dev/
- FinRL: https://github.com/AI4Finance-Foundation/FinRL
