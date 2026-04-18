# RL-Agent课题调研（2026）

## 一、课题现状（你现在切入的点）

RL-Agent方向在招聘侧主要分三类：
1. 纯RL训练与控制：DQN/PPO/SAC，强调稳定训练与评测。
2. RL + 工程化：可复现训练流水线、日志监控、自动评估、回测系统。
3. RL + Agent工作流：把训练、评估、诊断、报告做成可自动执行的实验智能体。

你的优势切入点：
- 先做“可运行可解释”的RL baseline；
- 再叠加Agent编排能力；
- 最后迁移到交易或具身场景。

## 二、代表性开源项目（建议优先顺序）

### 1) Stable-Baselines3（优先）
- 定位：工业界/学术界常用的稳定RL基线库（PyTorch）。
- 价值：统一API、文档完整、上手快，适合先建立“稳”的实验框架。
- 入口：
  - https://stable-baselines3.readthedocs.io/en/master/guide/quickstart.html
  - https://github.com/DLR-RM/stable-baselines3

### 2) RL Baselines3 Zoo
- 定位：SB3训练框架，包含调参、评测、可视化和已调好的超参。
- 价值：直接学习规范实验流程和benchmark思路。
- 入口：
  - https://github.com/DLR-RM/rl-baselines3-zoo

### 3) CleanRL（强烈推荐阅读代码）
- 定位：单文件高质量实现，研究友好。
- 价值：非常适合理解算法细节和调试路径。
- 入口：
  - https://docs.cleanrl.dev/
  - https://github.com/vwxyzjn/cleanrl

### 4) FinRL / FinRL-X（交易场景）
- 定位：金融强化学习框架，包含数据、环境、训练与回测流程。
- 价值：可直接参考交易任务组织方式和评估逻辑。
- 入口：
  - https://github.com/AI4Finance-Foundation/FinRL

### 5) RLlib（后期再上）
- 定位：分布式RL训练框架（Ray生态）。
- 价值：大规模并行和工程化能力强，但学习成本较高。
- 入口：
  - https://docs.ray.io/en/master/rllib/index.html

## 三、常见研究坑

1. 只看收益不看风险：会导致策略不可用。
2. 数据泄漏：训练/验证/测试切分不严格。
3. 缺少基线：没有和简单策略对比，结果无法解释。
4. 仅一次实验：没有多seed评估，结论不稳。
5. 不可复现：缺少配置管理、日志、版本记录。

## 四、你可以讲给面试官的研究路线

1. 阶段1：快速建立DQN/PPO基线，验证实验流水线。
2. 阶段2：引入交易环境与风险约束，做基线对比。
3. 阶段3：加入Agent编排（训练、评估、失败诊断、报告生成）。
4. 阶段4：做walk-forward与多seed评估，形成可复现结论。
