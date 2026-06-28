# llm_marl_mappo_selfplay
# LLM_MARL_MAPPO_SELFPLAY
LLM多智能体MAPPO+Self-Play工具强化学习平台
适配Moonshot Kimi复杂长流程自主智能体业务场景

## 项目整体介绍
本项目基于CTDE范式实现多智能体强化学习，自研三层分层奖励函数，内置Self-Play自博弈迭代数据生成逻辑，基于FSDP分布式底座实现7B大模型多卡并行训练，配套Rust安全代码沙箱、vLLM低延迟推理引擎，完整覆盖仿真交互、训练、线上推理全链路。

## 项目目录说明
1. configs：训练、推理超参配置文件
2. dist_train：FSDP分布式训练底座、4卡启动脚本
3. docs：全套理论推导、架构、消融实验方案文档
4. experiments：消融实验启动脚本、指标绘图代码
5. inference_engine：vLLM异步推理服务模块
6. rl_core：PPO/MAPPO/Self-Play/奖励函数核心算法
7. rust_sandbox：安全隔离代码执行沙箱
8. simulator：多智能体仿真交互环境

## 核心模块能力
1. MAPPO多智能体协作：全局集中式Critic解决多角色局部最优、协作崩塌问题
2. Self-Play自博弈：自动生成高难度长流程样本，降低人工标注成本
3. FSDP分布式训练：显存分片优化，单机4卡可平滑扩容至千卡集群
4. Rust沙箱：异步批量执行代码，安全隔离恶意脚本，并发性能优于原生Python
5. 完整消融实验：验证所有算法、工程优化模块有效性，提供量化对比指标

## 开发环境
GitHub Codespaces 2核实例，Python3.11 + Rust，PyTorch训练框架