---
license: mit
---

# AgentDoG-Lite Test Set Dataset Card

## Dataset Summary

本仓库提供 **AgentDoG-Lite：基于 AgentDoG 1.5 的轻量级 Agent 安全诊断挑战** 的测试数据，用于评估参赛方案在 **Agent 轨迹级安全判断** 任务上的表现。

与传统只判断单轮输入或最终回复是否安全的评测不同，本测试集关注完整 Agent 执行轨迹中的安全风险。每条样本包含一段多轮 Agent 轨迹，可能涉及用户请求、工具描述、工具调用、工具返回、环境反馈和最终回复。参赛系统需要根据完整轨迹判断 Agent 是否执行了不安全行为或不安全决策模式。

本测试集主要用于评测参赛者基于 `Qwen3.5-0.8B` 构建的安全判断系统，包括：

- 单人赛：training-free harness 方法
- 团队赛：基于 `Qwen3.5-0.8B` 微调后的模型

核心输出为二分类安全判断：

```json
{
  "judgment": "safe"
}

或

{
  "judgment": "unsafe"
}
```

本数据集不应用于：
  训练或微调模型
  
  few-shot 示例选择
  
  prompt 搜索或测试集调参
  
  生成真实攻击流程或执行危险命令
  
  作为真实部署环境中的唯一安全判断依据