# LLM-RL-Cross-Papers
Monitoring recent cross-research on LLM &amp; RL on arXiv

Welcome to launch PRs if there are good papres.

***

todo: table of content



***

todo: Paper list

## Paper



- **RL-GPT: Integrating Reinforcement Learning and Code-as-policy**

  - Paper Link :* [arxiv 2402.19299](https://arxiv.org/abs/2402.19299),  [homepage](https://sites.google.com/view/rl-gpt/)

  - Method Overview: An overview of RL-GPT includes a slow agent (orange) and a fast agent (green). The slow agent breaks down the task and determines “which actions” to learn. The fast agent codes and sets RL configurations for execution. The LLM can generate environment configurations (task, observation, reward, action space) for a subtask. By considering the agent’s behavior to solve the subtask, the LLM provides higher-level actions, enhancing RL’s sample efficiency.

  - Framework Overview: The overall framework consists of a slow agent (orange) and a fast agent (green). The slow agent decomposes the task and determines “which actions” to learn. The fast agent writes code and RL configurations for low-level execution.

<img src="/home/ubuntu/Desktop/LLM-RL-Cross-Papers/images/RL-GPT framework.png" alt="RL-GPT framework" style="zoom: 50%;" />











