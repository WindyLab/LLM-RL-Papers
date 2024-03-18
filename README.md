# LLM-RL-Cross-Papers
Monitoring recent cross-research on LLM &amp; RL on arXiv

Welcome to launch PRs if there are good papres.

***

[TOC]



***

## Paper



- **RL-GPT: Integrating Reinforcement Learning and Code-as-policy**

  - Paper Link : [arXiv 2402.19299](https://arxiv.org/abs/2402.19299) ,  [homepage](https://sites.google.com/view/rl-gpt/)

  - Method Overview: An overview of RL-GPT includes a slow agent (orange) and a fast agent (green). The slow agent breaks down the task and determines “which actions” to learn. The fast agent codes and sets RL configurations for execution. The LLM can generate environment configurations (task, observation, reward, action space) for a subtask. By considering the agent’s behavior to solve the subtask, the LLM provides higher-level actions, enhancing RL’s sample efficiency.

  - Framework Overview: The overall framework consists of a slow agent (orange) and a fast agent (green). The slow agent decomposes the task and determines “which actions” to learn. The fast agent writes code and RL configurations for low-level execution.

<img src="./images/RL-GPT framework.png" alt="RL-GPT framework" style="zoom: 50%;" />

+++

- **Natural Language Reinforcement Learning**
  - Paper Link: [arXiv 2402.07157](https://arxiv.org/abs/2402.07157) 
  - Method Overview: NLRL is inspired by human learning processes. It redefines traditional RL concepts like task objectives, policies, value functions, and policy iteration using natural language space. 
  - Framework Overview: 

<img src="./images/NLRL.png" style="zoom:50%;" />

+++

- 

