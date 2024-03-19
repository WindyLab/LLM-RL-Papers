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

  - Framework Overview: 

<img src="./images/RL-GPT framework.png" alt="RL-GPT framework" style="zoom: 50%;" />

Fig 1.  The overall framework consists of a slow agent (orange) and a fast agent (green). The slow agent decomposes the task and determines “which actions” to learn. The fast agent writes code and RL configurations for low-level execution.

***

- **Natural Language Reinforcement Learning**
  - Paper Link: [arXiv 2402.07157](https://arxiv.org/abs/2402.07157) 
  - Method Overview: NLRL is inspired by human learning processes. It redefines traditional RL concepts like task objectives, policies, value functions, and policy iteration using natural language space. 
  - Framework Overview: 

<img src="./images/NLRL.png" style="zoom:50%;" />

Fig1. The authors present an illustrative example of grid-world MDP to show how NLRL and traditional RL differ for task objective, value function, Bellman equation, and generalized policy iteration. In this grid-world, the robot needs to reach the crown and avoid all dangers. We assume the robot policy takes optimal action at each non-terminal state, except a uniformly random policy at state b.

***

- **RLingua: Improving Reinforcement Learning Sample Efficiency in Robotic Manipulations With Large Language Models**
  - Paper Link: [arXiv 2403.06420](https://arxiv.org/abs/2403.06420) , [homepage](https://rlingua.github.io/)
  - Method Overview: RLingua is the combination of LLM Controller & RL. It extracts the LLM's knowledge about robot motion to improve the sample efficiency of RL.  
  - Framework Overview:

<img src="/home/ubuntu/Desktop/LLM-RL-Cross-Papers/images/RLingua framework.png" alt="RLingua framework" style="zoom: 80%;" />

Fig.1:  (a) Motivation: LLMs do not need environment samples and are easy to communicate for non-experts. However, the robot controllers generated directly by LLMs may have inferior performance. In contrast, RL can be used to train robot controllers to achieve high performance. However, the cost of RL is its high sample complexity. (b) Framework: RLingua extracts the internal knowledge of LLMs about robot motion to a coded imperfect controller, which is then used to collect data by interaction with the environment. The robot control policy is trained with both the collected LLM demonstration data and the interaction data collected by the online training policy

<img src="/home/ubuntu/Desktop/LLM-RL-Cross-Papers/images/RLingua 2.png" alt="RLingua 2" style="zoom:50%;" />

Fig2. The framework of prompt design with human feedback. The task descriptions and coding guidelines are prompted in sequence. The human feedback is provided after observing the preliminary LLM controller execution process on the robot.

***



