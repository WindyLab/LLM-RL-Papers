

# LLM-RL-Cross-Papers

1. Monitoring recent cross-research on LLM &amp; RL on arXiv.
2. Focusing on combining LLM & RL capabilities for control (such as game characters).

3. Welcome to launch PRs if there are good papres.

***

[TOC]



***

## Paper



- **RL-GPT: Integrating Reinforcement Learning and Code-as-policy**

    - Paper Link : [arXiv 2402.19299](https://arxiv.org/abs/2402.19299) ,  [homepage](https://sites.google.com/view/rl-gpt/)

    - Framework Overview: 

        <img src="./images/RL-GPT framework.png" alt="RL-GPT framework" style="zoom: 50%;" />

        Fig 1.  The overall framework consists of a slow agent (orange) and a fast agent (green). The slow agent decomposes the task and determines “which actions” to learn. The fast agent writes code and RL configurations for low-level execution.

    - Method Overview:  RL-GPT includes a slow agent and a fast agent.  The LLM can generate environment configurations (task, observation, reward, action space) for a subtask. By considering the agent’s behavior to solve the subtask, the LLM provides higher-level actions, enhancing RL’s sample efficiency.


***

- **Natural Language Reinforcement Learning**
  - Paper Link: [arXiv 2402.07157](https://arxiv.org/abs/2402.07157) 

  - Framework Overview: 

      <img src="./images/NLRL.png" style="zoom:50%;" />

      Fig1. The authors present an illustrative example of grid-world MDP to show how NLRL and traditional RL differ for task objective, value function, Bellman equation, and generalized policy iteration. In this grid-world, the robot needs to reach the crown and avoid all dangers. We assume the robot policy takes optimal action at each non-terminal state, except a uniformly random policy at state b.

  - Method Overview: NLRL is inspired by human learning processes. It redefines traditional RL concepts like task objectives, policies, value functions, and policy iteration using natural language space. 

***

- **RLingua: Improving Reinforcement Learning Sample Efficiency in Robotic Manipulations With Large Language Models**
  - Paper Link: [arXiv 2403.06420](https://arxiv.org/abs/2403.06420) , [homepage](https://rlingua.github.io/)
  
  - Framework Overview:
  
      <img src="./images/RLingua framework.png" alt="RLingua framework" style="zoom: 80%;" />
  
      Fig.1:  (a) Motivation: LLMs do not need environment samples and are easy to communicate for non-experts. However, the robot controllers generated directly by LLMs may have inferior performance. In contrast, RL can be used to train robot controllers to achieve high performance. However, the cost of RL is its high sample complexity. (b) Framework: RLingua extracts the internal knowledge of LLMs about robot motion to a coded imperfect controller, which is then used to collect data by interaction with the environment. The robot control policy is trained with both the collected LLM demonstration data and the interaction data collected by the online training policy
  
      <img src="./images/RLingua 2.png" alt="RLingua 2" style="zoom:50%;" />
  
      Fig2. The framework of prompt design with human feedback. The task descriptions and coding guidelines are prompted in sequence. The human feedback is provided after observing the preliminary LLM controller execution process on the robot.
  
  - Method Overview: RLingua is the combination of LLM Controller & RL. It extracts the LLM's knowledge about robot motion to improve the sample efficiency of RL.  

***

- **Hierarchical Continual Reinforcement Learning via Large Language Model**

    - Paper Link: [arXiv 2401.15098](https://arxiv.org/abs/2401.15098)

    - Framework Overview:

      <img src="images/Hi_Core framework.png" alt="Hi_Core framework" style="zoom:67%;" />

      Fig 1. The illustration of the proposed framework. The middle section depicts the internal interactions (**light gray line**) and external interactions (**dark gray line**) in Hi-Core. Internally, the CRL agent is structured in two layers: the high-level policy formulation (**orange**) and the low-level policy learning (**green**). Furthermore, the policy library (**blue**) is constructed to store and retrieve policies. The three surrounding boxes illustrate their internal workflow when the agent encounters new tasks.

    - Method Overview: The high level LLM is used to generate a series of goals g_i . The low level is a RL with goal-directed, it needs to generate a policy in response to the goals. Policy library is used to store successful policy. When encountering new tasks, the library can retrieve relevant experience to assist high and low level policy agent.

***

- **True Knowledge Comes from Practice: Aligning LLMs with Embodied Environments via Reinforcement Learning**

    - Paper Link: [arXiv 2401.14151](https://arxiv.org/abs/2401.14151) , [homepage](https://github.com/WeihaoTan/TWOSOME)

    - Framework Overview: 

        <img src="./images/TWOSOME framework.png" style="zoom: 67%;" />

        Fig 1. Overview of how TWOSOME generates a policy using joint probabilities of actions. The color areas in the token blocks indicate the probabilities of the corresponding token in the actions.

    - Method Overview: 

        The authors propose *True knoWledge cOmeS frOM practicE*(**TWOSOME**) online framework. It deploys LLMs as embodied agents to efficiently interact and align with environments via RL to solve decision-making tasks w.o. prepared dataset or prior knowledge of the environments. They use the loglikelihood scores of each token provided by LLMs to calculate the joint probabilities of each action and form valid behavior policies.

***

## Open source RL environment 

- Skillhack: https://github.com/ucl-dark/skillhack

    <img src="./images/skillshack.png" style="zoom: 33%;" />

- Minigrid: https://github.com/Farama-Foundation/MiniGrid?tab=readme-ov-file

    <img src="./images/door-key-curriculum.gif" style="zoom: 50%;" />

- Crafter: https://github.com/danijar/crafter?tab=readme-ov-file

    ![](./images/crafter.gif)

- Multi Agent RL Environment: https://github.com/Bigpig4396/Multi-Agent-Reinforcement-Learning-Environment

    <img src="./images/MultiAgentRLenvDrones.gif" style="zoom:80%;" />

- OpenAI Multi Agent Particle Env: https://github.com/openai/multiagent-particle-envs

    <img src="./images/MultiAgentParticle.gif" style="zoom: 50%;" />

- MAgent2: https://github.com/Farama-Foundation/MAgent2?tab=readme-ov-file

    <img src="./images/MAgent.gif" style="zoom: 67%;" />
