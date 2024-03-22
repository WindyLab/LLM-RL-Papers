

# LLM-RL-Cross-Study

1. Monitoring recent cross-research on LLM &amp; RL on arXiv.
2. Focusing on combining LLM & RL capabilities for control (such as game characters).

3. Feel free to open PRs if you want to share the good papers you’ve read.

***

- [LLM-RL-Cross-Study](#llm-rl-cross-study)
  * [Papers](#papers)
    + [The RL/LLM Taxonomy Tree: Reviewing Synergies Between Reinforcement Learning and Large Language Models](#the-rl-llm-taxonomy-tree--reviewing-synergies-between-reinforcement-learning-and-large-language-models)
    + [Yell At Your Robot: Improving On-the-Fly from Language Corrections](#yell-at-your-robot--improving-on-the-fly-from-language-corrections)
    + [EnvGen: Generating and Adapting Environments via LLMs for Training Embodied Agents](#envgen--generating-and-adapting-environments-via-llms-for-training-embodied-agents)
    + [RL-GPT: Integrating Reinforcement Learning and Code-as-policy](#rl-gpt--integrating-reinforcement-learning-and-code-as-policy)
    + [How Can LLM Guide RL? A Value-Based Approach](#how-can-llm-guide-rl--a-value-based-approach)
    + [Policy Improvement using Language Feedback Models](#policy-improvement-using-language-feedback-models)
    + [Natural Language Reinforcement Learning](#natural-language-reinforcement-learning)
    + [RLingua: Improving Reinforcement Learning Sample Efficiency in Robotic Manipulations With Large Language Models](#rlingua--improving-reinforcement-learning-sample-efficiency-in-robotic-manipulations-with-large-language-models)
    + [Hierarchical Continual Reinforcement Learning via Large Language Model](#hierarchical-continual-reinforcement-learning-via-large-language-model)
    + [True Knowledge Comes from Practice: Aligning LLMs with Embodied Environments via Reinforcement Learning](#true-knowledge-comes-from-practice--aligning-llms-with-embodied-environments-via-reinforcement-learning)
    + [SPRING: Studying the Paper and Reasoning to Play Games](#spring--studying-the-paper-and-reasoning-to-play-games)
    + [Guiding Pretraining in Reinforcement Learning with Large Language Models](#guiding-pretraining-in-reinforcement-learning-with-large-language-models)
    + [Large Language Model as a Policy Teacher for Training Reinforcement Learning Agents](#large-language-model-as-a-policy-teacher-for-training-reinforcement-learning-agents)
    + [Language and Sketching: An LLM-driven Interactive Multimodal Multitask Robot Navigation Framework](#language-and-sketching--an-llm-driven-interactive-multimodal-multitask-robot-navigation-framework)
    + [LLM Augmented Hierarchical Agents](#llm-augmented-hierarchical-agents)
    + [Reward Design with Language Models](#reward-design-with-language-models)
    + [Do As I Can, Not As I Say: Grounding Language in Robotic Affordances](#do-as-i-can--not-as-i-say--grounding-language-in-robotic-affordances)
  * [Open source RL environment](#open-source-rl-environment)

<small><i><a href='http://ecotrust-canada.github.io/markdown-toc/'>Table of contents generated with markdown-toc</a></i></small>

***

## Papers
###  The RL/LLM Taxonomy Tree: Reviewing Synergies Between Reinforcement Learning and Large Language Models
- Paper Link: [arXiv 2402.01874](https://arxiv.org/abs/2402.01874) 

- Overview:

    <img src="./images/tree.png" style="zoom: 67%;" />

    This study proposes a novel taxonomy of three main classes based on how RL and LLMs interact with each other:
    RL4LLM: RL is used to improve the performance of LLMs on tasks related to Natural Language Processing.
    LLM4RL: An LLM assists the training of an RL model that performs a task not inherently related to natural language.
    RL+LLM: An LLM and an RL agent are embedded in a common planning framework without either of them contributing to training or fine-tuning of the other
***
### Yell At Your Robot: Improving On-the-Fly from Language Corrections

- Paper Link: [arXiv 2403.12910](https://arxiv.org/abs/2403.12910) , [Homepage](https://yay-robot.github.io/)

- Framework Overview: 

    ![](./images/YAYrobotframework.jpeg)

    ​	The authors operate in a hierarchical setup where a high-level policy generates language instructions for a low-level policy that executes the corresponding skills. During deployment, humans can intervene through corrective language commands, temporarily overriding the high-level policy and directly influencing the low-level policy for on-the-fly adaptation. These interventions are then used to finetune the high-level policy, improving its future performance.

    ![](./images/YAYrobotframework2.png)

    ​	The system processes RGB images and the robot's current joint positions as inputs, outputting target joint positions for motor actions. The high-level policy uses a Vision Transformer to encode visual inputs and predicts language embeddings. The low-level policy uses ACT, a Transformer-based model to generate precise motor actions for the robot, guided by language instructions. This architecture enables the robot to interpret commands like “Pick up the bag” and translate them into targeted joint movements.

***

### EnvGen: Generating and Adapting Environments via LLMs for Training Embodied Agents

- Paper Link: [arXiv 2403.12014](https://arxiv.org/abs/2403.12014) , [Homepage](https://envgen-llm.github.io/)

- Framework Overview: 

    ![](./images/EnvGen.png)

    ​	In EnvGen framework, the authors generate multiple environments with an LLM to let the agent learn different skills effectively, with the Ncycle training cycles, each consisting of the following four steps. **Step 1:** they provide an LLM with a prompt composed of four components (*i.e*., task description, environment details, output template, and feedback from the previous cycle), and ask the LLM to fill the template and output various environment configurations that can be used to train agents on different skills. **Step 2:** they train a small RL agent in the LLM-generated environments. **Step 3:** they train the agent in the original environment to allow for better generalization and then measure the RL agent’s training progress by letting it explore the original environment. **Step 4:** they provide the LLM with the agent performance from the original environment (measured in step 3) as feedback for adapting the LLM environments in the next cycle to focus on the weaker performing skills.

***

### RL-GPT: Integrating Reinforcement Learning and Code-as-policy

- Paper Link : [arXiv 2402.19299](https://arxiv.org/abs/2402.19299) ,  [homepage](https://sites.google.com/view/rl-gpt/)

- Framework Overview: 

    <img src="./images/RL-GPT framework.png" alt="RL-GPT framework" style="zoom: 50%;" />

    ​	The overall framework consists of a slow agent (orange) and a fast agent (green). The slow agent decomposes the task and determines “which actions” to learn. The fast agent writes code and RL configurations for low-level execution.

- Method Overview:  

    ​	RL-GPT includes a slow agent and a fast agent.  The LLM can generate environment configurations (task, observation, reward, action space) for a subtask. By considering the agent’s behavior to solve the subtask, the LLM provides higher-level actions, enhancing RL’s sample efficiency.

***

### How Can LLM Guide RL? A Value-Based Approach

- Paper Link: [arXiv 2402.16181](https://arxiv.org/abs/2402.16181) , [Homepage](https://github.com/agentification/Language-Integrated-VI)

- Framework Overview: 

    ![](./images/SLINVIT.png)
    
    ​	Demonstration of the SLINVIT algorithm in the ALFWorld environment when N=2 and the tree breadth of BFS is set to k=3. The task is to “clean a cloth and put it on countertop”. The hallucination that LLM faces, i.e., the towel should be taken (instead of cloth), is addressed by the inherent exploration mechanism in our RL framework.

***

### Policy Improvement using Language Feedback Models

- Paper Link : [arXiv 2402.07876](https://arxiv.org/abs/2402.07876) 

- Framework Overview: 

    ![](./images/PILFMframework.png)


***

### Natural Language Reinforcement Learning

- Paper Link: [arXiv 2402.07157](https://arxiv.org/abs/2402.07157) 

- Framework Overview: 

    <img src="./images/NLRL.png" style="zoom:50%;" />

    ​	The authors present an illustrative example of grid-world MDP to show how NLRL and traditional RL differ for task objective, value function, Bellman equation, and generalized policy iteration. In this grid-world, the robot needs to reach the crown and avoid all dangers. They assume the robot policy takes optimal action at each non-terminal state, except a uniformly random policy at state b.

- Method Overview: 

    ​	NLRL is inspired by human learning processes. It redefines traditional RL concepts like task objectives, policies, value functions, and policy iteration using natural language space. 

***

### RLingua: Improving Reinforcement Learning Sample Efficiency in Robotic Manipulations With Large Language Models

- Paper Link: [arXiv 2403.06420](https://arxiv.org/abs/2403.06420) , [homepage](https://rlingua.github.io/)

- Framework Overview:

    <img src="./images/RLingua framework.png" alt="RLingua framework" style="zoom: 80%;" />

    ​	(a) Motivation: LLMs do not need environment samples and are easy to communicate for non-experts. However, the robot controllers generated directly by LLMs may have inferior performance. In contrast, RL can be used to train robot controllers to achieve high performance. However, the cost of RL is its high sample complexity. (b) Framework: RLingua extracts the internal knowledge of LLMs about robot motion to a coded imperfect controller, which is then used to collect data by interaction with the environment. The robot control policy is trained with both the collected LLM demonstration data and the interaction data collected by the online training policy

    <img src="./images/RLingua 2.png" alt="RLingua 2" style="zoom:50%;" />

    ​	The framework of prompt design with human feedback. The task descriptions and coding guidelines are prompted in sequence. The human feedback is provided after observing the preliminary LLM controller execution process on the robot.

- Method Overview: 

    ​	RLingua is the combination of LLM Controller & RL. It extracts the LLM's knowledge about robot motion to improve the sample efficiency of RL.  

***

### Hierarchical Continual Reinforcement Learning via Large Language Model

- Paper Link: [arXiv 2401.15098](https://arxiv.org/abs/2401.15098)

- Framework Overview:

  <img src="images/Hi_Core framework.png" alt="Hi_Core framework" style="zoom:67%;" />

  ​	The illustration of the proposed framework. The middle section depicts the internal interactions (**light gray line**) and external interactions (**dark gray line**) in Hi-Core. Internally, the CRL agent is structured in two layers: the high-level policy formulation (**orange**) and the low-level policy learning (**green**). Furthermore, the policy library (**blue**) is constructed to store and retrieve policies. The three surrounding boxes illustrate their internal workflow when the agent encounters new tasks.

- Method Overview: 

    ​	The high level LLM is used to generate a series of goals g_i . The low level is a RL with goal-directed, it needs to generate a policy in response to the goals. Policy library is used to store successful policy. When encountering new tasks, the library can retrieve relevant experience to assist high and low level policy agent.

***

### True Knowledge Comes from Practice: Aligning LLMs with Embodied Environments via Reinforcement Learning

- Paper Link: [arXiv 2401.14151](https://arxiv.org/abs/2401.14151) , [homepage](https://github.com/WeihaoTan/TWOSOME)

- Framework Overview: 

    <img src="./images/TWOSOME framework.png" style="zoom: 67%;" />

    ​	Overview of how TWOSOME generates a policy using joint probabilities of actions. The color areas in the token blocks indicate the probabilities of the corresponding token in the actions.

- Method Overview: 

    ​	The authors propose *True knoWledge cOmeS frOM practicE*(**TWOSOME**) online framework. It deploys LLMs as embodied agents to efficiently interact and align with environments via RL to solve decision-making tasks w.o. prepared dataset or prior knowledge of the environments. They use the loglikelihood scores of each token provided by LLMs to calculate the joint probabilities of each action and form valid behavior policies.

***

### SPRING: Studying the Paper and Reasoning to Play Games

- Paper Link: [arXiv 2305.15486](https://arxiv.org/abs/2305.15486), [Homepage](https://github.com/Holmeswww/SPRING)

- Framework Overview: 

    ![](./images/SPRINGframework.png)

    ​	Overview of SPRING. The context string, shown in the middle column, is obtained by parsing the LATEX source code of Hafner (2021). The LLM-based agent then takes input from a visual game descriptor and the context string. The agent uses questions composed into a DAG for chain-of-thought reasoning, and the last node of the DAG is parsed into action.

***

### Guiding Pretraining in Reinforcement Learning with Large Language Models

- Paper Link: [arXiv 2302.06692](https://arxiv.org/abs/2302.06692) , [Homepage](https://github.com/yuqingd/ellm)

- Framework Overview: 

    ![](./images/ELLM.png)
  
  ​	ELLM uses a pretrained large language model (LLM) to suggest plausibly useful goals in a task-agnostic way. Building on LLM capabilities such as context-sensitivity and common-sense, ELLM trains RL agents to pursue goals that are likely meaningful without requiring direct human intervention.

***

### Large Language Model as a Policy Teacher for Training Reinforcement Learning Agents

- Paper Link: [arXiv 2311.13373](https://arxiv.org/abs/2311.13373)

- Framework Overview: 

    <img src="./images/LLM4Teach.png" style="zoom:67%;" />

    ​	An illustration of our LLM4Teach framework using the MiniGrid environment as an exemplar. The LLM-based teacher agent responds to observations of the state provided by the environment by offering soft instructions. These instructions take the form of a distribution over a set of suggested actions. The student agent is trained to optimize two objectives simultaneously. 	The first one is to maximize the expected return, the same as in traditional RL algorithms. The other one is to encourage the student agent to follow the guidance provided by the teacher. As the student agent’s expertise increases during the training process, the weight assigned to the second objective gradually decreases over time, reducing its reliance on the teacher.

***

### Language and Sketching: An LLM-driven Interactive Multimodal Multitask Robot Navigation Framework

- Paper Link: [arXiv 2311.08244](https://arxiv.org/abs/2311.08244) 

- Framework Overview: 

    <img src="./images/LIM2N.png" style="zoom: 50%;" />

    ​	The framework contains an LLM module, an Intelligent Sensing Module, and a Reinforcement Learning Module.

***

### LLM Augmented Hierarchical Agents

- Paper Link: [arXiv 2311.05596](https://arxiv.org/abs/2311.05596) 

- Framework Overview: 

    <img src="./images/arXiv_2311_05596.png" style="zoom: 67%;" />

    ​	The LLM to guides the high-level policy and accelerates learning. It is prompted with the context, some examples, and the current task and observation. The LLM’s output biases high-level action selection.

***

### Reward Design with Language Models

- Paper Link: [arXiv 2303.00001](https://arxiv.org/abs/2303.00001)

- Framework Overview: 

    <img src="./images/arXiv230300001.png" style="zoom: 50%;" />

    ​	Depiction of the framework on the DEAL OR NO DEAL negotiation task. A user provides an example and explanation of desired negotiating behavior (e.g., versatility) before training. During training, (1) they provide the LLM with a task description, a user’s description of their objective, an outcome of an episode that is converted to a string, and a question asking if the outcome episode satisfies the user objective. (2-3) They then parse the LLM’s response back into a string and use that as the reward signal for the Alice the RL agent. (4) Alice updates their weights and rolls out a new episode. (5) They parse the episode outcome int a string and continue training. During evaluation, they sample a trajectory from Alice and evaluate whether it is aligned with the user’s objective.

***

### Do As I Can, Not As I Say: Grounding Language in Robotic Affordances

- Paper Link: [arXiv 2204.01691](https://arxiv.org/abs/2204.01691) , [Homepage](https://say-can.github.io/)

- Framework Overview: 

    <img src="./images/saycan_framework.png" style="zoom:67%;" />

    ​	Given a high-level instruction, SayCan combines probabilities from a LLM (the probability that a skill is useful for the instruction) with the probabilities from a value function (the probability of successfully executing said skill) to select the skill to perform. This emits a skill that is both possible and useful. The process is repeated by appending the skill to the response and querying the models again, until the output step is to terminate. 

    ![](./images/saycan_valuefunction.png)

    ​	A value function module (a) is queried to form a value function space of action primitives based on the current observation. Visualizing “pick” value functions, in (b) “Pick up the red bull can” and “Pick up the apple” have high values because both objects are in the scene, while in (c) the robot is navigating an empty space, and thus none of the pick up actions receive high values.

***

## Open source RL environment 

- ALFworld: https://github.com/alfworld/alfworld?tab=readme-ov-file

    <img src="./images/ALFworld.png" style="zoom:50%;" />

- Skillhack: https://github.com/ucl-dark/skillhack

    <img src="./images/skillshack.png" style="zoom: 33%;" />

- Minigrid: https://github.com/Farama-Foundation/MiniGrid?tab=readme-ov-file

    <img src="./images/door-key-curriculum.gif" style="zoom: 50%;" />

- Crafter: https://github.com/danijar/crafter?tab=readme-ov-file

    ![](./images/crafter.gif)

- OpenAI procgen: https://github.com/openai/procgen

    ![](./images/procgen.gif)

- OpenAI Multi Agent Particle Env: https://github.com/openai/multiagent-particle-envs

    <img src="./images/MultiAgentParticle.gif" style="zoom: 50%;" />

- Multi Agent RL Environment: https://github.com/Bigpig4396/Multi-Agent-Reinforcement-Learning-Environment

    <img src="./images/MultiAgentRLenvDrones.gif" style="zoom:80%;" />

- MAgent2: https://github.com/Farama-Foundation/MAgent2?tab=readme-ov-file

    <img src="./images/MAgent.gif" style="zoom: 67%;" />
