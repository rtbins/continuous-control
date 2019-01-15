[//]: # (Image References)

[image1]: https://user-images.githubusercontent.com/10624937/43851024-320ba930-9aff-11e8-8493-ee547c6af349.gif "Trained Agent"
[image2]: https://user-images.githubusercontent.com/10624937/43851646-d899bf20-9b00-11e8-858c-29b5c2c94ccc.png "Crawler"

# Continuous Control

## Deep Reinforcement Learning Nanodegree, Udacity

@rtbins

## Synopsis

### Introduction

This project uses the [Reacher](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#reacher) environment.

![Trained Agent][image1]

In this environment, a double-jointed arm can move to target locations. A reward of `+0.1` is provided for each step that the agent's hand is in the goal location. Thus, the goal of an agent is to maintain its position at the target location for as many time steps as possible.

The observation space consists of `33 variables` corresponding to `position`, `rotation`, `velocity`, and `angular velocities` of the arm. Each `action` is a vector with `four numbers`, corresponding to torque applicable to two joints. Every entry in the action vector should be a number `between -1 and 1`.

### Solving the Environment

The task is episodic, and in order to solve the environment,  an agent must get an average score of +30 over 100 consecutive episodes.

### Getting Started

1. Request (Udacity) and download the environment from one of the links below.  You need only select the environment that matches your operating system:

    - **One (1) Agent environment**
        - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Linux.zip)
        - Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher.app.zip)
        - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Windows_x86.zip)
        - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Windows_x86_64.zip)

    
    (_For Windows users_) Check out [this link](https://support.microsoft.com/en-us/help/827218/how-to-determine-whether-a-computer-is-running-a-32-bit-version-or-64) if you need help with determining if your computer is running a 32-bit version or 64-bit version of the Windows operating system.

    (_For AWS_) If you'd like to train the agent on AWS (and have not [enabled a virtual screen](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md)), then please use [this link](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Linux_NoVis.zip) (version 1) or [this link](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Linux_NoVis.zip) (version 2) to obtain the "headless" version of the environment.  You will **not** be able to watch the agent without enabling a virtual screen, but you will be able to train the agent.  (_To watch the agent, you should follow the instructions to [enable a virtual screen](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md), and then download the environment for the **Linux** operating system above._)

2. Place the file in the root folder, and unzip (or decompress) the file.

3. Create (and activate) a new environment with Python 3.6.

- __Linux__ or __Mac__: 

```bash
conda create --name drlnd python=3.6
source activate drlnd
```

- __Windows__: 

```bash
conda create --name drlnd python=3.6 
activate drlnd
```

4. Clone the repository, and navigate to the `python/` folder.  Then, install several dependencies.

```bash
git clone https://github.com/udacity/deep-reinforcement-learning.git
cd deep-reinforcement-learning/python
pip install .
```

5. Create an [IPython kernel](http://ipython.readthedocs.io/en/stable/install/kernel_install.html) for the `drlnd` environment. 

```bash
python -m ipykernel install --user --name drlnd --display-name "drlnd"
```

6. Before running code in a notebook, change the kernel to match the `drlnd` environment by using the drop-down `Kernel` menu.

### Instructions

The project consists of 5 files

- Continuous_Control.ipynb - this is jupyter notebook containing all analysis.
- ddpg_agent.py - contains the implementation for a DDPG agent
- model.py - the neural net model
- checkpoint_actor.pth - saved trained model for actor(pytorch)
- checkpoint_critic.pth - saved trained model for critic(pytorch)
- config.py - configs and different hyperparameters for training.
- memory.py - implementation class for replay buffer.
