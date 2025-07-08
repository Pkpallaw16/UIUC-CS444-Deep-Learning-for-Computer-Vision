
# Assignment 5: Deep Reinforcement Learning (DRL)

In this assignment, we train DRL agents to play the [Atari Breakout](https://www.gymlibrary.dev/environments/atari/breakout/) game.  
Specifically, we implement and compare three RL agents: deep Q-network (DQN), double DQN (DDQN), and DDPG.

---

## 🎮 DQN and DDQN

The action space of Breakout is discrete:

| Num | Action  |
|:---:|---------|
| 0   | No Op   |
| 1   | Fire    |
| 2   | Right   |
| 3   | Left    |

DQN and DDQN are inherently suitable for this task.  
We use standard RL training techniques like a replay buffer, epsilon-greedy exploration, and target networks.  
DDQN improves on DQN by decoupling action selection and evaluation, reducing overestimation in Q-values.

### Results:
| Agent | Mean Reward |
|-------|-------------|
| **DQN**  | 6.96        |
| **DDQN** | 14.65       |

- Highest evaluation reward achieved by **DDQN**: 14.65 at episode 2499
- Learning rate at that point: ~1.64e-6
- Replay memory length: 892,262

---

## 🧪 Analysis

- **DQN**: Achieved highest reward of 6.96 after 2500 episodes. Learning was slow and plateaued for ~1200 episodes.
- **DDQN**: Achieved highest reward of 14.65 after ~1800 episodes — much faster and more stable than DQN.
- **Implementation Notes**:
  - Used Huber loss to update the target network.
  - In `memory.py`, ensured actions were moved to CPU before appending to replay buffer to avoid GPU–CPU conflicts.


---

## 🤖 DDPG

Although DDPG is designed for continuous action spaces, we adapted it by treating its 4-dimensional output as scores for the 4 discrete actions, and chose the action with the highest score at each step.

---

## 📈 Plots

- Mean evaluation reward curve for **DDQN** — highest reward at episode 2499.
![alt text](image.png)
---

## 👥 Team

- **Name(s):** Pallaw Kumar, Neha Jain
- **NetID(s):** pallawk2, nehaj4

---
