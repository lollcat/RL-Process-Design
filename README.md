# Deep Reinforcement Learning for Design of Chemical Engineering Processes

# Table of Contents
1. [Idea_Overview] (#Idea_Overview)
2. [Hybrid_Action_Environment_&_P-DQN_Agent](#Hybrid_Action_Environment_&_P-DQN_Agent)
3. [Discrete_Action_Environment_&_DQN_Agent](#Discrete_Action_Environment_&_DQN_Agent)
4. [Future_Plans](#Future_Plans)

## Idea_Overview
Reinfocement Learning can be used to design chemical engineering processes in a simulator
(Add picture of whole of chemical engineering phrased as RL problem

## Hybrid_Action_Environment_&_P-DQN_Agent
Discrete Action Space: Select Light Key For Distillation
Continuous Action Space: Select Split Factor
State: Current Stream Composition
Nested Reward: Inner = Pay Back Period
               Outer = Product Purity Streams, Column Cost
P-DQN_Agent:
Based on https://arxiv.org/abs/1810.06394

## Discrete_Action_Environment_&_DQN_Agent:
PFR or CSTR
PFR or CSTR and cool
Distillation only selecting compound with assumed split rates

## Future_plans
Add planning: E.g. Monte Carlo Tree Search

Extend problems to be closer to full chem eng agent as described in Idea Overview

## Other Notes
See specific folders for more depth on each environment and agent

