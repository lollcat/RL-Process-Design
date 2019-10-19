# Deep Reinforcement Learning for Design of Chemical Engineering Processes

# Table of Contents
1. [Hybrid_Action_Environment_&_P-DQN_Agent](#Hybrid_Action_Environment_&_P-DQN_Agent)
2. [Discrete_Action_Environment_&_DQN_Agent](#Discrete_Action_Environment_&_DQN_Agent)

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
