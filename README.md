# Deep Reinforcement Learning for Design/Synthesis of Chemical Engineering Processes

# Final year chemical engineering thesis by Laurence Midgley and Michael Thomson

## See [Report](https://github.com/lollcat/RL-Process-Design/blob/master/Thesis%20Report.%20RL%20for%20Process%20Synthesis.pdf) for details

A good starting point to get a basic idea of RL for process synthesis is provided in [this colab notebook](https://colab.research.google.com/github/lollcat/RL-Process-Design/blob/master/Discrete/PFR%20or%20CSTR/PFR_or_CSTR.ipynb). In this example we look at a simple problem where an RL agent designs a sequence of reactors through making the binary decision of choosing to add a PFR or CSTR (of fixed volume and conditions) to the "current" process stream (which is either the starting stream, or the stream coming out of the previous reactor). The full specification for this problem is provided in our [report](https://github.com/lollcat/RL-Process-Design/blob/master/Thesis%20Report.%20RL%20for%20Process%20Synthesis.pdf).


## Abstract

This thesis demonstrated, for the first time, that reinforcement learning (RL) can be applied to chemical engineering process synthesis (sequencing and design of unit operations to generate a process flowsheet). 
Two case studies were used, with simple toy process synthesis problems for the proof of concept.


The first case study was a toy reactors-in-series sequencing problem with only two actions (“PFR” or “CSTR”) and a known solution. 
The RL agent applied deep-Q learning, which is a simple well-known variant of RL. 
The agent was able to find the optimal configuration of reactors. 
The application of high level RL coding libraries, together with the development of visualisation tools in this case study makes the example accessible to chemical engineers without advanced knowledge of RL.

The second case study was a toy distillation column (DC) train synthesis problem.
Solving this was more complex due to the branching structure of the DC train and the hybrid action space containing both discrete and continuous actions. Consequently, this case study began to approach the open-ended domains in which RL may have an advantage over conventional approaches.


In this case study, a P-DQN agent that could produce both discrete and continuous actions was used. 
The agent was able to learn and outperform multiple heuristic designs, often creating unexpected configurations for the DC trains. These counterintuitive results are an indicator of the potential for RL to generate novel process designs that would not necessarily be found by conventional methods.

In an exploration of further developments within RL’s application to process synthesis:
(1) We compared RL and conventional process synthesis techniques. 
RL has the potential to be superior due to its ability to learn and generalise within open-ended problems, while taking advantage of computers’ ability to analyse large amounts of data and rapidly interact with simulations. 
(2) We proposed an expansion of the RL agent’s action space used in the simple case studies to a more general action space for process synthesis that could be used to generate complete chemical engineering processes. 
(3) We highlighted that RL is well suited to process synthesis problems
governed by general equations/laws like thermodynamics. 
(4) We critiqued the model free RL approach that we used and recommended that model is given RL should rather be used in future developments of RL’s application to process synthesis, as the model free RL made the problem unnecessarily complex. 
(5) We proposed benchmarks for future RL research on process synthesis.


In the future, RL for process synthesis is well suited to take advantage of improvements in chemistry and physics simulation. 
This thesis hopes to stimulate research within this area – with the long-term goal of an RL agent creating novel, profitable processes.


## Please cite us at:

@misc{laurence_midgley_2019_3556549,
  author       = {Laurence Midgley and
                  Michael Thomson},
  title        = {{Reinforcement learning for chemical engineering 
                   process synthesis}},
  month        = nov,
  year         = 2019,
  note         = {{Final year undergraduate chemical engineering 
                   thesis}},
  publisher    = {Zenodo},
  version      = {v1.0},
  doi          = {10.5281/zenodo.3556549},
  url          = { https://doi.org/10.5281/zenodo.3556549 }
}

## Further Notes
I have extended this work further in "Distillation Gym" see [code repository](https://github.com/lollcat/DistillationTrain-Gym) and [paper](https://arxiv.org/abs/2009.13265). Feel free to email me at laurence.midgley@gmail.com if you are interested! 
