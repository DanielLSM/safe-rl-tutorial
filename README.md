# safe-rl-tutorial
This repository provides the code source of the tutorial to be held on safe reinforcement learning. The key concepts of the tutorial are following: 
- (1) Understanding a simple DRL algorithm: Deep Deterministic Policy Gradient (DDPG)
- (2) Training said algorithm in a couple of environments, 2 benchmarks and one altered environment for safe reinforcement learning
- (3) Create a shield, to enable safe training for the agent!

----------
# Instalation

##  Ubuntu 20.04/18.04  (tested)

### Requirements:
- Anaconda 3

##  Instructions 

0) Open a terminal
1) Clone the repository 
```
cd ~
git clone https://github.com/DanielLSM/safe-rl-tutorial
```
2) Move to the repository in your system
```
cd safe-rl-tutorial
```
3) Install the anaconda environment
```
conda env create -f safe-rl.yml
```
4) Load the anaconda environment
```
conda activate safe-rl
```

## References

[1] Teodor Mihai Moldovan, Pieter Abbeel,
  Safe exploration in Markov decision processes
  [[ref]](https://arxiv.org/abs/1205.4810/)

[2] Javier García,Fernando Fernández  
A Comprehensive Survey on Safe Reinforcement Learning
  [[ref]](https://www.jmlr.org/papers/volume16/garcia15a/garcia15a.pdf
)

[3] Mohammed Alshiekh, Roderick Bloem, Rudiger Ehlers, Bettina Konighofer, Scott Niekum, Ufuk Topcu,
Safe Reinforcement Learning via Shielding
  [[ref]](https://arxiv.org/abs/1708.08611
)

[4] Rémi Munos, Thomas Stepleton, Anna Harutyunyan, Marc G. Bellemare
Safe and Efficient Off-Policy Reinforcement Learning
  [[ref]](https://arxiv.org/abs/1606.02647)

[5] Lillicrap, Timothy P.
Continuous control with deep reinforcement learning.
  [[ref]](https://arxiv.org/abs/1509.02971)

2 ideas: use inverted car pendulum with a trust region. use lunar landing with certain thurst