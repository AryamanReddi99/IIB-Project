# IIB-Project
My IIB Project for the University of Cambridge M.Eng.

## Purpose
This repository aims to observe Q-learning applications in games and identify metrics that affect model performance based on game types and game parameters.

## Nim

### Q-learner vs Random Player

<p align="center">
<img src="https://github.com/AryamanReddi99/IIB-Project/blob/master/Images/policy_not_learned.png?raw=true">  
</p>
<p align="center">
Fig 1. Q-learner playing a random player with untuned hyperparameters. Model learns slowly and does not achieve optimal policy within 10,000 games.
</p>

<p align="center">
<img src="https://github.com/AryamanReddi99/IIB-Project/blob/master/Images/policy_learned.png?raw=true">  
</p>
<p align="center">
Fig 2. Q-learner playing a random player with tuned hyperparameters. Model learns optimal policy. Large alpha and gamma allow for Q-value propagation down length of game.
</p>

## Q-learner vs Perfect Player

<p align="center">
<img src="https://github.com/AryamanReddi99/IIB-Project/blob/master/Images/vs_perfect_player_short.png?raw=true">  
</p>
<p align="center">
Fig 3. Q-learned quickly learns optimal policy when facing a perfect opponent.
</p>

<p align="center">
<img src="https://github.com/AryamanReddi99/IIB-Project/blob/master/Images/vs_perfect_player_long.png?raw=true">  
</p>
<p align="center">
Fig 4. Unlike when facing the random player, Q-learner eventually learns optimal policy for very long games (n ~ 400) when facing a perfect opponent.
</p>

