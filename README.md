# Gomoku_Alphazero
Simple implementation of AlphaZero algorithm in Gomoku

## Requirements
- pygame
- pytorch
- numpy

## Changes from original paper
- Didn't use residual layer, used 5 convolutional layer
- Didn't use Dirichlet noise, just used temperature parameter 1
- Learning rate does not decreases

## References
- Mastering Chess and Shogi by Self-Play with a General Reinforcement Learning Algorithm
- Mastering the game of Go without human knowledge
