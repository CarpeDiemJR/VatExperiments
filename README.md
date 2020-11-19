<script type="text/javascript" src="http://cdn.mathjax.org/mathjax/latest/MathJax.js?config=default"></script>

# Vat Experiments

VAT, the **Vitual Adversarial Training** model, 
could be refered in _Virtual Adversarial Training: a Regularization Method for Supervised and Semi-Supervised Learning_. \[[Article](http://arxiv.org/abs/1704.03976), [Tensorflow 1.0 implementation](https://github.com/takerum/vat_tf)\]

## Goal of this project

I want to rewrite the code in Tensorflow 2.0 manner and then show that imbalanced dataset has a significant impact on VAT model. To save the computational cost, the neaural network used here would be dense nets and dataset would be MNIST.

## Context
Objective function, <img src="https://latex.codecogs.com/svg.latex?sin(x)" title="sin(x)" />