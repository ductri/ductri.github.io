---
layout: post
title: xx
date: 2024-02-09 21:23:00
description: Application of sample complexities bound on matrix completion 
tags: matric completion, nmf, 
categories: note
published: false
usemathjax: false
pdf_dir: 
---

A summary of my strategy on solving programming problems on leetcode and the similar.
<!--more-->

y = A(x) + n

We want to utilize a pretrained diffusion model to solve this inverse problem: given y, find x.
The pretrained diffusion model will give us a sampling process p(x). By Bayes rule, we get

P(x|y) ~ P(y|x) + P(x).

Note that this description is very general and has nothing to do with diffusion.

P(x) is accessible by pretrained diffusion. The remaining is P(y|x).
How did people deal with this term?

For example, Song in the medical imaging paper solves it using the following idea:

- Using linear model A(x) = Ax
- assume a specific structure on A (proposition 1)
- They solve the inverse problem at every step of the diffusion inference process?
- I don't see any guarantee
- 

