I encounter ADMM several time, did go through the derivation 2 times at
least, still I cannot write it down from scratch when a friend of mine
asked me about it. This note is to summary my understand about it.

# Derivation {#sec:derivation}

ADMM deals with an optimization problem in the following form:

$$
\begin{align*}
    & \mathop{\mathrm{minimize}}\quad f(x) + g(z) \\
    & \text{subject to } \quad Ax + Bz = c
    \end{align*}
    $$
