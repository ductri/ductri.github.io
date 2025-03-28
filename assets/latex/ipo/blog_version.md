---
author:
- |
  Tri Nguyen\
  \
  `nguyetr9@oregonstate.edu`\
  \
bibliography: refs.bib
title: |
  LLM Alignment Fine-tuning via Human Preference:\
  DPO vs IPO
---

Large language model (LLM) alignment is a crucial step in increasing
LLM's usability and safety. This topic has been received a lot of
attentions, resulting in very interesting development over the last 5
years. I'm particularly interested in a line of development including 3
works:

-   **Reinforcement Learning from Human Feedback (RLHF)**
    [@ouyang2022training; @christiano2017deep; @bai2022constitutional; @stiennon2020learning]
    from OpenAI: learning score model, using RL to perform alignment

-   **Direct preference optimization (DPO)** [@rafailov2023direct] from
    Stanford: learning score model, alignment is "automatically"
    obtained.

-   **IPO (or $\Psi$PO )** [@azar2023general] from Google DeepMind: no
    need for learning score model, at all :))

is a very nice build-up as one addresses the previous work's issues
and reduce learning process' complexity. An overview comparing the three
method is shown in Figure [1](#fig:overview){reference-type="ref"
reference="fig:overview"}.

<figure id="fig:overview">

<figcaption>An overview comparison of RLHF, DPO, and IPO.</figcaption>
</figure>

In the first 2 works, both RLHF and DPO's idea are elegant and simple to
grasp. This is however not the case for IPO. When I first read the IPO
paper the whole thing screams "ad-hoc" and *unnecessarily-complicated*
;). Their derivation is a bit confusing, their empirical loss looks
counter-intuitive, they don't even have any LLM alignment experiments.
But the fact that IPO method works well empirically (from my own
experience as well as from other papers) bothers me quite a lot. After
spending some effort to examining this method more carefully, IPO's idea
turns out to be quite nice and very clever. If you don't get it from
skimming through the paper (like I did), I hope this blog post can
convince you to have another look at this method.

# LLM alignment {#sec:introduction}

LLMs obtained after a pre-training step over vast un-labeled datasets
possesses an amazingly ability of natural-looking text completions.
However, due to the nature of unsupervised training, the resulting
models exhibit many undesired behaviors (via the generation), including
being unhelpful, biased, sexist, racism, hallucination.

LLM alignment aims to address this issue by steering model's behavior
toward the desired characterizations using following formulation
$$\label{eq:original_obj}
\mathop{\mathrm{\text{maximize}}}_{\pi} \quad  \mathop{\mathbb{E}}_{\bm{x} \sim \mathcal{D}} \left[  \mathop{\mathbb{E}}_{\bm{y} \sim \pi} \left[ s(\bm{x}, \bm{y})  - \beta \Dkl{\pi}{\pi_{\rm ref}}\right]
\right]$$ Here, the so-called *score function* $s(\bm{x}, \bm{y})$ is
assumed to produce a scalar value indicating how strongly the response
$\bm{y}$ is aligned with the desired characteristic given the prompt
$\bm{x}$. We wish find a policy (a language model) $\pi$ such that it
retains most of the good text-generation capability of $\pi_{\rm ref}$,
and at the same time producing responses $\bm{y}$ to maximize the score
function. The balance between 2 objectives is controlled by $\beta$. A
too small $\beta$ might lead to overly optimized $\pi$ that loses the
generally good text generation of $\pi_{\rm ref}$, while too large
$\beta$ prevents $\pi$ from adjusting toward better alignment. The
reference policy $\pi_{\rm ref}$ is can be seen as a good
initialization.

An obvious barrier in solving
[\[eq:original_obj\]](#eq:original_obj){reference-type="eqref"
reference="eq:original_obj"} is the unknown $s(\bm{x}, \bm{y})$. It is
very non-trivial to hand-crafting the score function
$s(\bm{x}, \bm{y})$. Instead, a popular approach is to learn
$s(\bm{x}, \bm{y})$ using pairwise preference datasets. A sample of such
dataset consists of a tuple $(\bm{x}, \bm{y}_1, \bm{y}_2, c)$ where
$\bm{x}$ is a prompt, $\bm{y}_1, \bm{y}$ are 2 possible responses (a
continuation of prompt $\bm{x}$). These three elements are sent to a
human annotator, who assigns a label $c\in \left\{1,2\right\}$ to
indicate which response is more preferred with the alignment objective.
For instance, $c=1$ implies that $\bm{y}_1$ is preferred over
$\bm{y}_2$, denoting as $\bm{y}_1 \succ \bm{y}_2$.

Different methods propose to solve
[\[eq:original_obj\]](#eq:original_obj){reference-type="eqref"
reference="eq:original_obj"} in different ways but they all share the
same principle: using the preference data to "infer" the unknown
function $s(\bm{x}, \bm{y})$. As the only available supervised signal is
the preference dataset, it is necessary to assume certain specification
to relate the unknown score function $s(\bm{x}, \bm{y})$ and the
collected pairwise preference data.

::: tcolorbox
$$\label{eq:general_spec}
\text{Unknown score function } s(\bm{x}, \bm{y}) \quad \xleftrightarrow{\quad \text{\color{red}certain specification}\quad } \quad \text{Preference data } (\bm{x}, \bm{y}_1, \bm{y}_2, c)$$
:::

In the following, we will go through different realizations of this
relations, leading to 3 different popular techniques: RLHF, DPO, and
IPO.

# RLHF and DPO {#sec:rlhf_and_dpo}

## RLHF {#sub:rlhf}

The structure of the optimization problem
[\[eq:original_obj\]](#eq:original_obj){reference-type="eqref"
reference="eq:original_obj"} is very much a typical reinforcement
learning (RL) problem, except that the score function
$s(\bm{x}, \bm{y})$ is unknown. Naturally, if one can learn
$s(\bm{x}, \bm{y})$, then an off-the-shelf RL method can be deployed to
solve [\[eq:original_obj\]](#eq:original_obj){reference-type="eqref"
reference="eq:original_obj"}. This is the very idea proposed by RLHF.

To learn $s(\bm{x}, \bm{y})$, RLHF assumes the Bradley-Terry (BT) model
[@bradley1952rank] for the preference data generation, which
hypothesizes that
$$\textsf{Pr}\left(\bm{y}_1 \succ \bm{y}_2 \mid \bm{x}\right) \triangleq \textsf{Pr}\left(c=1 \mid \bm{x}, \bm{y}_{1}, \bm{y}_{2}\right) = \dfrac{\exp(s(\bm{x}, \bm{y}_{1}))}{\exp(s(\bm{x}, \bm{y}_{1})) + \exp(s(\bm{x}, \bm{y}_{2}))}
= \sigma(s(\bm{x}, \bm{y}_{1}) - s(\bm{x}, \bm{y}_{2}))$$ Since the
score function gives a higher value for a better aligned response, it is
more likely that that response is preferred over the other.

We can see that this assumption specifies the relation as mentioned in
[\[eq:general_spec\]](#eq:general_spec){reference-type="eqref"
reference="eq:general_spec"}. If we further assume the true
$s(\bm{x}, \bm{y})$ belong to certain class of deep neural network (such
as an LLM-based network), then the true score function can be recovered
using MLE: $$\begin{aligned}
\label{eq:score_learning}
& \mathop{\mathrm{\text{minimize}}}_{\boldsymbol \theta} \quad -\mathop{\mathbb{E}}_{(\bm{x}, \bm{y}_1, \bm{y}_2), c \; \sim \mathcal{D}} \left[\mathcal{L}_{\rm logistic}(\sigma(s_{\boldsymbol \theta}(\bm{x}, \bm{y}_1) - s_{\boldsymbol \theta}(\bm{x}, \bm{y}_2)), c)\right]
\end{aligned}$$ where
$$\mathcal{L}_{\rm logistic}(p, c) = \mathbb{I}[c=1] \log p + \mathbb{I}[c=2] \log \left( 1-p\right).$$
In practice, $s_{\boldsymbol \theta}(\bm{x}, \bm{y})$ can be
parameterized using another large language model. After obtaining the
optimal solution $\boldsymbol \theta^{\star }$ to
[\[eq:score_learning\]](#eq:score_learning){reference-type="eqref"
reference="eq:score_learning"}, the alignment fine-tuning is performed
by plugging $s_{\boldsymbol \theta^{\star }}$ into
[\[eq:original_obj\]](#eq:original_obj){reference-type="eqref"
reference="eq:original_obj"} and an off-the-shelf RL techniques, such as
PPO [@schulman2017proximal] is invoked to solve
[\[eq:original_obj\]](#eq:original_obj){reference-type="eqref"
reference="eq:original_obj"}.

This approach while being straightforward, suffers from several
technical difficulties:

-   A 2-stage training pipeline is complex and prone to error
    accumulation.

-   The use of RL require intensive and careful hyperparameter tuning.

These challenges are the main motivations for the development of DPO.

## DPO {#sub:dpo}

DPO improves upon RLHF by eliminating the RL step. In particular, DPO's
authors realizes that under the same preference model (BT model), the
relation between preference label and score function only depends on the
relative difference in score, not the absolute score values. This
enables them to use a clever trick to re-parameterize the score function
via an optimal policy, effectively eliminate the needs of deploying RL.

Notice that the objective in
[\[eq:original_obj\]](#eq:original_obj){reference-type="eqref"
reference="eq:original_obj"} can be expressed as: $$\begin{aligned}
\mathop{\mathbb{E}}_{\bm{x}} \left[ \mathop{\mathbb{E}}_{\bm{y} \sim \pi(\cdot \mid \bm{x})} \left[  s(\bm{y}, \bm{x}) - \beta \Dkl{\pi}{\pi_{\rm ref}}\right] \right] 
&= \mathop{\mathbb{E}}_{\bm{x}} \left[  \Dkl{\pi}{\dfrac{1}{Z(\bm{x})} \pi_{\rm ref}(\bm{y} \mid \bm{x})\exp\left(\beta^{-1} s(\bm{y}, \bm{x})\right) } \right] + \text{const}
\end{aligned}$$ where $Z(\bm{x})$ is an intractable normalizing factor.
As KL-divergence reaches minimum value at $0$, this expression suggests
an optimal solution $\pi^{\star }$ as
$$\pi^{\star }(\bm{y} \mid \bm{x}) = \dfrac{1}{Z(\bm{x})} \pi_{\rm ref}(\bm{y} \mid \bm{x}) \exp \left( \beta^{-1}  s(\bm{y}, \bm{x}) \right),$$
which equivalently implies
$$s(\bm{y}, \bm{x}) = \beta \log \dfrac{\pi^{\star }(\bm{y} \mid \bm{x})}{\pi_{\rm ref}(\bm{y} \mid \bm{x})} + \beta \log Z(\bm{x})$$
This identity establishes a relation between an arbitrary score function
$s(\bm{x}, \bm{y})$ and a corresponding optimal policy
$\pi^{\star }(\bm{y}\mid \bm{x})$ with respect to that score function.
It is not very useful by itself due to the intractable factor
$Z(\bm{x})$. However, the relative score difference, which is all that
matters, is **independent** of $Z(\bm{x})$: $$\label{eq:qoblq}
s(\bm{x}, \bm{y}_1) - s(\bm{x}, \bm{y}_2) = \beta \log \dfrac{\pi^{\star }(\bm{y}_1\mid \bm{x})}{\pi_{\rm ref}(\bm{y}_1 \mid \bm{x})} - \beta \log \dfrac{\pi^{\star }(\bm{y}_2\mid \bm{x})}{\pi_{\rm ref}(\bm{y}_2 \mid \bm{x})}.$$
To be more concise, define
$$h_{\boldsymbol \theta}(\bm{x}, \bm{y}_1, \bm{y}_2)=\beta \log \dfrac{\pi_{\boldsymbol \theta }(\bm{y}_1\mid \bm{x})}{\pi_{\rm ref}(\bm{y}_1 \mid \bm{x})} - \beta \log \dfrac{\pi_{\boldsymbol \theta }(\bm{y}_2\mid \bm{x})}{\pi_{\rm ref}(\bm{y}_2 \mid \bm{x})}.$$
then equation [\[eq:qoblq\]](#eq:qoblq){reference-type="eqref"
reference="eq:qoblq"} can be shorten as $$\label{eq:h_theta_cond}
s(\bm{x}, \bm{y}_1) - s(\bm{x}, \bm{y}_2) = h_{\boldsymbol \theta}(\bm{x}, \bm{y}_1, \bm{y}_2).$$
This condition plays a key in ensuring policy
$\pi_{\boldsymbol \theta }$ being the optimal solution to the original
alignment formulation
[\[eq:original_obj\]](#eq:original_obj){reference-type="eqref"
reference="eq:original_obj"}. Particularly, it is shown in DPO that any
policy $\pi_{\boldsymbol \theta}$ satisfying
condition [\[eq:h_theta_cond\]](#eq:h_theta_cond){reference-type="eqref"
reference="eq:h_theta_cond"} is an optimal solution to the alignment
formula in
[\[eq:original_obj\]](#eq:original_obj){reference-type="eqref"
reference="eq:original_obj"}, up to some trivial ambiguity.

With this insight, DPO proposed to use
$h_{\boldsymbol \theta}(\bm{x}, \bm{y}_1, \bm{y}_2)$ to parameterize the
relative score difference between 2 responses $\bm{y}_1, \bm{y}_2$. As
before, they employ the BT model and use MLE to derive the loss
function: $$\begin{aligned}
\label{eq:score_learning_dpo}
& \mathop{\mathrm{\text{minimize}}}_{\boldsymbol \theta} \quad -\mathop{\mathbb{E}}_{(\bm{x}, \bm{y}_1, \bm{y}_2), c \; \sim \mathcal{D}} \left[\mathcal{L}_{\rm logistic}\left(\sigma(h_{\boldsymbol \theta}(\bm{x}, \bm{y}_1, \bm{y}_2)), c\right)\right].
\end{aligned}$$ With this parameterization, an optimal solution
$\boldsymbol \theta^{\star }$ to
[\[eq:score_learning_dpo\]](#eq:score_learning_dpo){reference-type="eqref"
reference="eq:score_learning_dpo"} gives us an optimal policy
$\pi_{\boldsymbol \theta^{\star }}$ to
[\[eq:original_obj\]](#eq:original_obj){reference-type="eqref"
reference="eq:original_obj"} simultaneously.

## The Common Theme {#sub:the_common_theme}

In terms of modeling, both RLHF and DPO relies on the same
specification:

::: tcolorbox
:::

There are 2 factors in this specification:

-   The BT model is used to relate the score function and the preference
    data

-   The score function is assumed to belong to certain known hypothesis
    class $\mathcal{F}$, either an arbitrary neural network as in RLHF,
    or a structured neural network as in DPO.

The combination of the two enables learning $s(\bm{x}, \bm{y})$ using
preference data.

#### Drawback.

-   The preference data relies on BT model. BT model is intuitive and is
    successfully deployed in many domains, such as economy, however, it
    is still restrictive and who knows if real data generation truly
    follows it.

-   The particular functional form of BT model, i.e., the sigmoid
    function makes it computationally difficult to model deterministic
    cases. Specifically, to have
    $\textsf{Pr}\left(c=1 \mid \bm{x}, \bm{y}_1, \bm{y}_2\right) \to 1$,
    the quantity $s(\bm{x}, \bm{y}_1) - s(\bm{x}, \bm{y}_2)$ in RLHF, or
    $h_{\boldsymbol \theta}(\bm{x}, \bm{y}_1, \bm{y}_2)$ in DPO needs to
    reach $\infty$. This behavior causes a particular detrimental
    effect: worsening the reward hacking in the second phase in RLHF, or
    overfitting in DPO where the learned policy drifting arbitrarily far
    away from $\pi_{\rm ref}$ regardless of $\beta$.

-   And subtly, the alignment fine-tuning task is cast into a score
    learning problem. This is in turn solved indirectly. What we wish to
    estimate is in the "logits" space, i.e.,
    $\widehat{h}(\bm{y}_1, \bm{y}_2, \bm{x}) \approx s(\bm{y}_1, \bm{x}) - s(\bm{y}_2, \bm{x})$
    but what the criterion enforcing is in the probability space, i.e.,
    $\sigma(\widehat{h}(\bm{y}_1, \bm{y}_2, \bm{x})) \approx \sigma(s(\bm{y}_1, \bm{x}) - s(\bm{y}_2, \bm{x}))$.
    A small mismatch in the later could result in a much large mismatch
    in the former estimation, which affects the ultimate alignment task.
    This is, to me, the major instability issue of RLHF and DPO.

And these drawback lead to the development of IPO.

# IPO {#sec:ipo}

The method in RLHF/DPO can be seen to boil down to a binary
classification problem: Given a sample $(\bm{x}, \bm{y}_1, \bm{y}_2)$,
one wish to predict if the label is $1$ or $2$. Their loss functions are
very intuitive following this view. Take DPO for example, if annotator
is very certain that the label
$c=1 \Leftrightarrow \bm{y}_1 \succ \bm{y}_2$, the score gap
$\bm{h}_{\boldsymbol \theta}(\bm{y}_1, \bm{y}_2, \bm{x})$ between the 2
responses should be large. The loss functions DPO
[\[eq:score_learning_dpo\]](#eq:score_learning_dpo){reference-type="eqref"
reference="eq:score_learning_dpo"} promotes this by increasing the
estimated likelihood
$\sigma(h_{\boldsymbol \theta}(\bm{y}_1, \bm{y}_2, \bm{x}))$. Similar
mechanism can be observed in RLHF as well.

This very intuitive idea might have spoiled me so much that when I first
encountered the IPO paper, every step of it felt wrong. Let me
articulate. Let's look at their final loss: $$\begin{aligned}
{2}
    & \mathop{\mathrm{\text{minimize}}}_{\boldsymbol \theta} \quad && \mathop{\mathbb{E}}_{\bm{x}, \bm{y}_w, \bm{y}_{\ell }} \left[  (h_{\boldsymbol \theta}(\bm{y}_{w}, \bm{y}_{\ell }, \bm{x}) - 0.5)^2 \right],
\end{aligned}$$ where notation $\bm{y}_{w}, \bm{y}_{\ell }$ is an
equivalent form of using the label $c\in\left\{1,2\right\}$ and imply
$\bm{y}_w \succ \bm{y}_{\ell }$. This is nothing special and everyone
uses it.

Now come to the unintuitive parts:

-   Why regression? How can it turn a classification to a regression
    problem?

-   And why regressing toward $0.5$?

-   Furthermore, IPO claims that they even **don't need to learn the
    score function**. Then how do they do alignment at all!!!

These baffling points have been in my mind for a while. Unfortunately,
their derivation in the paper didn't help much. Recently, I have some
troubles comparing against this IPO method, and thus I've spent some
more time on it. Turned out, their solution is more elegant than I
thought.

## IPO's Loss Derivation {#sub:ipo_s_loss_derivation}

Note that IPO's goal is to perform the alignment fine-tuning task via
the same formulation
[\[eq:original_obj\]](#eq:original_obj){reference-type="eqref"
reference="eq:original_obj"} as RLHF and DPO do. Unlike previous methods
where there is no structure on $s(\bm{x}, \bm{y})$, IPO assumes the
following (unknown) score function: $$\label{eq:s_v}
s^{\natural}(\bm{x}, \bm{y}) =  \mathop{\mathbb{E}}_{\bm{y}' \sim \mu(\cdot \mid \bm{x})} \left[ v(\bm{y}, \bm{y}', \bm{x}) \right].$$
The scalar-valued function $v(\bm{y}, \bm{y}', \bm{x})\in [0,1]$ denotes
the probability of $\bm{y} \succ \bm{y}'$. This score function can be
seen as a quantification of how a response $\bm{y}$ is preferred on
average with respect to a predefined policy $\mu$. For now, $\mu$ is
just an arbitrary policy. Using this score function, the alignment task
is to find a policy $\pi^{\star }$ within the proximity of
$\pi_{\rm ref}$ and be better than policy $\mu$ as much as possible. In
essence, the goal is quite the same, except that we have certain
structure on the score function to work with. As we will see, using this
unknown score function, IPO cleverly perform alignment without
explicitly learning it.

Note that in the paper, IPO denote
$v(\bm{y}, \bm{y}', \bm{x}) \triangleq p( \bm{y} \succ \bm{y}' \mid \bm{x} )$.
Although using $p( \bm{y} \succ \bm{y}' \mid \bm{x} )$ can be intuitive,
I feel much comfortable using $v(\bm{y}, \bm{y}', \bm{x})$ to explicitly
remind myself that it is nothing but a special unknown function. As $v$
denote a probability of preference, we require it to satisfy:
$$\begin{align}
&v(\bm{y}, \bm{y}', \bm{x}) \in [0, 1], \; \forall  \bm{y}, \bm{y}', \bm{x} \label{eq:ipo_boundedness}\\
&v(\bm{y}, \bm{y}, \bm{x}) = 0.5, \; \forall  \bm{y}, \bm{x} \label{eq:ipo_self_compare}\\
&v(\bm{y}, \bm{y}', \bm{x}) = 1 - v(\bm{y}', \bm{y}, \bm{x}) \label{eq:ipo_symmetric}
\end{align}$$ Condition
[\[eq:ipo_boundedness\]](#eq:ipo_boundedness){reference-type="eqref"
reference="eq:ipo_boundedness"} is to ensure valid probabilities,
condition
[\[eq:ipo_self_compare\]](#eq:ipo_self_compare){reference-type="eqref"
reference="eq:ipo_self_compare"} and
[\[eq:ipo_symmetric\]](#eq:ipo_symmetric){reference-type="eqref"
reference="eq:ipo_symmetric"} are to enforce the physical meaning of
pairwise preference. With these notations, IPO's specification on the
relation between preference labels and score function is as follows

::: tcolorbox
:::

Notice that $v$ is still unknown but IPO don't aim to learn $v$. The
new, and important requirement is that the pair of responses are assumed
to be drawn from the same behavior policy $\mu$. Recall that $\mu$ plays
a role in defining the score function. This seems to be a limitation.
However, it is possible that particular choice of $\mu$ does not change
the optimal policy much (or at all).

As in DPO, IPO enforces the condition
[\[eq:h_theta_cond\]](#eq:h_theta_cond){reference-type="eqref"
reference="eq:h_theta_cond"} to ensure optimal policy:
$$h_{\boldsymbol \theta}(\bm{x}, \bm{y}_1, \bm{y}_2) = s(\bm{x}, \bm{y}_1) - s(\bm{x}, \bm{y}_2) = \mathop{\mathbb{E}}_{\bm{y} \sim \mu}[v(\bm{y}_1, \bm{y}, \bm{x}) - v(\bm{y}_2, \bm{y}, \bm{x})],$$

which can be realized by solve an optimization problem:
$$\begin{aligned}
{2}
    \label{eq:ipo_first_square}
    & \mathop{\mathrm{\text{minimize}}}_{\boldsymbol \theta} \quad && \mathop{\mathbb{E}}_{\bm{x}, \bm{y}_1, \bm{y}_2\sim \mu} \left[ \left(  h_{\boldsymbol \theta}(\bm{x}, \bm{y}_1, \bm{y}_2)  - \mathop{\mathbb{E}}_{\bm{y} \sim \mu}\left[v(\bm{y}_1, \bm{y}, \bm{x}) - v(\bm{y}_2, \bm{y}, \bm{x}) \right] \right) ^2\right]
\end{aligned}$$ Notice that the regression target is unknown. However,
by exploiting 2 key factors in their specification: (i) the pair of
responses are drawn from the same policy $\mu$ which is also used in
defining the score function, and (ii) the structure of
$h_{\boldsymbol \theta}(\bm{x}, \bm{y}_1, \bm{y}_2)$, IPO shows that
optimization problem
[\[eq:ipo_first_square\]](#eq:ipo_first_square){reference-type="eqref"
reference="eq:ipo_first_square"} is equivalent to $$\begin{aligned}
{2}
    \label{eq:ipo_square_2}
    & \mathop{\mathrm{\text{minimize}}}_{\boldsymbol \theta} \quad && \mathop{\mathbb{E}}_{\bm{x}, \bm{y}_1, \bm{y}_2\sim \mu} \left[ \left(  h_{\boldsymbol \theta}(\bm{x}, \bm{y}_1, \bm{y}_2)  - v(\bm{y}_1, \bm{y}_2, \bm{x}) \right) ^2\right].
\end{aligned}$$ I have a more rigorous derivation in the pdf attachment
proving this part
(Section [\[subs:technical_derivation\]](#subs:technical_derivation){reference-type="ref"
reference="subs:technical_derivation"}). The derivation is not super
complex but it certainly is not trivial. I am very baffled as how they
came up with this observation.

The equivalent optimization
[\[eq:ipo_square_2\]](#eq:ipo_square_2){reference-type="eqref"
reference="eq:ipo_square_2"} is tremendously useful since the target in
[\[eq:ipo_square_2\]](#eq:ipo_square_2){reference-type="eqref"
reference="eq:ipo_square_2"} can be approximated as
$$v(\bm{y}_1, \bm{y}_2, \bm{x}) \approx c-1$$

With such approximation, IPO proposed the following optimization
problem: $$\begin{aligned}
{2}
    & \mathop{\mathrm{\text{minimize}}}_{\boldsymbol \theta} \quad && \mathop{\mathbb{E}}_{\bm{x}, \bm{y}_1, \bm{y}_2\sim \mu, c} \left[ \left(  h_{\boldsymbol \theta}(\bm{x}, \bm{y}_1, \bm{y}_2)  - c+1 \right) ^2\right].
\end{aligned}$$ This can be further simplified when considering
symmetrical property of the preference data. For instance, a sample
$(\bm{y}, \bm{y}', \bm{x}, 1)$ induces another sample
$(\bm{y}', \bm{y}, \bm{x}, 2)$. Exploiting this structure and denote
$\bm{y}_w, \bm{y}_\ell$ based on the label $c$ such that
$\bm{y}_w \succ \bm{y}_\ell$, the previous problem is equivalent to
$$\begin{aligned}
{2}
    & \mathop{\mathrm{\text{minimize}}}_{\boldsymbol \theta} \quad && \mathop{\mathbb{E}}_{\bm{x}, \bm{y}_w, \bm{y}_{\ell }\sim \mu} \left[ \left(  h_{\boldsymbol \theta}(\bm{x}, \bm{y}_w, \bm{y}_\ell )  - 0.5 \right) ^2\right].
\end{aligned}$$

## Take-Home Points {#sub:take_home_points}

To answer the questions in the beginning on this section:

-   How can it turn a classification to a regression problem? It does
    not. RLHF/DPO learn the underling model by casting the problem as a
    classification problem. IPO does not learn any underlying model, and
    hence there is no need of any classification problem. The regression
    problem stems from enforcing condition
    [\[eq:h_theta_cond\]](#eq:h_theta_cond){reference-type="eqref"
    reference="eq:h_theta_cond"} for the optimal policy.

-   Why regressing toward $0.5$? This fixed number $0.5$ is a result to
    a quite rough approximation
    $\textsf{Pr}\left(c=1\mid \bm{x}, \bm{y}_1, \bm{y}_2\right) = v(\bm{y}_1, \bm{y}_2, \bm{x}) \approx c-1$.

-   Lastly, as IPO's working directly on the logit space, it seems to be
    more numerically robust compared to previous methods.

## Technical Derivation

::: lemma
Optimization
problems [\[eq:ipo_first_square\]](#eq:ipo_first_square){reference-type="eqref"
reference="eq:ipo_first_square"} and
[\[eq:ipo_square_2\]](#eq:ipo_square_2){reference-type="eqref"
reference="eq:ipo_square_2"} are equivalent.
:::

::: proof
*Proof.* Define
$$\bm{q}_{\boldsymbol \theta}(\bm{y}, \bm{x}) = \beta \log \dfrac{\pi_{\boldsymbol \theta}(\bm{y} \mid \bm{x})}{\pi_{\rm ref }(\bm{y} \mid \bm{x})},$$
then
$\bm{h}_{\boldsymbol \theta}(\bm{y}_1, \bm{y}_2, \bm{x})= \bm{q}_{\boldsymbol \theta}(\bm{y}_1, \bm{x}) - \bm{q}_{\boldsymbol \theta}(\bm{y}_2, \bm{x})$.
Expanding the squared term in the objective of
[\[eq:ipo_first_square\]](#eq:ipo_first_square){reference-type="eqref"
reference="eq:ipo_first_square"}, the optimization problem reduces to
$$\label{eq:qioqli}
\min_{\boldsymbol \theta}  \mathop{\mathbb{E}}_{\substack{\bm{x},\\ \bm{y}_1, \bm{y}_2\sim \mu}} \left[ (q_{\boldsymbol \theta}(\bm{y}_1, \bm{x})- q_{\boldsymbol \theta}(\bm{y}_2, \bm{x}))^2 - 2 [q_{\boldsymbol \theta}(\bm{y}_1, \bm{x}) - q_{\boldsymbol \theta}(\bm{y}_2, \bm{x})]\mathop{\mathbb{E}}_{\bm{y} \sim \mu}\left[v(\bm{y}_1, \bm{y}, \bm{x}) - v(\bm{y}_2, \bm{y}, \bm{x}) \right]  \right]$$
The issue lays in the unknown factor in the second term. Now comes to
the IPO's peculiar derivations. The second term can be written as
$$\begin{aligned}
\quad \quad &\mathop{\mathbb{E}}_{\bm{y}_1, \bm{y}_2 \sim \mu} \left[ [q_{\boldsymbol \theta}(\bm{y}_1, \bm{x})- q_{\boldsymbol \theta}(\bm{y}_2, \bm{x})]  \mathop{\mathbb{E}}_{\bm{y} \sim \mu}\left[ v(\bm{y}_1, \bm{y}, \bm{x}) - v(\bm{y}_2, \bm{y}, \bm{x}) \right]  \right] \\
&= \mathop{\mathbb{E}}_{\bm{y}_1, \bm{y}_2, \bm{y} \sim \mu}[({\color{green}q_{\boldsymbol \theta}(\bm{y}_1, \bm{x})} - {\color{blue}q_{\boldsymbol \theta}(\bm{y}_2, \bm{x})}) \left({\color{green}v(\bm{y}_1, \bm{y}, \bm{x})} - {\color{blue}v(\bm{y}_2, \bm{y}, \bm{x})} \right)] \\
&= \underbrace{\mathop{\mathbb{E}}_{\bm{y}_1, \bm{y}_2, \bm{y} \sim \mu} [{\color{green}q_{\boldsymbol \theta}({\color{green}\bm{y}_1}, \bm{x}) v(\bm{y}_1, \bm{y}, \bm{x})} + {\color{blue}q_{\boldsymbol \theta}(\bm{y}_2, \bm{x}) v(\bm{y}_2, \bm{y}, \bm{x})}]}_{(*)} \\
&\qquad \qquad - \underbrace{\mathop{\mathbb{E}}_{\bm{y}_1, \bm{y}_2, \bm{y} \sim \mu } [{\color{green}q_{\boldsymbol \theta}(\bm{y}_1, \bm{x})} {\color{blue}v(\bm{y}_2, \bm{y}, \bm{x})}+ {\color{blue}q_{\boldsymbol \theta}(\bm{y}_2, \bm{x})} {\color{green}v(\bm{y}_1, \bm{y}, \bm{x})}] }_{(**)}
\end{aligned}$$ For (\*), notice that the two terms are essentially the
same under expectation when $\bm{y}_1, \bm{y}_2, \bm{y}$ are i.i.d.
$$\begin{aligned}
(*) &=\mathop{\mathbb{E}}_{\bm{y}_1, \bm{y}_2, \bm{y}} [q_{\boldsymbol \theta}(\bm{y}_1, \bm{x}) v(\bm{y}_1, \bm{y}, \bm{x}) + q_{\boldsymbol \theta}(\bm{y}_2, \bm{x}) v(\bm{y}_2, \bm{y}, \bm{x})] \\
&=\mathop{\mathbb{E}}_{\bm{y}_1, \bm{y}} [q_{\boldsymbol \theta}(\bm{y}_1, \bm{x}) v(\bm{y}_1, \bm{y}, \bm{x})] + \mathop{\mathbb{E}}_{\bm{y}_2, \bm{y}} [q_{\boldsymbol \theta}(\bm{y}_2, \bm{x} ) v(\bm{y}_2, \bm{y}, \bm{x})] \\
&=\mathop{\mathbb{E}}_{\bm{y}_1, \bm{y}_2} [{q_{\boldsymbol \theta}(\bm{y}_1, \bm{x}) v(\bm{y}_1, \bm{y}_2, \bm{x})}] + \mathop{\mathbb{E}}_{\bm{y}_1, \bm{y}} [{q_{\boldsymbol \theta}(\bm{y}_1, \bm{x}) v(\bm{y}_1, \bm{y}, \bm{x})}] \\
&=\mathop{\mathbb{E}}_{\bm{y}_1, \bm{y}_2} [{q_{\boldsymbol \theta}(\bm{y}_1, \bm{x}) v(\bm{y}_1, \bm{y}_2, \bm{x})}] + \mathop{\mathbb{E}}_{\bm{y}_1, \bm{y}_2} [{q_{\boldsymbol \theta}(\bm{y}_1, \bm{x}) v(\bm{y}_1, \bm{y}_2, \bm{x})}] \\
&= 2\mathop{\mathbb{E}}_{\bm{y}_1, \bm{y}_2} [q_{\boldsymbol \theta}(\bm{y}_1, \bm{x}) v(\bm{y}_1, \bm{y}_2, \bm{x})] \addtocounter{equation}{1}\tag{\theequation}\label{eq:qobpal}
\end{aligned}$$

::: tcolorbox
The particular names of variables under expectation do not matter as
long as they share the same distribution. A simple example demonstrating
this point:
$$\mathop{\mathbb{E}}_{x \sim p}[f(x)] + \mathop{\mathbb{E}}_{y \sim p}[f(y)]
= \mathop{\mathbb{E}}_{x \sim p}[f(x)] + \mathop{\mathbb{E}}_{x \sim p}[f(x)]
= 2\mathop{\mathbb{E}}_{x\sim p}[f(x)].$$
:::

Similar trick can be applied for $(**)$ as well: $$\begin{aligned}
(**)&=\mathop{\mathbb{E}}_{\bm{y}_1, \bm{y}_2, \bm{y} \sim \mu } [q_{\boldsymbol \theta}(\bm{y}_1, \bm{x}) v(\bm{y}_2, \bm{y}, \bm{x}) + q_{\boldsymbol \theta}(\bm{y}_2, \bm{x}) v(\bm{y}_1, \bm{y}, \bm{x})]  \\
&= 2\mathop{\mathbb{E}}_{\bm{y}_1, \bm{y}_2, \bm{y} \sim \mu } [q_{\boldsymbol \theta}(\bm{y}_1,\bm{x}) v(\bm{y}_2, \bm{y}, \bm{x})] \\
&= 2\mathop{\mathbb{E}}_{\bm{y}_1 \sim \mu } [q_{\boldsymbol \theta}(\bm{y}_1, \bm{x})] \mathop{\mathbb{E}}_{\bm{y}_2, \bm{y} \sim \mu}[v(\bm{y}_2, \bm{y}, \bm{x})] \\
&=\mathop{\mathbb{E}}_{\bm{y}_1 \sim \mu } [q_{\boldsymbol \theta}(\bm{y}_1, \bm{x})] \addtocounter{equation}{1}\tag{\theequation}\label{eq:qoaibl}
\end{aligned}$$ where the last equality holds because
$\bm{y}_1, \bm{y}_2, \bm{y}$ are independent, and
$\mathop{\mathbb{E}}_{\bm{y}_2, \bm{y} \sim \mu}[v(\bm{y}_2, \bm{y}, \bm{x})] = 0.5$
which in turn can be derived from conditions
[\[eq:ipo_self_compare\]](#eq:ipo_self_compare){reference-type="eqref"
reference="eq:ipo_self_compare"} and
[\[eq:ipo_symmetric\]](#eq:ipo_symmetric){reference-type="eqref"
reference="eq:ipo_symmetric"}. Combining
[\[eq:qobpal\]](#eq:qobpal){reference-type="eqref"
reference="eq:qobpal"},
[\[eq:qoaibl\]](#eq:qoaibl){reference-type="eqref"
reference="eq:qoaibl"} gives $$\begin{aligned}
(*) - (**)
&= 2\mathop{\mathbb{E}}_{\bm{y}_1, \bm{y}_2}[q_{\boldsymbol \theta}(\bm{y}_1, \bm{x}) v(\bm{y}_1, \bm{y}_2, \bm{x})] - \mathop{\mathbb{E}}_{\bm{y}_1} [q_{\boldsymbol \theta}(\bm{y}_1, \bm{x}) ] \\
&= \mathop{\mathbb{E}}_{\bm{y}_1, \bm{y}_2} \left[2q_{\boldsymbol \theta}(\bm{y}_1, \bm{x}) v(\bm{y}_1, \bm{y}_2, \bm{x}) - (v(\bm{y}_1, \bm{y}_2, \bm{x})+v(\bm{y}_2, \bm{y}_1, \bm{x}))q_{\boldsymbol \theta}(\bm{y}_1, \bm{x} )  \right] \quad \text{(by \eqref{eq:ipo_symmetric})}\\
&= \mathop{\mathbb{E}}_{\bm{y}_1, \bm{y}_2} \left[q_{\boldsymbol \theta}(\bm{y}_1, \bm{x}) v(\bm{y}_1, \bm{y}_2, \bm{x}) - v(\bm{y}_2, \bm{y}_1, \bm{x})q_{\boldsymbol \theta}(\bm{y}_1, \bm{x} )  \right] \\
&= \mathop{\mathbb{E}}_{\bm{y}_1, \bm{y}_2} \left[q_{\boldsymbol \theta}(\bm{y}_1, \bm{x}) v(\bm{y}_1, \bm{y}_2, \bm{x}) - v(\bm{y}_1, \bm{y}_2, \bm{x})q_{\boldsymbol \theta}(\bm{y}_2, \bm{x} )  \right] \\
&= \mathop{\mathbb{E}}_{\bm{y}_1, \bm{y}_2} \left[(q_{\boldsymbol \theta}(\bm{y}_1, \bm{x})-q_{\boldsymbol \theta}(\bm{y}_2, \bm{x})) v(\bm{y}_1, \bm{y}_2, \bm{x}) \right]
\end{aligned}$$ The expression in
[\[eq:qioqli\]](#eq:qioqli){reference-type="eqref"
reference="eq:qioqli"} then becomes $$\begin{aligned}
&\mathop{\mathrm{arg\,min}}_{\boldsymbol \theta} \; \mathop{\mathbb{E}}_{\substack{\bm{x}, \\ \bm{y}_1, \bm{y}_2 \sim \mu}} \left[ (q_{\boldsymbol \theta}(\bm{y}_1, \bm{x}) - q_{\boldsymbol \theta}(\bm{y}_2, \bm{x}))^2 \right] - 2 \mathop{\mathbb{E}}_{\substack{\bm{x}, \\ \bm{y}_1, \bm{y}_2 \sim \mu}} \left[ (q_{\boldsymbol \theta}(\bm{y}_1, \bm{x}) - q_{\boldsymbol \theta}(\bm{y}_2, \bm{x})) v(\bm{y}_1, \bm{y}_2, \bm{x}) \right] \\
&=\mathop{\mathrm{arg\,min}}_{\boldsymbol \theta} \mathop{\mathbb{E}}_{\substack{\bm{x}, \\ \bm{y}_1, \bm{y}_2 \sim \mu}} \left[ ( h_{\boldsymbol \theta}(\bm{y}_1, \bm{y}_2, \bm{x}) - v(\bm{y}_1, \bm{y}_2, \bm{x}))^2 \right]
\end{aligned}$$ ◻
:::

#### "Notation".

I am aware that I have used some terminologies quite freely and that
could confuse rigorous readers. Here I like to clarify some of them

-   By "learning", I mean finding a mapping that can produce the right
    output given unseen input.

-   By "policy", I mean a language model

-   By "generation", I mean certain mechanism to sample
    $\bm{y} \sim \pi(\bm{y}\mid \bm{x})$
