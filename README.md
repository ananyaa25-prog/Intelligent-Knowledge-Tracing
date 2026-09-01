# Knowledge Tracing 

> **An AI-driven framework for modeling student knowledge states and predicting future learning performance.**

## Overview

Knowledge Tracing (KT) aims to estimate a student's evolving knowledge state from historical learning interactions and predict performance on future exercises.

This project explores a **graph-based and temporal approach to Knowledge Tracing**, motivated by recent advances in deep learning, attention mechanisms, heterogeneous graphs, long- and short-term knowledge modeling, and robust student-state representation.

Rather than treating student interactions as isolated sequences, the project investigates how **concepts, exercises, relationships, temporal patterns, and learner behavior** can be jointly used to obtain a more meaningful representation of student knowledge.

## Research Motivation

Existing KT approaches have achieved strong predictive performance, but important limitations remain:

* Sequence-based models can struggle to represent complex relationships between concepts and exercises.
* Graph-based models may focus on limited relationship types or short-term interactions.
* Long-term learning history can be underrepresented when models emphasize recent interactions.
* Different relationships may contribute differently to a student's knowledge state.
* Student responses can contain noise or abnormal behavior such as guessing and plagiarism.
* Cognitive knowledge alone may not fully describe the learning process.

These limitations motivate a more comprehensive knowledge-state modeling framework.

## Proposed Direction

The project is designed around four core ideas:

**1. Temporal Knowledge Modeling**
Capture both recent learning behavior and longer-term learning history to represent the evolving knowledge state.

**2. Graph-based Representation**
Model relationships between concepts and exercises instead of treating interactions independently. This follows the direction of graph-based KT and multi-association knowledge modeling.
**3. Adaptive Knowledge Fusion**
Combine different knowledge-state representations according to their relevance to the current learning context rather than assigning every information source equal importance.

**4. Robust Student Modeling**
Investigate how noisy, biased, or irregular responses influence knowledge-state estimation and explore mechanisms for obtaining more reliable representations. Recent work demonstrates that biased behavior can significantly affect KT performance.

## Conceptual Pipeline

```text
Student Interaction History
          ↓
Exercise & Concept Representation
          ↓
Knowledge Relationship Modeling
          ↓
Long-Term + Short-Term Knowledge States
          ↓
Graph / Temporal Representation Learning
          ↓
Adaptive Knowledge-State Fusion
          ↓
Future Response Prediction
          ↓
Personalized Learning Insights
```

## Research Foundation

The project is aligned with the research progression established by the provided references:

| Research Direction | Key Idea                                                         |
| ------------------ | ---------------------------------------------------------------- |
| DKT / Deep KT      | Sequential modeling of student knowledge                         |
| AKT / SAKT         | Attention-based learning from historical interactions            |
| DKTMR              | Multiple knowledge relations and attention-based fusion          |
| DGEKT              | Dual graph modeling of heterogeneous relationships               |
| ELAKT              | Locality, knowledge aggregation and stochastic-behavior handling |
| L-SKSKT            | Joint long-term and short-term knowledge-state modeling          |
| DACE               | Debiased and robust knowledge-state representation               |
| MSKT               | Incorporating higher-order learner factors                       |

The literature shows a clear movement toward **richer, more structured and more reliable representations of student knowledge**.

## Expected Outcome

The objective is to develop a Knowledge Tracing framework that can:

* represent student knowledge more comprehensively,
* capture relationships between concepts and exercises,
* account for both recent and historical learning patterns,
* improve future response prediction,
* provide more interpretable knowledge-state information,
* and provide a foundation for personalized and adaptive learning systems.

## Publication Perspective

The project is being developed with a **research-oriented objective**, focusing not only on implementation but also on identifying a measurable gap beyond existing KT approaches.

The final research contribution will be established through:

* a clearly defined research gap,
* a novel modeling component or combination,
* rigorous baseline comparison,
* ablation studies,
* robustness analysis,
* and evaluation on established KT datasets.

> **Current status:** Research framework and direction under development.
>
> **Goal:** Develop and experimentally validate a novel, reproducible and publication-oriented Knowledge Tracing approach.

## References

The project is grounded in the reference papers supplied for this research, covering multi-relational KT, attentive KT, graph-based KT, long/short-term knowledge states, debiased KT, and metacognitive modeling.

