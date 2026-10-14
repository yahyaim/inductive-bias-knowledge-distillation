# Inductive Bias and Knowledge Distillation Between MLPs

Project progress report (University of Southampton, October 2022).

## Question
When one MLP classifier is used to train another MLP of the same architecture,
how does the L2 regularisation parameter (lambda), treated as an explicit
inductive bias, change the similarity of their last hidden-layer activations?

## Method
- Two-hidden-layer MLP (5 units per layer) trained on a randomly generated
  XOR task (400 points), then on a 4-dimensional logical function.
- Strong vs weak inductive bias identified by testing generalisation with reduced
  training data (lambda = 0.0001 strong, lambda = 0.015 weak for the XOR task).
- Response-based knowledge distillation: model 1 teaches model 2.
- Representational similarity: cosine similarity of activations, plus cluster
  analysis using Euclidean distances and Pearson correlation.

## Findings
- The two models always showed some similarity in the last hidden layer.
- The strongest similarity appeared when the teacher had a strong bias and
  the student a weak one, not when both were equally strong.
- Results were consistent across the 2-D and 4-D tasks.

## Limitations
The task is arbitrary, only one architecture was tested, and "strong" and "weak"
bias are relative to each other, so the findings are specific to this setup.

## Tools
Python, TensorFlow/Keras, NumPy.
