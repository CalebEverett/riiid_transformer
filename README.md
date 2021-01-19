## Intro

This notebook trains a transformer model on the [EdNet dataset](https://github.com/riiid/ednet) using the [google/trax library](https://github.com/google/trax). The EdNet dataset is large set of student responses to multiple choice questions related to English language learning. A recent Kaggle competition, [Riiid! Answer Correctness Prediction](https://www.kaggle.com/c/riiid-test-answer-prediction), provided as subset of this data, consisting of 100 million responses to 13 thousand questions from 300 thousand students.

The state of the art result, detailed in [SAINT+: Integrating Temporal Features for EdNet Correctness Prediction](https://arxiv.org/abs/2010.12042), achieves an AUC ROC of 0.7914. The winning solution in the [Riiid! Answer Correctness Prediction](https://www.kaggle.com/c/riiid-test-answer-prediction) competition achieved an AUC ROC of 0.820. This notebook achieves an AUC ROC of 0.776 implementing an approach similar to the state of the art approach, training for 25,000 steps. It demonstrates several techniques that may be useful to those getting started with the [google/trax library](https://github.com/google/trax) or deep learning in general. This notebook demonstrates how to:

* Use BigQuery to perform feature engineering
* Create TFRecords with multiple sequences per record
* Modify the trax Transformer model to accommodate a knowledge tracing dataset:
    * Utilize multiple encoder and decoder embeddings - aggregated either by concatenation or sum
    * Include a custom metric - AUC ROC
    * Utilize a combined padding and future mask
* Use trax's [gin-config](https://github.com/google/gin-config) integration to specify training parameters
* Display training progress using trax's tensorboard integration

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/CalebEverett/riiid_transformer/blob/master/riiid-trax-transformer.ipynb)
