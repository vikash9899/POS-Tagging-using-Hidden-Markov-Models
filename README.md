# PART OF SPEECH TAGGING

## Abstract —

In this project I create the part of speech tagger
using the basic probabilistic and hidden markov model. prob-
abilistic model is based on the calculate the N-Gram probability
for tags and probability that the tag is attach to the word.
In hidden markov model I used viterbi algorithm for tag the
sentence. for the implementation I use the penn tree bank 10%
data which is available at the nltk library. whole dataset(penn
tree bank) is not available for free so i use only 10% of that.

# 1. Introduction

Tagging is the process of tag the each word in sentence
corresponding to a particular part of speech tag, based on
its definition and context.
Part of speech tagging is very important in the Natural
language processing. In this project i make POS tagger for
the English language. I used the Probabilistic model and
Hidden markov model.

![imagename](images/pos_tag.png)
Fig. 1. Part of speech flow chart

An Hidden morkov model is a probabilistic sequence
model given a sequence of units. It calculate the probability
distribution over possible sequence of labels and chooses
the best sequence.

## 1.1. Motivation

Part of speech tags gives information about words in
language. Tag to the word also gives info about the word
and its neighbors. Part of speech tagging have application in
various tasks such as Information retrieval, parsing, Text to
Speech, semantics analysis, language translation and many
more. There is almost all the application of NLP required
Part of speech tagging as the sub task.

# 2. State of the art/Background

2.1. TnT – A Statistical Part-of-Speech Tagger [ 1 ]
Trigrams’n’Tags (TnT) is an efficient statistical part-of-
speech tagger based on the Hidden markov model. Hidden
markov models performs as well as the other current ap-
proaches, including the maximum entropy framework.
They use the

|Model performance|
|-----------------|
|Penn Tree Bank dataset |  Accuracy|
|--------|---------|
|known words | 97.0 %|
|--------|---------|
|unknown words|85.5 %|
|--------|---------|
|overall|96.7 %|

Table 1: Accuracy for the TnT model
