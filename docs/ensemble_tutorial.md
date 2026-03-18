# Ensemble tutorial

## Rationale

Clearly it would be convenient to have a single model to run for all species to get the all traits off of the herbarium sheets.
However it became apparent that this was not going to work as desired. I have been told by botanists that many of the traits like flowering and fruiting are extremely difficult to determine even by knowledgable people for certain families.
Additionally, there are preservation issues with many of the herbarium sheets.
In short the data is a mess.
*Sigh* This does not completely rule out skill issues for my own part.

In an effort improve the efficacy of the model parsing we took the following:
1. We reduced the models to only deal with a single trait at a time.
A model for flowers, another for fruits, etc.
2. Removal of certain problematic families (& genera) from the training data.
If the project botanist could not reliably determine the traits then we couldn't score model's results.
We did this programmatically as described [here](../README.md#Data-pruning).
3. Threshold moving. A lot of models are used with a 0.5 cutoff and no room for doubt.
We took the approach that the model could be "unsure" and a score in the neighborhood of 0.5 indicates this. The unsure zone was also found programmatically as described [here](../README.md#Threshold-moving).
4. Finally, we combined the results of 3 models via a vote.
A vote could be positive, negative, or unsure.
You need 2 of 3 votes to be either positive or negative, and everything else is considered unsure.

## Ensemble choice

So I'm sitting there with over 50 model checkpoints run with various parameters.
I chose the 3 models for the ensemble by trying


## Running the ensemble
