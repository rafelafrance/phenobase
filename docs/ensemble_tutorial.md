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
You need 2 of 3 votes to be either positive or negative to be considered either, and everything else is considered unsure.

## Ensemble choice

Over time I built up hundreds of model checkpoints run with various parameters.
Of these, there were roughly 190 model checkpoints that I considered still relevant to the project.
I looked at all of the 3-combinations of the relevant models and chose the best one.
Note that I used the models as-is, that is they were not changed in any way, additionally,
I used the same model thresholds described in the README section on [threshold moving](../README.md#Threshold-moving). Also note that 190-choose-3 is a large number of combinations to test/score,
so I pruned models that did not meet a given accuracy limit (--checkpoint-limit).

See this bash [script](../args/ensemble_search.bash) for how I actually ran the ensemble search script.
The important output from this script is given in the --output-json argument.
Finally note that I used cached scores so that I didn't have to rerun all of the model checkpoints
(--score-csv).

The JSON output is located [here](https://zenodo.org/records/17079402).
It contains the checkpoint names and threshold parameters for running the models.
It also contains various measurements like F1, precision, accuracy, etc. the models got on the test data.
The only reason for the "top 10" is to see how many times a model was in the best ensembles.

## Running the ensemble

1. To run the ensemble you will have to download and ungzip the models located [here](https://zenodo.org/records/17079402).
2. Infer each model on the same data separately.
It creates a CSV file with the model results. Because there are 10s of millions of records for the
final data set I split 3 models into numerous runs for each model. One run would handle the 1st million records, the next the next 3 million records, and so on. These runs are grouped into a directory,
so model 1 would have approximately 30 CSV files in a single directory, and the model 2 would have
its own directory, and model 3... See this local bash [script](../args/local/model_infer_flowers.bash)
3. You need to merge the results of the 3 models into a single CSV vote for each herbarium sheet.
I glob the results of the 3 models, align them, and then vote on the results using each model's parameters. See the bash [script](../args/ensemble_vote.bash) what I did for the voting process.
4. Finally, I cleaned the data and formatted the final results.
  1. We only wanted records with both an event date and a latitude/longitude.
  2. Then we selected certain fields from the database and formatted the data before writing the results.
  3. See this bash [script](../args/ensemble_format.bash) for an example of how this was done.

The process is a bit cumbersome, but it works fine.
