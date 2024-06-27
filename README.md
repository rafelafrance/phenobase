# phenobase ![Python application](https://github.com/rafelafrance/phenobase/workflows/CI/badge.svg)

Classifiers for identifying phenology traits on images of herbarium sheets.

There is a lot of effort to digitize and annotate photographs of plant images and herbarium specimens. However, this effort is, up until now, mostly manual, error-prone, and labor-intensive resulting in only a fraction of these images being fully annotated. This project uses neural networks to automate the annotation of some biologically significant traits related to [phenology](https://en.wikipedia.org/wiki/Phenology): flowering, fruiting, leaf-out, etc.

The basic steps are:

1. Obtain a database of plant images with corresponding annotations.
   1. I'm using data from the [iDigBio](https://www.idigbio.org/) project to get the URL of images to download.
      1. Clean the database to only contain records with a single Angiosperm herbarium sheet, that also contain phenology annotations.
   2. We can either use the records from above that are pre-identified or have experts annotate the images. The later is preferable.
2. Train a neural network(s) to recognize the traits. We are using the [pytorch](https://pytorch.org/) library to build the neural networks. I am also, using models and scripts from [HuggingFace](https://huggingface.co/docs/transformers/model_doc/vit_mae).
   1. Because it can be difficult to get a significant amount of quality annotations I'm using masked [autoencoders](https://arxiv.org/abs/2111.06377v2) for a pretraining step.
   2. Use the encoding part of the masked autoencoder as a backbone for the actual phenology trait classifier.
3. Use the trained neural networks to annotate images _en masse_.

## Stay tuned

Coming soon!

- More thrills
- More spills
- More explanations of what I'm actually doing here.

## Setup

1. `git clone https://github.com/rafelafrance/phenobase.git`
2. `cd phenobase`
3. `make install`
