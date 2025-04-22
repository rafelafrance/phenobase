# phenobase ![Python application](https://github.com/rafelafrance/phenobase/workflows/CI/badge.svg)

Classifiers for identifying phenology traits on images of herbarium sheets.

There is a lot of effort to digitize and annotate photographs of plant images and herbarium specimens. However, this effort is, up until now, mostly manual, error-prone, and labor-intensive resulting in only a fraction of these images being fully annotated. This project uses neural networks to automate the annotation of some biologically significant traits related to [phenology](https://en.wikipedia.org/wiki/Phenology): flowering, fruiting, leaf-out, etc.

The basic steps are:

1. Have experts annotate traits on images of herbarium sheets.
2. Train a neural network(s) to recognize the traits. We are using the [pytorch](https://pytorch.org/) library to build the neural networks. I am also, using models and scripts from [HuggingFace](https://huggingface.co/docs/transformers/model_doc/vit_mae).
3. Use the trained neural networks to annotate images _en masse_.

## Setup

1. `git clone https://github.com/rafelafrance/phenobase.git`
2. `cd phenobase`
3. `make install`
