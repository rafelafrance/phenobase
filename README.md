# phenobase ![Python application](https://github.com/rafelafrance/phenobase/workflows/CI/badge.svg)
Classifiers for identifying phenology traits on images of plants

There is a lot of effort to digitize and annotate photographs of plant images and herbarium specimens. However, this effort is, up until now, mostly manual, error-prone, and labor-intensive resulting in only a fraction of these images being fully annotated. This project uses neural networks to automate the annotation of some biologically significant traits related to [phenology](https://en.wikipedia.org/wiki/Phenology): flowering, fruiting, leaf-out, etc.

The basic steps are:

1. Obtain a database of plant images with corresponding annotations.
   1. Make sure we can parse the annotations into classifiable traits.
   2. Clean and filter the images with annotations to contain only angiosperm records.
   3. Make sure we have an associated phylogenetic order for each specimen.
   4. Each specimen must have exactly one image associated with it. More than one image creates confusion as to which image contains the trait.
2. Train a neural network(s) to recognize the traits. We are using the [pytorch](https://pytorch.org/) library to build the neural networks.
   1. Supervised multi-class multi-label classifiers.
   2. I am starting with pretrained models and then using transfer learning and fine-tuning to train them for these data.
3. Use the networks to annotate images.

## Setup

1. `git clone https://github.com/rafelafrance/phenobase.git`
2. `python -m pip install .`
