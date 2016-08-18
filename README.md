Named Entity Recognition
=======

CRF_NER
-------
This is a NER model evaluator for on conditional random fields. It uses 
pythoncrfsuite and sklearn. It is mainly used to experiment with different 
features extracted from the dataset and comparing the performance between 
models.

CRF_NER is meant to be used with the czech named entity recognition corpus,
(http://ufal.mff.cuni.cz/cnec/cnec2.0)

Setup
-----
The RAM usage (and model size) highly depends on the number of features picked,
I will add estimates later.

### Data
First download the CNEC2.0 dataset from 
https://lindat.mff.cuni.cz/repository/xmlui/handle/11858/00-097C-0000-0023-1B22-8

### Dependencies
To install the dependencies, use pip

    pip install sklearn
    pip install python-crfsuite

Usage
-----

You have to define a model file which specifies which features to extract. They
are listed in the file features.txt. Each line should contain the name of the
model followed by the feature names, separated by whitespace.
Example:

    1_nbr label neighbours_1

The first line means the model will extract the label (i.e. the word) and 1
neighbour to the left and to the right. It will be saved as 1_nbr.crfmodel

Then, with the CNEC2.0 plaintext files in the same folder, simply run:

    python CRF_NER.py named_ent_train.txt named_ent_etest.txt model.txt

We plan to add further features to the feature extractor (POS tags, brown
clusters, gazetteer) and evaluate their performance.
