Named Entity Recognition
=======
Named Entity Recognition is recognizing entities (Persons, Organizations, Locations, etc.) in a
sentence.
These are two NER projects using the same feature extractor. The first one is based on CRFs
using CRFsuite and the other is based on RNN's written in Keras. 
They are meant to be used with the czech named entity recognition corpus,
(http://ufal.mff.cuni.cz/cnec/cnec2.0), but can also work with a .json format. We plan to include
loading options for the CONLL2003 dataset as well.


CRF_NER
-------
This is a NER model evaluator for on conditional random fields. It uses 
pythoncrfsuite and sklearn. It is mainly used to experiment with different 
features extracted from the dataset and comparing the performance between 
models.

Setup
-----
The RAM usage (and model size) highly depends on the number of features picked,
I will add estimates later.

### Data
Download the CNEC2.0 dataset from 
https://lindat.mff.cuni.cz/repository/xmlui/handle/11858/00-097C-0000-0023-1B22-8

### Dependencies
To install the dependencies, use pip

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

### Train

With the CNEC2.0 plaintext files in the same folder, simply run:

    python3 -m src.CRF_NER.CRF_NER.py --train named_ent_train.txt named_ent_etest.txt model.txt merge

where merge can be either "BIO", "supertypes" or "none"
It generates modelname.crfmodel and modelname.log (performance evaluation on test set)


### Predict

Using the same model.txt file, run:

    python3 -m src.CRF_NER.CRF_NER.py --predict named_ent_train.txt named_ent_etest.txt model.txt merge

this will load the models defined in model.txt, extract the same features and run a tagger on the
test set.

Licenses
--------
CRFSuite library is licensed under the BSD license, python-crfsuite is licensed under MIT license
