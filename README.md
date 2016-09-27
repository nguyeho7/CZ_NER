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
Download the CNEC2.0 dataset from 
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

keras_NER
---------
Named entity recognition using recurrent neural networks written in Keras with Theano backend. This
is currently highly volatile as we are running all sorts of experiments. 
