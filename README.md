# Three-Class Classifier Implementation in Python.
#### By Michael Glushchenko for UCSB CS165B Spring 2022 (Machine Learning).

## Table of Contents
* [Purpose](https://github.com/mglush/three-class-classifier/blob/main/README.md#purpose)
* [How To Run](https://github.com/mglush/three-class-classifier/blob/main/README.md#how-to-run)
* [Parameters and Format](https://github.com/mglush/three-class-classifier/blob/main/README.md#parameters-and-format)

## Purpose
[This repository](https://github.com/mglush/three-class-classifier) aims to build a 3-class classifier that works with any amount of data points in each class, and any number of features per data point. Project aimed at improving data-processing & data formatting skills while exploring the functionality of multi-class classifiers.

## How to Run
~~~
git clone git@github.com:mglush/three-class-classifier.git    # clone repository.
cd three-class-classifier                                     # enter repo folder.
python hw2.py training_filename testing_filename              # run file on a training/testing file input pair.
~~~

## Parameters and Format
**Input**:\
Two text files, one containing the training data, one containing the testing data. Check data/training*.txt for format examples.

**Output**:\
A dictionary containing the true positive rate,  false positive rate, error rate, accuracy, and precision of the classifier for the given test data.\

**Example**:\
{\
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;"tpr": _____,\
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;"fpr": _____,\
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;"error_rate": _____,\
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;"accuracy": _____,\
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;"precision": _____,\
}
