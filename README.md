# semivae-morph-reinflection

To run experiments for [SIGMORPHON 2016](http://ryancotterell.github.io/sigmorphon2016/), first download training, validatation and test data [here](https://github.com/ryancotterell/sigmorphon2016/tree/master/data/).  

Second, process data to make the following preparations as shown in line 65 in semi_models/run.py:
1. vocabulary size of all characters (voc_size), total number of labels (class_num), the list of number of classes for each label (label_list)
2. dictionary that maps index to character (ix_to_char), dictionary that map index to labels (ix_to_label).  

To process train, valid, test data, for each word, we first return a list of character indexes in the vocabulary and a list of labels. Note that a word doesn't possess all linguistic labels, in this case, we augment each label with an additional class that means the word doesn't have this label.  
During training, we batch a group of words by padding them to an identical length. For labels, we process labels of a batch of words as one-hot vectors for each label respectively and return a list of batched one-hot labels.

A running example is given under the run_commands folder for turkish.