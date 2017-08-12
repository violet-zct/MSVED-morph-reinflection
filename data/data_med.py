__author__ = 'chuntingzhou'
import numpy as np
import theano
from collections import OrderedDict
import copy
import logging
import codecs
'''Task 1

For task 1, the fields are: lemma, MSD, target form. An example from the Spanish training data:

hablar  pos=V,mood=IND,polite=FORM,tense=FUT,per=3,num=SG       hablara
Task 2

In task 2, the fields are: source MSD, source form, target MSD, target form. For example:

pos=V,mood=IND,tense=PRS,per=1,num=SG,aspect=IPFV/PFV   hablo   pos=V,tense=PST,gen=MASC,num=PL hablados
Task 3

In task 3, the fields are: source form, target MSD, target form. For example:

hablo   pos=V,tense=PST,gen=MASC,num=PL hablados'''

logger = logging.getLogger(__name__)
datapath = '../data/'
langs = ['spanish', 'german', 'finnish', 'russian', 'turkish', 'georgian', 'navajo', 'arabic', 'hungarian', 'maltese']
task1_train = '-task1-train'
task1_test = '-task1-test' # treat as unlabeled data first
task1_dev = '-task1-dev' # test data
task2_train = '-task2-train'
task2_test = '-task2-test' # treat as unlabeled data first
task2_dev = '-task2-dev'
task3_train = '-task3-train'
task3_test = '-task3-test' # treat as unlabeled data first
task3_dev = '-task3-dev'

lang = ""

def read_task3(fname, allwords):
    # source_form target_label target_form
    res_tags = []
    with codecs.open(fname, 'r', "utf-8") as f:
        for line in f:
            fields = line.strip().split('\t')
            tags = fields[1].split(',')
            word_1 = fields[0]
            word_2 = fields[2]
            allwords += word_1 + word_2

            word = word_1 + word_2
            if '{' in word or '}' in word:
                logging.info(word)
            for tag in tags:
                res_tags.append(tag)
    return allwords, res_tags


def create_train_sup_task3(fname, src_char_to_ix, tgt_char_to_ix):
    # task3 test: source_form target_label target_form
    start = src_char_to_ix['<w>']
    # end = char_to_ix['</w>']
    lx = []
    ly = []
    with codecs.open(fname, 'r', "utf-8") as f:
        ct = 1
        for line in f:
            fields = line.strip().split('\t')
            srcx = fields[0]
            tgtx = fields[2]
            tgty = fields[1].split(',')
            word = []
            word.append(start)
            for tag in tgty:
                if tag in src_char_to_ix:
                    word.append(src_char_to_ix[tag])
            for char in srcx:
                if char not in src_char_to_ix:
                    logging.info("src error!! %s, %d", char, ct)
                    continue
                word.append(src_char_to_ix[char])
            # word.append(end)
            lx.append(word)

            tgt_word = []
            for char in tgtx:
                if char not in tgt_char_to_ix:
                    logging.info("tgt error!! %s, %d", char, ct)
                    continue
                tgt_word.append(tgt_char_to_ix[char])
            # tgt_word.append(end)
            ly.append(tgt_word)
            ct += 1
    return lx, ly


def preprocess():
    task3_labeled = datapath + lang + task3_train
    task3_unlabeled = datapath + lang + task3_test

    # test_data = datapath + langs[7] + task2_dev
    allwords = ""
    allwords, tags = read_task3(task3_labeled, allwords)

    chars = list(set(allwords))
    tgt_voc = copy.deepcopy(chars)
    tgt_voc_size = len(tgt_voc) + 1
    tags = list(set(tags))
    chars += ['<w>']
    chars += tags
    # chars += ['UNK']
    data_size, src_voc_size = len(allwords), len(chars)+1
    logging.info('data has %d characters, %d src voc size, %d tgt voc size.', data_size, src_voc_size, tgt_voc_size)
    src_char_to_ix = {ch: i+1 for i, ch in enumerate(chars)}
    src_ix_to_char = {i+1: ch for i, ch in enumerate(chars)}

    tgt_char_to_ix = {ch: i+1 for i, ch in enumerate(tgt_voc)}
    tgt_ix_to_char = {i+1: ch for i, ch in enumerate(tgt_voc)}

    logging.info(tgt_char_to_ix)
    logging.info(src_char_to_ix)
    # class_num = len(tags_pre)
    # label_list = [len(v)+1 for v in tags_pre.values()]

    train_x, train_y = create_train_sup_task3(task3_labeled, src_char_to_ix, tgt_char_to_ix)
    # test data: task3_test
    test_x, test_y = create_train_sup_task3(task3_unlabeled, src_char_to_ix, tgt_char_to_ix)
    return src_voc_size, tgt_voc_size, src_ix_to_char, tgt_ix_to_char, train_x, train_y, test_x, test_y


def get_batches(data_length, batch_size):
    num_batch = int(np.ceil(data_length/float(batch_size)))
    return [(i*batch_size, min(data_length, (i+1)*batch_size)) for i in range(0, num_batch)]


def prepare_data(seqs_x, seqs_y, maxlen=None):
    # x: a list of sentences
    lengths_x = [len(s) for s in seqs_x]
    lengths_y = [len(s) for s in seqs_y]

    if maxlen is not None:
        new_seqs_x = []
        new_seqs_y = []
        new_lengths_x = []
        new_lengths_y = []
        for l_x, s_x, l_y, s_y in zip(lengths_x, seqs_x, lengths_y, seqs_y):
            if l_x < maxlen and l_y < maxlen:
                new_seqs_x.append(s_x)
                new_lengths_x.append(l_x)
                new_seqs_y.append(s_y)
                new_lengths_y.append(l_y)
        lengths_x = new_lengths_x
        seqs_x = new_seqs_x
        lengths_y = new_lengths_y
        seqs_y = new_seqs_y

        if len(lengths_x) < 1 or len(lengths_y) < 1:
            return None, None, None, None

    n_samples = len(seqs_x)
    maxlen_x = np.max(lengths_x) + 1
    maxlen_y = np.max(lengths_y) + 1

    x = np.zeros((n_samples, maxlen_x)).astype('int64')
    y = np.zeros((n_samples, maxlen_y)).astype('int64')
    x_mask = np.zeros((n_samples, maxlen_x)).astype('float32')
    y_mask = np.zeros((n_samples, maxlen_y)).astype('float32')
    for idx, [s_x, s_y] in enumerate(zip(seqs_x, seqs_y)):
        x[idx, :lengths_x[idx]] = s_x
        x_mask[idx, :lengths_x[idx]+1] = 1.
        y[idx, :lengths_y[idx]] = s_y
        y_mask[idx, :lengths_y[idx]+1] = 1.

    return x, x_mask, y, y_mask


def prepare_x_batch(seqs_x, maxlen=None):
    # x: a list of words
    lengths_x = [len(s) for s in seqs_x]

    if maxlen is not None:
        new_seqs_x = []
        new_lengths_x = []
        for l_x, s_x in zip(lengths_x, seqs_x):
            if l_x < maxlen:
                new_seqs_x.append(s_x)
                new_lengths_x.append(l_x)
            else:
                print("length, x; ", l_x, s_x)
        lengths_x = new_lengths_x
        seqs_x = new_seqs_x

        if len(lengths_x) < 1:
            return None, None, None

    n_samples = len(seqs_x)
    maxlen_x = np.max(lengths_x) + 1

    x = np.zeros((n_samples, maxlen_x)).astype('int64')
    x_mask = np.zeros((n_samples, maxlen_x)).astype(theano.config.floatX)

    for idx, s_x in enumerate(seqs_x):
        x[idx, :lengths_x[idx]] = s_x
        x_mask[idx, :lengths_x[idx] + 1] = 1.

    return x, x_mask

if __name__ == '__main__':
    # test
    if False:
        pass
        # voc_size, class_num, label_list, ix_to_char, ix_to_label = preprocess()
        # print("number of unlabeled training set: ", len(ux))
        # print(voc_size, class_num, label_list, ix_to_char, ix_to_label)

    if True:
        f1 = "task1_test"
        f2 = "task2_test"


