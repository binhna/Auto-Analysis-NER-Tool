#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import re
import numpy as np

from itertools import chain
import matplotlib.pyplot as plt
import xlsxwriter

import six

#store number of each type of errors
wtag = 0
wrange = 0
wrange_tag = 0
noextraction = 0
noannotation = 0
#temporary variable to decide whether the error is no annotation or not
temp = 0
flag = False

# number of error sentences
number_err = 0

#variable for excel file
row = 0
col = 0
workbook = xlsxwriter.Workbook('/Users/binhnguyen/Desktop/test.xlsx')
worksheet = workbook.add_worksheet()
worksheet.write(row, col, "Item")
worksheet.write(row, col + 1, "Gold")
worksheet.write(row, col + 2, "Predict")
worksheet.write(row, col + 3, "Type-Predict")
worksheet.write(row, col + 5, "Type-Gold")
row + 1


#check what kind of error of current tags -- write the error type to excel file
def check_duplicate(predict, gold, row, col, detail_error):
    global flag, temp, wtag, wrange, wrange_tag, noextraction, noannotation

    c = tuple(sorted(set(predict[:2]))) + tuple(sorted(set(gold[:2])))
    d = tuple(sorted(set(gold[:2]))) + tuple(sorted(set(predict[:2])))

    label_predict = predict[2]
    label_gold = gold[2]
    #no annotation - 1234 and not duplicate
    if((c==tuple(sorted(c)) or (d==tuple(sorted(d)))) and len(c)==len(set(c))):
        temp+=1
    #completely the same but tag
    elif(len(c)==2*len(set(c))):
        wtag+=1
        worksheet.write(row+1+predict[0], col, "T")
        detail_error[label_predict]['wtag']+=1
        flag = True
    #have a bounary and tags are correct
    elif(label_predict[2]==label_gold[2]):
        wrange+=1
        worksheet.write(row+1+predict[0], col, "R")
        detail_error[label_predict]['wrange']+=1
        flag = True
    #have a bounary but wrong tag
    else:
        wrange_tag+=1
        worksheet.write(row+1+predict[0], col, "W")
        detail_error[label_predict]['wrange_tag']+=1
        flag = True


def read_conll_file(filenames, delimiter=u'\t', input_idx=0, output_idx=-1):
    sentence = []
    sentences = []
    n_features = -1
    for filename in filenames:
        with open(filename) as f:
            for line_idx, l in enumerate(f):
                l_split = str_to_unicode_python2(l).strip().split(delimiter)
                l_split = [_.strip() for _ in l_split]
                if len(l_split) <= 1:
                    if len(sentence) > 0:
                        sentences.append(sentence)
                        sentence = []
                    continue
                else:
                    if n_features == -1:
                        n_features = len(l_split)

                    if n_features != len(l_split):
                        val = (str(len(l_split)), str(line_idx))
                        err_msg = 'Invalid input feature sizes: "%s". \
                        Please check at line [%s]' % val
                        raise ValueError(err_msg)
                    sentence.append(l_split)
        if len(sentence) > 0:
            sentences.append(sentence)
    return sentences


# Conll 03 shared task evaluation code (IOB format only)


def conll_eval(gold_predict_pairs, flag=True, tag_class=None):
    """
    Args : gold_predict_pairs = [
                                    gold_tags,
                                    predict_tags
                                ]

    """

    # flag will be false. gold_tags and predicted_tags is list tags sentence.
    if flag:
        gold_tags, predict_tags = zip(*gold_predict_pairs)
    else:
        gold_tags, predict_tags = gold_predict_pairs

    n_phrase_gold = 0
    n_phrases_tag_gold = 0
    n_phrases_dict = {}
    n_phrases_dict['gold'] = {}
    n_phrases_dict['predict'] = {}
    cnt_phrases_dict = {}
    evals = {}

    # store all the test file (gold tags) and the words to a list (for writing words to excel file purpuse)
    global row, col

    #dict where all the errors is stored
    detail_error = {}
    detail_error_gold = {}

    fileself = []
    sentself = []
    import codecs
    with codecs.open("/Users/binhnguyen/Downloads/Data-NER/viet.test.txt", 'r', 'utf-8') as fi:
        myfile = fi.readlines()
    for l in myfile:
        wordself = l.strip()
        if wordself == "":
            # sentence split
            fileself.append(sentself)
            sentself = []
        else:
            sentself.append(wordself)
    if len(sentself):
        fileself.append(sentself)
        sentself = []

    
    for tag_name in tag_class:

        # detail_error[PER][wrong_range] = 15, ....
        detail_error[tag_name.encode('ascii')]={}
        detail_error[tag_name.encode('ascii')]['wrange'] = 0
        detail_error[tag_name.encode('ascii')]['wtag'] = 0
        detail_error[tag_name.encode('ascii')]['wrange_tag'] = 0
        detail_error[tag_name.encode('ascii')]['noannotation'] = 0

        detail_error_gold[tag_name.encode('ascii')]={}
        detail_error_gold[tag_name.encode('ascii')]['wrange'] = 0
        detail_error_gold[tag_name.encode('ascii')]['wtag'] = 0
        detail_error_gold[tag_name.encode('ascii')]['wrange_tag'] = 0
        detail_error_gold[tag_name.encode('ascii')]['noextraction'] = 0

        # cnt_phrases_dict[PER][predict_cnt] = 0, ...
        cnt_phrases_dict[tag_name] = {}
        cnt_phrases_dict[tag_name]['predict_cnt'] = 0
        cnt_phrases_dict[tag_name]['gold_cnt'] = 0
        cnt_phrases_dict[tag_name]['correct_cnt'] = 0

    for i in six.moves.xrange(len(gold_tags)):
        gold = gold_tags[i]
        predict = predict_tags[i]

        # list range of tags in current sentence, ex: [(1,2,PER), (3,5,ORG), ...]
        gold_phrases = IOB_to_range_format_one(gold, is_test_mode=True)
        predict_phrases = IOB_to_range_format_one(predict, is_test_mode=True)

        # analyze the errors and write that error to excel file
        gold_range_set = set(gold_phrases)
        predict_range_set = set(predict_phrases)
            # number of error sentences
        global number_err

        if ((predict_range_set | gold_range_set) != (predict_range_set & gold_range_set)):
            number_err+=1

            tool(gold_range_set, predict_range_set, row, 5, detail_error_gold)
            tool(predict_range_set, gold_range_set, row, 3, detail_error)

            worksheet.write(row, col, number_err)
            row += 1

            for l in xrange(len(fileself[i])):
                worksheet.write(row, col, fileself[i][l].split()[0])
                worksheet.write(row, col + 1, fileself[i][l].split()[-1])
                worksheet.write(row, col + 2, predict_tags[i][l])
                row += 1


        # number of each tag's type in gold and predict. ex: n_phrases_dict: {gold: {LOC: 15, ORG: 2}, predict: {LOC: 15, ORG: 2}}
        for p in gold_phrases:
            tag_name = p[-1]
            n_phrases_dict['gold'][tag_name] = n_phrases_dict['gold'].get(tag_name, 0) + 1

        for p in predict_phrases:
            tag_name = p[-1]
            n_phrases_dict['predict'][tag_name] = n_phrases_dict['predict'].get(tag_name, 0) + 1

        # sumary number of correct, gold, predict tags in current sentence by tag class (PER, ORG)
        for tag_name in tag_class:

            _gold_phrases = [_ for _ in gold_phrases if _[-1] == tag_name]
            _predict_phrases = [_ for _ in predict_phrases if _[-1] == tag_name]

            correct_cnt, gold_cnt, predict_cnt = range_metric_cnt(_gold_phrases, _predict_phrases)
            cnt_phrases_dict[tag_name]['gold_cnt'] += gold_cnt if len(_gold_phrases) > 0 else 0
            cnt_phrases_dict[tag_name][
                'predict_cnt'] += predict_cnt if len(_predict_phrases) > 0 else 0
            cnt_phrases_dict[tag_name]['correct_cnt'] += correct_cnt

    workbook.close()

    # list number of each tag class by gold and predict. ex: gold: {LOC: 100, PER: 300, ORG: 250}...
    lst_gold_phrase = n_phrases_dict['gold']
    lst_predict_phrase = n_phrases_dict['predict']

    # number of gold tags and predict tags
    num_gold_phrase = sum(six.itervalues(n_phrases_dict['gold']))
    num_predict_phrase = sum(six.itervalues(n_phrases_dict['predict']))

    # overall information above
    phrase_info = [num_gold_phrase, num_predict_phrase, lst_gold_phrase, lst_predict_phrase]

    # recall, precision, f1
    for tag_name in tag_class:
        if cnt_phrases_dict[tag_name]['gold_cnt']:
            recall = cnt_phrases_dict[tag_name]['correct_cnt'] / \
                float(cnt_phrases_dict[tag_name]['gold_cnt'])
        else:
            recall = 0.0
        if cnt_phrases_dict[tag_name]['predict_cnt']:
            precision = cnt_phrases_dict[tag_name]['correct_cnt'] / \
                float(cnt_phrases_dict[tag_name]['predict_cnt'])
        else:
            precision = 0.0
        sum_recall_precision = 1.0 if recall + precision == 0.0 else recall + precision
        f_measure = (2 * recall * precision) / (sum_recall_precision)
        evals[tag_name] = [precision * 100.0, recall * 100.0, f_measure * 100.0]

    correct_cnt = sum([ev['correct_cnt'] for tag_name, ev in six.iteritems(cnt_phrases_dict)])
    gold_cnt = sum([ev['gold_cnt'] for tag_name, ev in six.iteritems(cnt_phrases_dict)])
    predict_cnt = sum([ev['predict_cnt'] for tag_name, ev in six.iteritems(cnt_phrases_dict)])

    recall = correct_cnt / float(gold_cnt) if gold_cnt else 0.0
    precision = correct_cnt / float(predict_cnt) if predict_cnt else 0.0
    sum_recall_precision = 1.0 if recall + precision == 0.0 else recall + precision
    f_measure = (2 * recall * precision) / (sum_recall_precision)
    evals['All_Result'] = [precision * 100.0, recall * 100.0, f_measure * 100.0]


    global wrange, wtag, wrange_tag, noannotation

    labels = 'Wrong Range', 'Wrong Tag', 'Wrong Range&Tag', 'No Annotation'
    colors = ['gold', 'yellowgreen', 'lightcoral', 'lightskyblue']
    sizes = [wrange, wtag, wrange_tag, noannotation]

    plt.pie(sizes, labels=labels, colors=colors,
        autopct='%1.1f%%', shadow=True, startangle=140)
    
    plt.axis('equal')
    #plt.show()

    #print(detail_error)
    #print("wrong range: ", wrange, ",wrong tag: ", wtag, ", wrong range & tag: ", wrange_tag, ", no annotaion: ", noannotation)
    #print(number_err)
    #print evals

    mydict = before_table(detail_error, cnt_phrases_dict, "predict")
    mydict2 = before_table(detail_error_gold, cnt_phrases_dict, "gold")
    
    return evals, phrase_info, mydict, mydict2


def before_table(dicti, cnt_phrases_dict, s):
    mydict = {}
    if s == "predict":
        lb = "predict_cnt"
    else:
        lb = "gold_cnt"

    for key, value in dicti.items():
        mydict[key] = [cnt_phrases_dict[unicode(key, "utf-8")]['correct_cnt'], sum(value.values()), cnt_phrases_dict[unicode(key, "utf-8")][lb]] + [value[i] for i in sorted(value.keys())]
    
    length = len(mydict)
    temp = [0]*7
    for i in xrange(length):
        a=mydict[mydict.keys()[i]]
        temp=[x+y for x, y in zip(temp, a)]
    print temp
    mydict["All_Tags"]=temp

    return mydict


def tool(seta, setb, row, col, detail):
    
    global noannotation, noextraction, flag, temp, number_err
    set1 = sorted(seta - setb)
    set2 = sorted(setb - seta)

    if col == 3:
        label = "A"
    else:
        label = "E"

    for a in set1:
        temp = 0
        if(len(set2) == 0):
            if label == "A":
                noannotation+=1
                detail[a[2]]['noannotation']+=1
                worksheet.write(row+1+a[0], col, label)
            else:
                noextraction+=1
                detail[a[2]]['noextraction']+=1
                worksheet.write(row+1+a[0], col, label)
        for b in set2:
            check_duplicate(a,b,row, col,detail)
            if(temp==len(set2)):
                if label == "A":
                    noannotation+=1
                    detail[a[2]]['noannotation']+=1
                    worksheet.write(row+1+a[0], col, label)
                else:
                    noextraction+=1
                    detail[a[2]]['noextraction']+=1
                    worksheet.write(row+1+a[0], col, label)
                flag = True
                temp=0
            if flag:
                flag = False
                break

# range of tags in one sentence and its type. Ex: (1,2,PER)
def IOB_to_range_format_one(tag_list, is_test_mode=False):
    sentence_lst = []
    ner = []
    ner_type = []
    # print(tag_list)
    for i in six.moves.xrange(len(tag_list)):
        prev_tag = tag_list[i - 1] if i != 0 else ''
        prev_tag_type = prev_tag[2:]
        # prev_tag_head = tag_list[i-1][0] if i != 0 else ''
        tag = tag_list[i]
        tag_type = tag[2:]
        tag_head = tag_list[i][0]
        ner_is_exist = len(ner) > 0
        ner_is_end = (ner_is_exist and tag_head == u'O') or (ner_is_exist and tag_head == u'B') or (
            ner_is_exist and tag_head == u'I' and prev_tag_type != tag_type)
        # NOTE: In Conll 2003 evaluation code, I-ORG means NE start!!
        # ner_is_end_conll = (tag_head == u'I' and prev_tag_type != tag_type and not ner_is_exist)
        if ner_is_end:
            if is_test_mode and len(set(ner_type)) != 1:
                ner_type = [list(set(ner_type))[0]]
            assert len(set(ner_type)) == 1
            ner_type = ner_type[0]
            ner = (ner[0], ner[-1], ner_type) if len(ner) > 1 else (ner[0], ner[0], ner_type)
            sentence_lst.append(ner)
            ner = [i] if tag_head == u'B' or tag_head == u'I' else []
            ner_type = [tag_type] if tag_head == u'B' or tag_head == u'I' else []
        elif tag_head == u'B' or (tag_head == u'I' and (tag_type != prev_tag_type)):
            ner = [i]
            ner_type = [tag_type]
        elif tag_head == u'I' and prev_tag_type == tag_type and ner_is_exist:
            ner.append(i)
            ner_type.append(tag_type)

    if len(ner) > 0:
        ner_type = ner_type[0]
        ner = (ner[0], ner[-1], ner_type) if len(ner) > 1 else (ner[0], ner[0], ner_type)
        sentence_lst.append(ner)
    return sentence_lst


def range_metric_cnt(gold_range_list, predict_range_list):
    gold_range_set = set(gold_range_list)
    predict_range_set = set(predict_range_list)
    TP = gold_range_set & predict_range_set
    _g = len(gold_range_set)
    _p = len(predict_range_set)
    correct_cnt = len(TP)
    gold_cnt = _g
    predict_cnt = _p
    return correct_cnt, gold_cnt, predict_cnt


def to_str(s):
    """
    Convert to str
    :param s: something
    :return: str
    """
    if six.PY2 and isinstance(s, unicode):
        s = unicode_to_str_python2(s)
    elif not isinstance(s, str):
        s = str(s)
    return s


def unicode_to_str_python2(u):
    """
    In Python 2.x, convert unicode to str
    :param u: unicode
    :return: str
    """
    if six.PY2 and isinstance(u, unicode):
        u = u.encode('utf-8')
    return u


def str_to_unicode_python2(s):
    """
    In Python 2.x, convert str to unicode
    :param s: str
    :return: unicode
    """
    if six.PY2 and isinstance(s, str):
        s = s.decode('utf-8')
    return s