import os
import sys
import re
import numpy as np

import xlsxwriter

from itertools import chain


#==================================================
wtag=0
wrange=0
wrange_tag=0
noextraction=0
noannotation=0
flag = False

number_err = 0

temp = 0
#a: predict, b: gold
def check_duplicate(a, b, row):
    global flag
    global temp
    global wtag
    global wrange
    global wrange_tag
    global noextraction
    global noannotation
    c = tuple(sorted(set(a[:2]))) + tuple(sorted(set(b[:2])))
    d = tuple(sorted(set(b[:2]))) + tuple(sorted(set(a[:2])))
    #no annotation - 1234 and not duplicate
    if((c==tuple(sorted(c)) or (d==tuple(sorted(d)))) and len(c)==len(set(c))):
        temp+=1
    #completely the same but tag
    elif(len(c)==2*len(set(c))):
        wtag+=1
        worksheet.write(row+1+a[0], 3, "T")
        flag = True
    #have a bounary and tags are correct
    elif(a[2]==b[2]):
        wrange+=1
        worksheet.write(row+1+a[0], 3, "R")
        flag = True
    #have a bounary but wrong tag
    else:
        wrange_tag+=1
        worksheet.write(row+1+a[0], 3, "W")
        flag = True
#===============================================
pattern_num = re.compile(r'[0-9]')
CHAR_PADDING = u"_"
UNKWORD = u"UNKNOWN"
PADDING = u"PADDING"
BOS = u"<BOS>"


def flatten(l):
    return list(chain.from_iterable(l))


def replace_num(text):
    return pattern_num.sub(u'0', text)


def build_vocab(dataset, min_count=0):
    vocab = {}
    vocab[PADDING] = len(vocab)
    vocab[UNKWORD] = len(vocab)
    vocab_cnt = {}
    for d in dataset:
        for w in d:
            vocab_cnt[w] = vocab_cnt.get(w, 0) + 1

    for w, cnt in sorted(vocab_cnt.items(), key=lambda x: x[1], reverse=True):
        if cnt >= min_count:
            vocab[w] = len(vocab)

    return vocab


def build_tag_vocab(dataset, tag_idx=-1):
    pos_tags = list(set(flatten([[w[tag_idx] for w in word_objs]
                                 for word_objs in dataset])))
    pos_tags = sorted(pos_tags)
    vocab = {}
    for pos in pos_tags:
        vocab[pos] = len(vocab)
    return vocab


def write_vocab(filename, vocab):
    f = open(filename, 'w')
    for w, idx in sorted(vocab.items(), key=lambda x: x[1]):
        line = '\t'.join([w.encode('utf-8'), str(idx)])
        f.write(line + '\n')


def load_vocab(filename):
    vocab = {}
    for l in open(filename):
        w, idx = l.decode('utf-8').strip().split(u'\t')
        vocab[w] = int(idx)
    return vocab


def read_raw_file(filename, delimiter=u' '):
    sentences = []
    for l in open(filename):
        words = l.decode('utf-8').strip().split(delimiter)
        words = [_.strip() for _ in words]
        if len(words) and len(words[0]):
            words = [(w, -1) for w in words]
            sentences.append(words)
    return sentences


def read_conll_file(filename, delimiter=u'\t', input_idx=0, output_idx=-1):
    sentence = []
    sentences = []
    n_features = -1
    for line_idx, l in enumerate(open(filename, 'r')):
        l_split = l.strip().decode('utf-8').split(delimiter)
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
                val = (str(len(l_split)), str(len(line_idx)))
                err_msg = 'Invalid input feature sizes: "%s". \
                Please check at line [%s]' % val
                raise ValueError(err_msg)
            sentence.append(l_split)
    if len(sentence) > 0:
        sentences.append(sentence)
    return sentences


def load_glove_embedding(filename, vocab):
    word_ids = []
    word_vecs = []
    for i, l in enumerate(open(filename)):
        l = l.decode('utf-8').split(u' ')
        word = l[0].lower()

        if word in vocab:
            word_ids.append(vocab.get(word))
            vec = l[1:]
            vec = map(float, vec)
            word_vecs.append(vec)
    word_ids = np.array(word_ids, dtype=np.int32)
    word_vecs = np.array(word_vecs, dtype=np.float32)
    return word_ids, word_vecs


def load_glove_embedding_include_vocab(filename):
    word_vecs = []
    vocab = {}
    vocab[PADDING] = len(vocab)
    vocab[UNKWORD] = len(vocab)

    for i, l in enumerate(open(filename)):
        l = l.decode('utf-8').split(u' ')
        word = l[0].lower()
        if word not in vocab:
            vocab[word] = len(vocab)
            vec = l[1:]
            vec = map(float, vec)
            word_vecs.append(vec)

    # PADDING, UNKWORD
    word_vecs.insert(0, np.random.random((len(vec),)))
    word_vecs.insert(0, np.random.random((len(vec),)))

    word_vecs = np.array(word_vecs, dtype=np.float32)
    return word_vecs, vocab

# Conll 03 shared task evaluation code (IOB format only)


def eval_accuracy(predict_lists):
    sum_cnt = 0
    correct_cnt = 0

    for (gold, pred) in predict_lists:
        sum_cnt += len(gold)
        correct_cnt += sum(gold == pred)

    sum_cnt = 1 if sum_cnt == 0 else sum_cnt
    accuracy = float(correct_cnt) / sum_cnt
    accuracy = accuracy * 100.0
    return accuracy


row = 0
col = 0
workbook = xlsxwriter.Workbook('/Users/binhnguyen/Desktop/test-gold.xlsx')
worksheet = workbook.add_worksheet()
worksheet.write(row, col, "Item")
worksheet.write(row, col + 1, "Gold")
worksheet.write(row, col + 2, "Predict")
worksheet.write(row, col + 3, "Type-Predict")
row + 1



def conll_eval(gold_predict_pairs, flag=True, tag_class=None):
    """
    Args : gold_predict_pairs = [
                                    gold_tags,
                                    predict_tags
                                ]

    """

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
    # ========================================================================
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

    errorN = []
    error = {}
    listError = []
    accuracy = {}

    global row, col
    
    # ========================================================================

    for tag_name in tag_class:
        # ======================
        error[tag_name] = {}
        error[tag_name]['error'] = 0
        accuracy[tag_name] = 0

        # ======================
        cnt_phrases_dict[tag_name] = {}
        cnt_phrases_dict[tag_name]['predict_cnt'] = 0
        cnt_phrases_dict[tag_name]['gold_cnt'] = 0
        cnt_phrases_dict[tag_name]['correct_cnt'] = 0

    for i in xrange(len(gold_tags)):
        gold = gold_tags[i]
        predict = predict_tags[i]

        gold_phrases = IOB_to_range_format_one(gold, is_test_mode=True)
        predict_phrases = IOB_to_range_format_one(predict, is_test_mode=True)
        #=======================================
        

        gold_range_set = set(gold_phrases)
        predict_range_set = set(predict_phrases)
        global number_err
        if (len(gold_range_set - predict_range_set) != 0):
            number_err+=1
            tool(gold_range_set, predict_range_set, row)

            worksheet.write(row, col, number_err)
            row += 1

            for l in xrange(len(fileself[i])):
                worksheet.write(row, col, fileself[i][l].split()[0])
                worksheet.write(row, col + 1, fileself[i][l].split()[-1])
                worksheet.write(row, col + 2, predict_tags[i][l])
                row += 1

            
        #=======================================
        for p in gold_phrases:
            tag_name = p[-1]
            n_phrases_dict['gold'][tag_name] = n_phrases_dict['gold'].get(tag_name, 0) + 1

        for p in predict_phrases:
            tag_name = p[-1]
            n_phrases_dict['predict'][tag_name] = n_phrases_dict['predict'].get(tag_name, 0) + 1

        #================
        temp = []
        #================
        for tag_name in tag_class:
            _gold_phrases = [_ for _ in gold_phrases if _[-1] == tag_name]
            _predict_phrases = [_ for _ in predict_phrases if _[-1] == tag_name]

            correct_cnt, gold_cnt, predict_cnt, add, err = range_metric_cnt(_gold_phrases, _predict_phrases)

            #============================================

            num, collecttags = collect(_gold_phrases, _predict_phrases)

            if add and i not in errorN:
                errorN.append(i)
            if add:
                temp.append(collecttags)
            error[tag_name]['error'] += err if len(_gold_phrases) > 0 else 0
            accuracy[tag_name] += num
            #============================================
            cnt_phrases_dict[tag_name]['gold_cnt'] += gold_cnt if len(_gold_phrases) > 0 else 0
            cnt_phrases_dict[tag_name][
                'predict_cnt'] += predict_cnt if len(_predict_phrases) > 0 else 0
            cnt_phrases_dict[tag_name]['correct_cnt'] += correct_cnt

        # ----------------------------------
        if len(temp):
            listError.append(temp)
    

    #print cnt_phrases_dict
    #print error
    #print listError
    #print accuracy
    cnt = 0
    for i in listError:
        cnt += len(i)
    #print len(listError)
    #----------------------------------
    lst_gold_phrase = n_phrases_dict['gold']
    lst_predict_phrase = n_phrases_dict['predict']
    num_gold_phrase = sum(n_phrases_dict['gold'].values())
    num_predict_phrase = sum(n_phrases_dict['predict'].values())
    phrase_info = [num_gold_phrase, num_predict_phrase, lst_gold_phrase, lst_predict_phrase]

    for tag_name in tag_class:
        recall = cnt_phrases_dict[tag_name][
            'correct_cnt'] / float(cnt_phrases_dict[tag_name]['gold_cnt']) if cnt_phrases_dict[tag_name]['gold_cnt'] else 0.0
        precision = cnt_phrases_dict[tag_name]['correct_cnt'] / float(
            cnt_phrases_dict[tag_name]['predict_cnt']) if cnt_phrases_dict[tag_name]['predict_cnt'] else 0.0
        sum_recall_precision = 1.0 if recall + precision == 0.0 else recall + precision
        f_measure = (2 * recall * precision) / (sum_recall_precision)
        evals[tag_name] = [precision * 100.0, recall * 100.0, f_measure * 100.0]

    correct_cnt = sum([ev['correct_cnt'] for tag_name, ev in cnt_phrases_dict.items()])
    gold_cnt = sum([ev['gold_cnt'] for tag_name, ev in cnt_phrases_dict.items()])
    predict_cnt = sum([ev['predict_cnt'] for tag_name, ev in cnt_phrases_dict.items()])

    recall = correct_cnt / float(gold_cnt) if gold_cnt else 0.0
    precision = correct_cnt / float(predict_cnt) if predict_cnt else 0.0
    sum_recall_precision = 1.0 if recall + precision == 0.0 else recall + precision
    f_measure = (2 * recall * precision) / (sum_recall_precision)
    evals['All_Result'] = [precision * 100.0, recall * 100.0, f_measure * 100.0]
    #=======================================================

    resultself = codecs.open("/Users/binhnguyen/Downloads/deep-crf/deepcrf/result.txt", 'w+', 'utf-8')
    for i in errorN:
        resultself.write(str(errorN.index(i) + 1) + "\r\n")
        for l in xrange(len(fileself[i])):
            resultself.write(fileself[i][l].split()[0] + "\t" + fileself[i][l].split()[-1] + "\t" + predict_tags[i][l] + "\r\n")

    another = open("check.txt", "w+")
    cnt = 0
    for i in listError:
        cnt += 1
        another.write(str(cnt) + " " + str(i) + "\r\n")
    # ======================================================

    global wrange, wtag, wrange_tag, noannotation
    print("wrong range: ", wrange, ",wrong tag: ", wtag, ", wrong range & tag: ", wrange_tag, ", no annotaion: ", noannotation)
    print(number_err)

    workbook.close()
    return evals, phrase_info

def tool(gold_range_set, predict_range_set, row):
    

    global noannotation, flag, temp, number_err
    predict = sorted(predict_range_set - gold_range_set)
    gold = sorted(gold_range_set -predict_range_set)
    for a in gold:
        if(len(predict) == 0):
            noannotation+=1
            worksheet.write(row+1+a[0], 3, "E")

        temp = 0
        for b in predict:
            check_duplicate(a,b,row)
            if(temp==len(predict)):
                noannotation+=1
                worksheet.write(row+1+a[0], 3, "E")
                flag = True
                temp=0
            if flag:
                flag = False
                break


def IOB_to_range_format_one(tag_list, is_test_mode=False):
    sentence_lst = []
    ner = []
    ner_type = []
    # print tag_list
    for i in xrange(len(tag_list)):
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

#==========================================
def collect(gold_range_list, predict_range_list):
    gold_range_set = set(gold_range_list)
    predict_range_set = set(predict_range_list)
    TP = gold_range_set & predict_range_set
    temp = predict_range_set - TP
    num = len(TP)
    return num, temp
#==========================================

def range_metric_cnt(gold_range_list, predict_range_list):
    add = False
    gold_range_set = set(gold_range_list)
    predict_range_set = set(predict_range_list)
    TP = gold_range_set & predict_range_set
    _g = len(gold_range_set)
    _p = len(predict_range_set)
    correct_cnt = len(TP)
    gold_cnt = _g
    predict_cnt = _p
    if predict_cnt != correct_cnt:
        add = True
    err = len(predict_range_set - TP)

    #========================================
    
    #========================================

    return correct_cnt, gold_cnt, predict_cnt, add, err


def parse_to_word_ids(sentences, xp, vocab, UNK_IDX, idx=0):
    x_data = [xp.array([vocab.get(w[idx].lower(), UNK_IDX)
                        for w in sentence], dtype=xp.int32)
              for sentence in sentences]
    return x_data


def parse_to_char_ids(sentences, xp, vocab_char, UNK_IDX, idx=0):
    x_data = [[xp.array([vocab_char.get(c, UNK_IDX) for c in w[idx]],
                        dtype=xp.int32)
               for w in sentence]
              for sentence in sentences]
    return x_data


def parse_to_tag_ids(sentences, xp, vocab, UNK_IDX, idx=-1):
    x_data = [xp.array([vocab.get(w[idx], UNK_IDX)
                        for w in sentence], dtype=xp.int32)
              for sentence in sentences]
    return x_data


def parse_raw_text(sentence, xp, vocab, vocab_char, UNK_IDX, CHAR_UNK_IDX):
    x_data = [xp.array([vocab.get(w.lower(), UNK_IDX) for w in sentence],
                       dtype=xp.int32)]
    x_data_char = [[xp.array([vocab_char.get(c, CHAR_UNK_IDX) for c in w],
                             dtype=xp.int32) for w in sentence]]

    return x_data, x_data_char


def uniq_tagset(alltags_list, tag_names=[]):
    for tags in alltags_list:
        for tag in tags:
            tagname = '-'.join(tag.split(u'-')[1:])
            if tagname != u'' and tagname not in tag_names:
                tag_names.append(tagname)
    return tag_names
