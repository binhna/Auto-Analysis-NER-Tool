import re

import tool2
import util_talbes


def load_file(filename):
    alltags_list = []
    tags = []
    for l in open(filename):
        tag = l.decode('utf-8').strip()
        if tag == u'':
            # sentence split
            alltags_list.append(tags)
            tags = []
        else:
            tags.append(tag)
    if len(tags):
        alltags_list.append(tags)
        tags = []
    return alltags_list


def uniq_tagset(alltags_list, tag_names=[]):
    for tags in alltags_list:
        for tag in tags:
            tagname = '-'.join(tag.split(u'-')[1:])
            if tagname != u'' and tagname not in tag_names:
                tag_names.append(tagname)
    return tag_names


def replace_S_E_tags(tag_lists):
    def rep_func(tag):
        tag = re.sub(r'^S-', "B-", tag)
        tag = re.sub(r'^E-', "I-", tag)
        return tag
    return [[rep_func(tag) for tag in tags] for tags in tag_lists]


def run(gold_file, predicted_file, **args):
    # list of tag sentence. ex: [[B-PER,O,O,O,B-ORG,0,0], [O,B-PER,I-PER]]
    gold_tags = load_file(gold_file)
    predicted_tags = load_file(predicted_file)

    # tag set
    # tag_names: number of unique tags in both predicted and gold, ex: tag_names = [PER, ORG, LOC]
    tag_names = []
    tag_names = uniq_tagset(gold_tags, tag_names)
    tag_names = uniq_tagset(predicted_tags, tag_names)

    
    gold_tags = replace_S_E_tags(gold_tags)
    predicted_tags = replace_S_E_tags(predicted_tags)
    gold_predict_pairs = [gold_tags, predicted_tags]
    result, phrase_info, detail_error, detail_error_gold = tool2.conll_eval(gold_predict_pairs, flag=False, tag_class=tag_names)

    #print phrase_info

    table = util_talbes.SimpleTable()
    table.set_header(('Tag Name', 'Precision', 'Recall', 'F_measure'))

    all_result = result['All_Result']
    table.add_row(['All_Result'] + all_result)

    for key in result.keys():
        if key != 'All_Result':
            r = result[key]
            table.add_row([key] + r)

    table.print_table()

    #print detail error
    table_error = util_talbes.SimpleTable()
    table_error.set_header(('Tags', 'Correct', 'Error', 'Total', 'No Annotation', 'Wrong Range', 'Wrong Range & Tag', 'Wrong Tag'))

    all_tags = detail_error["All_Tags"]
    table_error.add_row(["All Tags"] + all_tags)

    for key in detail_error.keys():
        if key != "All_Tags":
            r = detail_error[key]
            table_error.add_row([key] + r)

    table_error.print_table()

    #print detail error gold
    table_error_gold = util_talbes.SimpleTable()
    table_error_gold.set_header(('Tags', 'Correct', 'Error', 'Total', 'No Extraction', 'Wrong Range', 'Wrong Range & Tag', 'Wrong Tag'))

    all_tags = detail_error["All_Tags"]
    table_error_gold.add_row(["All Tags"] + all_tags)
    for key in detail_error_gold.keys():
        if key != "All_Tags":
            r = detail_error[key]
            table_error_gold.add_row([key] + r)

    table_error_gold.print_table()

    # accuracy = util.eval_accuracy(gold_predict_pairs, flag=False)
    # print 'Tag Accuracy:', accuracy
run("/Users/binhnguyen/Downloads/deep-crf/gold.txt", "/Users/binhnguyen/Downloads/deep-crf/predicted.txt")