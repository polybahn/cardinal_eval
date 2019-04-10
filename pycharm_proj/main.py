# Reads annotators' results
import os
import pandas as pd
import json
from operator import itemgetter
import numpy as np
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu, corpus_bleu
from nltk.stem import PorterStemmer
from nltk.tokenize import sent_tokenize, word_tokenize
from sari import SARIsent

data_path='/Users/polybahn/Desktop/cardinaldl/evaluation/annotator results new/human/'


def get_df(path):
    return pd.read_csv(path)


m = {'simplified sentence ': 'partial_text',
     'partial start time ' : 'partial_start_time',
     'owner ': 'owner',
     'original action ': 'ori_action',
     'most similar action ': 'action',
     'speed ' : 'speed',
     'duration ': 'duration',
     'rotation ': 'rotation',
     'translation ' : 'translation',
     'target ': 'target',
     'prop ' : 'prop',
     'location modifier ': 'modifierLocation',
     'direction modifier ': 'modifierDirection',
     'manner ': 'manner',
     'emotion ': 'emotion'
    }

ps = PorterStemmer()


def construct_dict(df):
    r = dict()
    for index, row in df.iterrows():
        key = row['action block 1'].replace("'", '"')
        l = list()
        for i in range(1, 13):
            i = str(i)
            if not row['original action ' + i] == row['original action ' + i]:
                    break
            obj = dict()
            for k, v in m.items():
                if isinstance(row[k+i], str):
                    assign = row[k+i].lower()
                    if 'null' in assign:
                        continue
                    if assign == 'yes':
                        assign = 'true'
                    if assign == 'no':
                        assign = 'false'
                    if assign == 'short':
                        assign = 1.5
                    if assign == 'neutral':
                        assign = 2
                    if assign == 'long':
                        assign = 4
                    if assign == 'fast':
                        assign = 1
                    if assign == 'neutral':
                        assign = 0.5
                    if assign == 'slow':
                        assign = 0.2
                elif not row[k+i] == row[k+i]:
                    assign = 'null'
                else:
                    assign = row[k+i]
                obj[v] = assign
            l.append(obj)
        if l:
            r[key] = l
    return r


def get_testers_answer():
    # read evaluator's answer
    files = [os.path.join(data_path, p) for p in os.listdir(data_path) if p.endswith('.csv')]
    # construct tester's answers
    testers = dict()
    for file in files:
        name = file.split('-')[-2].strip()
        print name
        df = get_df(file)
        testers[name] = construct_dict(df)
    return testers


def extract_simp(json_list):
    return [j['partial_text'] for j in json_list]


def extract_all(key, cand, all_ds):
    res = []
    for k,v in all_ds.items():
        if key in v and len(v[key]) > 0:
            ref = extract_simp(v[key])
            res += ref
    return res


def levenshtein(seq1, seq2):
    size_x = len(seq1) + 1
    size_y = len(seq2) + 1
    matrix = np.zeros((size_x, size_y))
    for x in xrange(size_x):
        matrix [x, 0] = x
    for y in xrange(size_y):
        matrix [0, y] = y

    for x in xrange(1, size_x):
        for y in xrange(1, size_y):
            if seq1[x-1] == seq2[y-1]:
                matrix [x,y] = min(
                    matrix[x-1, y] + 1,
                    matrix[x-1, y-1],
                    matrix[x, y-1] + 1
                )
            else:
                matrix [x,y] = min(
                    matrix[x-1,y] + 1,
                    matrix[x-1,y-1] + 1,
                    matrix[x,y-1] + 1
                )
    # print (matrix)
    return matrix[size_x - 1, size_y - 1]


def tokenize_and_stem(sent):
    # with stemmer
    # return [ps.stem(w) for w in word_tokenize(sent.lower())]
    # withour stemmer
    return word_tokenize(sent.lower())


# extract references that are with minimum distance with this candidate
def extract_med(ori_str, cand, all_ds):
    res = []
    for annotator, all_entries in all_ds.items():
        if ori_str in all_entries and len(all_entries[ori_str]) > 0:
            full_answer = extract_simp(all_entries[ori_str]) # extract all answers of this annoator on this block
            nearest_answer = min([(levenshtein(tokenize_and_stem(answer), tokenize_and_stem(cand)), answer) for answer in full_answer],
                                key=itemgetter(0))[1]
            res.append(nearest_answer)
    return res


# redefine bleu
def calc_bleu(candidate, references):
    if not candidate:
        return 0
    references = filter(lambda x: x, references)
    cc = SmoothingFunction()
    # score = sentence_bleu([ref.lower().strip('.').strip().split(' ') for ref in references], candidate.lower().strip('.').strip().split(' '),
    #                       smoothing_function=cc.method3)
    score = sentence_bleu([tokenize_and_stem(ref) for ref in references],
                          tokenize_and_stem(candidate),
                          smoothing_function=cc.method3)
    return score


def eval_avg_sentence_bleu(predicted, testers):
    cnt = 0
    score_sum = .0
    for k, v in predicted.items():
        candidates = v
        for cand in candidates:
            references = extract_med(k, cand, testers)
            if not references:
                continue
            score_sum += calc_bleu(cand, references)
            cnt += 1
    print(score_sum / cnt)


def eval_corpus_bleu(predicted, testers):
    hypothesis = list()
    references_list = list()
    for k, v in predicted.items():
        candidates = v
        for cand in candidates:
            references = extract_med(k, cand, testers)
            if not references:
                continue
            # references_list.append([ref.lower().strip('.').strip().split(' ') for ref in references])
            # hypothesis.append(cand.lower().strip('.').strip().split(' '))
            references_list.append([tokenize_and_stem(ref)for ref in references ])
            hypothesis.append(tokenize_and_stem(cand))
    # corpus bleu
    cc = SmoothingFunction()
    corpus_bleu_score = corpus_bleu(references_list, hypothesis,
                                    smoothing_function=cc.method3)
    print corpus_bleu_score
    return corpus_bleu_score


def eval_avg_sentence_SARI(predicted, testers):
    # SARI
    cnt = 0
    score_sum = .0
    for k, v in predicted.items():
        candidates = v
        for cand in candidates:
            references = extract_med(k, cand, testers)
            if not references:
                continue
            score_sum += SARIsent(k, cand, references)
            cnt += 1
    # print(cnt)
    print(score_sum / cnt)




if __name__=="__main__":
    # read evaluator's answer
    testers = get_testers_answer()

    # read YATS results
    yats = json.load(open('/Users/polybahn/Desktop/cardinaldl/evaluation/yats/yats.txt', 'r'))
    # read e2e results
    neural_res = json.load(open('/Users/polybahn/Desktop/cardinaldl/evaluation/system results/neural_500.txt', 'r'))
    # read sys results
    with open('/Users/polybahn/Desktop/cardinaldl/evaluation/system results/new_500.txt') as f:
        for line in f:
            sysout = eval(line)


    # produce example
    action_block = "Her gaze falls on Wall-E. He waves politely."
    print("Given action block: \n\n" + action_block +
          "\n\nOur Annotator produce answer:\n\n")

    cand1 = sysout[action_block]
    # cand2 = yats[action_block]
    # cand3 = neural_res[action_block]
    print(cand1)
    # print(cand2)
    # print(cand3)
    answer_1 = extract_simp(testers['emma'][action_block])

    proper_emma = min([(levenshtein(tokenize_and_stem(answer), tokenize_and_stem(cand1[0])), answer) for answer in answer_1],
        key=itemgetter(0))[1]
    print(proper_emma)




    # eval sentence avg bleu
    # eval bleu
    print "\nAVG BLEU Score:"
    print "AVG BLEU-- OURS:"
    eval_avg_sentence_bleu(sysout, testers)
    print "AVG BLEU-- YATS:"
    eval_avg_sentence_bleu(yats, testers)
    print "AVG BLEU-- NTS:"
    eval_avg_sentence_bleu(neural_res, testers)


    # eval corpus bleu
    print "\nCORPUS BLEU Score:"
    print "BLEU-- OURS:"
    eval_corpus_bleu(sysout, testers)
    print "BLEU-- YATS:"
    eval_corpus_bleu(yats, testers)
    print "BLEU-- NTS:"
    eval_corpus_bleu(neural_res, testers)


    # eval sari
    print "\nSENTENCE AVG SARI Score:"
    print "SARI-- OURS:"
    eval_avg_sentence_SARI(sysout, testers)
    print "SARI-- YATS:"
    eval_avg_sentence_SARI(yats, testers)
    print "SARI-- NTS:"
    eval_avg_sentence_SARI(neural_res, testers)





