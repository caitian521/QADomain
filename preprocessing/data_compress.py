import jieba
from snownlp import seg
import json
from preprocessing.retrieve.bm25 import BM25
from preprocessing.retrieve import utils
import os


def compress(sample):
    remain_seg = []
    sents = utils.get_sentences(sample["context"])
    doc = []

    for sent in sents:
        words = seg.seg(sent)
        words = utils.filter_stop(words)
        doc.append(words)
    #print(sample["context"])
    s = BM25(doc)
    similar_score = s.simall(seg.seg(sample["question"]))
    for i, score in enumerate(similar_score):
        if score > 0 :
            remain_seg.append(i)

    if remain_seg == []:
        remain_seg = [i for i in range(len(sents))]

    length = 0
    start_sent = 0
    start_in_sent = 0
    for i, sent in enumerate(sents):
        length += len(sent)
        if length > sample["start_position"] or i == len(sents)-1:
            start_sent = i
            break
        start_in_sent += len(sent)
    start_in_sent = sample["start_position"] - start_in_sent

    length = 0
    end_sent = 0
    end_in_sent = 0
    for i, sent in enumerate(sents):
        length += len(sent)
        if length > sample["end_position"] or i == len(sents)-1:
            end_sent = i
            break
        end_in_sent += len(sent)
    end_in_sent = sample["end_position"] - end_in_sent

    new_start, new_end = convert_long2short(sents, remain_seg, start_sent, end_sent, start_in_sent, end_in_sent)

    short_context = ""
    for seg_id in remain_seg:
        short_context += sents[seg_id]

    old_start = sample["start_position"]
    old_end = sample["end_position"]
    try:
        if sample["context"][old_start] == short_context[new_start] or sample["context"][old_end] != short_context[
            new_end] and sample["answer_type"] == "long-answer":
            pass
    except:
        remain_seg = [i for i in range(len(sents))]
        new_start = sample["start_position"]
        new_end = sample["end_position"]

    return remain_seg, sents, new_start, new_end

    #return remain_seg, sents, start_sent, end_sent, start_in_sent, end_in_sent

#def convert_short2long(start, end, sents, remain_segs):


def convert_long2short(sents, remain_seg, start_sent, end_sent,start_in_sent, end_in_sent):
    new_start = 0
    for seg_id in remain_seg:
        if seg_id < start_sent:
            new_start += len((sents[seg_id]))
        else:
            break
    new_end = 0
    for seg_id in remain_seg:
        if seg_id < end_sent:
            new_end += len((sents[seg_id]))
        else:
            break
    return new_start + start_in_sent, new_end + end_in_sent


if __name__ == '__main__':
    #samples = read_squad_data("/data/ct/Contest/QADomain/data/big_train_data.json", "/data/ct/Contest/QADomain/data/")
    # samples = json.loads("/data/ct/Contest/QADomain/data/dev.json")
    samples = []
    with open("/data/ct/Contest/QADomain/data/dev.json") as f:
        for sample in f:
            sample = json.loads(sample)
            remain_seg, sents, new_start, new_end = compress(sample)

            '''
            for seg_id in remain_seg:
                print(seg_id, sents[seg_id])
            print(start_sent, end_sent)
            print(sample["context"])
            print(sample["context"][old_start], sample["context"][old_end])
            print(sample["question"])
            print(sample["answer_text"])
            print(short_context[new_start], short_context[new_end])
            print(similar_score)
            print(old_start, old_end)
            '''
            '''
            print(sample["context"][old_start], sample["context"][old_end])
            for seg_id in remain_seg:
                print(seg_id, sents[seg_id])
            print(start_sent, end_sent)
            print(sample["context"])
            print(sample["question"])
            print(sample["answer_text"])
            '''