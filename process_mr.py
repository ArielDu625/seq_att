import numpy as np
import os
from data_utils import Vocab

def load_mr(data_dir):
    voc = Vocab(os.path.join(data_dir, 'dict_cleaned.txt'))
    
    split_paths = {}
    for split in ["train", "test"]:
        split_paths[split] = os.path.join(data_dir,split)
    
    data = {}
    max_sentence_length = 0
    count = 0
    sumlen = 0

    for split, path in split_paths.iteritems():
        sentencepath = os.path.join(path, "index.txt")
        labelpath = os.path.join(path, "labels.txt")
        
        splitdata=[]
        with open(sentencepath,'r') as sf, open(labelpath, 'r') as lf:
            for line,label in zip(sf.readlines(),lf.readlines()):
                sentence = line.strip()
                pair = {}
                pair['sentence'] = sentence
                pair['label'] = int(label.strip())
                
                splitdata.append(pair)
                if len(sentence) > max_sentence_length:
                    max_sentence_length = len(sentence)
                sumlen += len(sentence)
                count += 1
        data[split] = splitdata
    
    average_len = int(sumlen / count)
    return data, voc, max_sentence_length, average_len

def extract_data(data, fillnum = 56):
    seqdata = []
    seqlabels = []
    for pair in data:
        sentence = pair["sentence"]
        sidx = [int(i) for i in sentence.split()]
        seqdata.append(sidx)
        seqlabels.append(pair['label'])

    
    seqlngths = [len(s) for s in seqdata]

    maxl = max(seqlngths)
    #assert fillnum >= maxl

    seqarr = np.empty([len(seqdata), fillnum], dtype = "int32")
    seqarr.fill(-1)
    for i,s in enumerate(seqdata):
        if len(s) > fillnum:
            seqarr[i, 0:fillnum] = np.array(s[:fillnum], dtype="int32")
        else:
            seqarr[i, 0:len(s)] = np.array(s, dtype="int32")

    seqdata = seqarr

    return seqdata, seqlabels, seqlngths, maxl


def test_fn():
    data_dir = './mr'
    length = 56
    data, voc, max_sentence_length,average_len = load_mr(data_dir)
    print "vocab size:", voc.size()
    print "max sentence length:", max_sentence_length
    print "average length:", average_len

    for k,d in data.iteritems():
        print(k, len(d))
    
    d = data['test']
    a,b,c,_ = extract_data(d[0:1])
    print a,b,c

    for sentence in a:
        print sentence
        ss = []
        for idx in sentence:
            if idx != -1:
                word = voc.decode(idx)
                ss.append(word)
        print(" ".join(ss))

if __name__ == "__main__":
    test_fn()

