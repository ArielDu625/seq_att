import numpy as np
import os,re
from data_utils import Vocab

def clean_str(string):
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string) 
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"\'t", " \'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()

def load_sentences(data_dir, clean = True):
    train_sentences = []
    train_labels = []
    test_sentences = []
    test_labels = []
    
    dictionary = set()

    np.random.seed()

    pos = os.path.join(data_dir,'rt-polarity.pos')
    with open(pos,'r') as f:
        for line in f.readlines():
            if clean:
                sentence = clean_str(line)
            else:
                sentence = line.strip()

            label = str(1)
            idx = np.random.randint(0,10)
            if idx >= 8:
                test_sentences.append(sentence)
                test_labels.append(label)
            else:
                train_sentences.append(sentence)
                train_labels.append(label)
            
            for word in sentence.split():
                dictionary.add(word)

    neg = os.path.join(data_dir, 'rt-polarity.neg')
    with open(neg,'r') as f:
        for line in f.readlines():
            if clean:
                sentence = clean_str(line)
            else:
                sentence = line.strip()

            label = str(0)
            idx = np.random.randint(0,10) 
            if idx >= 8:
                test_sentences.append(sentence)
                test_labels.append(label)
            else:
                train_sentences.append(sentence)
                train_labels.append(label)
            
            for word in sentence.split():
                dictionary.add(word)

    print "train sentence num:",len(train_sentences),len(train_labels)
    print "test sentence num:", len(test_sentences),len(test_labels)
    
    if clean:
        dict_path = os.path.join(data_dir, 'dict_cleaned.txt')
    else:
        dict_path = os.path.join(data_dir, 'dict.txt')
    with open(dict_path, 'w') as f:
        f.writelines('\n'.join(list(dictionary)))

    split_paths = {}
    for split in ['train', 'test']:
        split_paths[split] = os.path.join(data_dir, split)
        if not os.path.isdir(split_paths[split]):
            os.makedirs(split_paths[split])

    with open(os.path.join(data_dir, 'test','sents.txt'),'w') as f:
        f.writelines('\n'.join(test_sentences))
    with open(os.path.join(data_dir, 'test','labels.txt'),'w') as f:
        f.writelines('\n'.join(test_labels))

    with open(os.path.join(data_dir, 'train','sents.txt'),'w') as f:
        f.writelines('\n'.join(train_sentences))
    with open(os.path.join(data_dir, 'train','labels.txt'),'w') as f:
        f.writelines('\n'.join(train_labels))

def encode_sentence(data_dir):
    vocab = Vocab(os.path.join(data_dir, 'dict_cleaned.txt'))
    split_paths = {}
    for split in ['train','test']:
        split_paths[split] = os.path.join(data_dir, split)
        encodes = []
        with open(os.path.join(split_paths[split], 'sents.txt'),'r') as sf:
            for line in sf.readlines():
                sentence = line.strip().split()
                index = [str(vocab.encode(word)) for word in sentence]
                encode = " ".join(index)
                encodes.append(encode)

        with open(os.path.join(split_paths[split], 'index.txt'),'w') as wf:
            wf.writelines('\n'.join(encodes))

            
def split_mr():
    data_dir = './mr'
    load_sentences(data_dir)
    encode_sentence(data_dir)

if __name__ == "__main__":
    split_mr()
