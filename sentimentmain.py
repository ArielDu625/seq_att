import data_utils as utils
import os
import sys
import numpy as np
import tensorflow as tf
import random

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import seaborn as sns
import pandas as pd
import time

import seq_att

SST_DIR = './sst/'
GLOVE_DIR ='./glove'
summaries_dir = './summaries/'
ckpt_dir = './ckpt/'

class Config(object):
    num_emb=None

    emb_dim = 100
    hidden_dim = 150 
    output_dim=None
    degree = 2

    num_epochs = 5000 
    
    dropout = 0.5
    reg=0.0001
    lmda = 0.5 

    #learning rate
    lr = 0.001
    begin_decay_epoch = 5 
    lrdecay_every_epoch = 1 
    emb_lr = 0.005
    
    batch_size = 20

    maxseqlen = None
    maxnodesize = None

    fine_grained = False
    
    #Add attention layer
    use_attention = True 
    attention_dim = 64 
    concat_dim = 300 
    #method: dot, general, location, or default concat
    method = "concat" 
    
    global_step = 0
    dev_every_step = 10
    test_every_epoch = 1 
    best_test_score = 0.0
    best_test_epoch = 0
    
def train(restore=False):
    config=Config()

    data,vocab = utils.load_sentiment_treebank(SST_DIR,config.fine_grained)
    train_set, dev_set, test_set = data['train'], data['dev'], data['test']
   
    num_emb = len(vocab)
    num_labels = 5 if config.fine_grained else 3
    for _, dataset in data.items():
        labels = [label for _, label in dataset]
        assert set(labels) <= set(xrange(num_labels)), set(labels)

    config.num_emb=num_emb
    config.output_dim = num_labels

    config.maxseqlen=utils.get_max_len_data(data)
    config.maxnodesize=utils.get_max_node_size(data)
    
    if config.fine_grained:
        classify_type = "fine_grained"
    else:
        classify_type = "binary"

    random.seed()
    np.random.seed()

    with tf.Graph().as_default():
        
        #model = seq_att.tf_seqLSTM(config)
        #model = seq_att.tf_seqLSTMAtt(config) 

        #model = seq_att.tf_seqbiLSTM(config)
        model = seq_att.tf_seqbiLSTMAtt(config)
        

        model_name = model.__class__.__name__
        print 'The model is running now:',model_name,classify_type
        
        ckpt_base = os.path.join(ckpt_dir, model_name, classify_type)
        summary_base = os.path.join(summaries_dir, model_name, classify_type)

        init=tf.global_variables_initializer()
        saver = tf.train.Saver()

        with tf.Session() as sess:
            tf.summary.FileWriter(summary_base, sess.graph)
                        
            sess.run(init)
            
            if restore:
                f = os.path.join(ckpt_base, 'lstm_weights')
                saver.restore(sess, f)
                
                test_score = model.evaluate(test_set, sess, isDev = False)
                print 'test_score:', test_score
                
                if config.use_attention:
                    visualize_attention(model, test_set, vocab, sess,ckpt_base)
            
            else:
                
                if config.emb_dim == 300:
                    glove_file = os.path.join(GLOVE_DIR, 'glove.840B.300d.txt')
                else:
                    tmp = 'glove.twitter.27B.' + str(config.emb_dim)+ 'd.txt'
                    glove_file = os.path.join(GLOVE_DIR,tmp)
                    
                glove_embeddings = utils.load_glove(glove_file, vocab, config.emb_dim)
                sess.run(model.embedding_init, feed_dict = {model.initial_emb: glove_embeddings})

                    
                avg_loss = model.train(train_set, test_set, sess,saver)
                print 'avg loss', avg_loss
                    
                        
def visualize_attention(model, data,voc, sess,ckpt_base):
    alphas, sentence,label,predlabel = model.plot_attention(data, voc, sess)
    print alphas
    print sentence
    print "pred label:", predlabel
    print "label", label
    
    length = len(sentence)
    indices = xrange(length)
    dummies = [""] * 15 

    words_with_index = ["{:03}_{}".format(i,w) for i,w in zip(indices, sentence)]
    words_for_annot = np.array([[w] for w in sentence])
    
    wordmap = pd.DataFrame({"words":words_with_index, "alphas":alphas[:length], "attention":dummies})
    datamap = wordmap.pivot("words","attention","alphas")

    plot = sns.heatmap(datamap, cmap = "YlGnBu", annot=False, xticklabels=False,yticklabels = words_for_annot ,fmt="")
    plt.yticks(rotation=0)
    fig = plot.get_figure()
    fname = "-".join([str(predlabel), str(label), time.strftime("%Y%m%d%H%M%S")]) + '.jpg'
    fdir = os.path.join(ckpt_base,fname)
    fig.savefig(fdir)
    


if __name__ == '__main__':
    if len(sys.argv) > 1:
        restore=True
    else:restore=False
    train(restore)

