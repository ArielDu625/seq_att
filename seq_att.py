
#from __future__ import print_function
import numpy as np
import tensorflow as tf
import os
import sys

from random import shuffle
from tensorflow.contrib import rnn
from data_utils import extract_seq_data

ckpt_dir = './ckpt/'
summaries_dir = './summaries/'

class tf_seqLSTM(object):

    def add_placeholders(self):

        self.batch_len = tf.placeholder(tf.int32,name="batch_len")

        self.max_time = tf.placeholder(tf.int32,name="max_time")
        
        self.initial_emb = tf.placeholder(tf.float32, shape=[self.num_emb, self.emb_dim], name = "initial_emb")

        self.input = tf.placeholder(tf.int32,shape=[None, self.config.maxseqlen],name="input")
        self.mask_idx = tf.placeholder(tf.float32, shape = [None, self.config.maxseqlen], name="mask_idx")

        self.labels = tf.placeholder(tf.int32,shape=None, name="labels")

        self.dropout = tf.placeholder(tf.float32,name="dropout")
        self.lr = tf.placeholder(tf.float32, name = "learning_rate")

        self.lngths = tf.placeholder(tf.int32,shape=None, name="lnghts")
        

    def __init__(self,config):
        self.emb_dim = config.emb_dim
        self.hidden_dim = config.hidden_dim
        self.num_emb = config.num_emb
        self.output_dim = config.output_dim
        self.config=config
        self.batch_size=config.batch_size
        self.reg=self.config.reg
        self.internal=4  #paramter for sampling sequences coresponding to subtrees 
        #attention setting
        self.attention_dim = self.config.attention_dim
        
        self.add_placeholders()
        
        with tf.variable_scope("Embed"):
            W = tf.get_variable("embedding", [self.num_emb, self.emb_dim])
            self.embedding_init = W.assign(self.initial_emb)
        
        emb_input = self.add_embedding()
        output_states = self.compute_states(emb_input)
        logits = self.create_output(output_states)
        self.pred = tf.nn.softmax(logits)
        self.loss,self.total_loss = self.calc_loss(logits)
        
        self.predlabel = tf.squeeze(tf.argmax(self.pred, axis = 1, output_type = tf.int32))
        self.accuracy = self.accuracy()        
        
        with tf.name_scope("model_scalar"):
            tf.summary.scalar('loss', self.loss)
            tf.summary.scalar('total_loss', self.total_loss)
            tf.summary.scalar("learning_rate", self.lr)
            tf.summary.scalar("accuracy", self.accuracy)

        self.train_op1,self.train_op2 = self.add_training_op()
        
        self.merged = tf.summary.merge_all()
        
        #class name
        model_name = self.__class__.__name__ 
        if self.config.fine_grained:
            classify_type = "fine_grained"
        else:
            classify_type = "binary"
        
        self.summary_base = os.path.join(summaries_dir,model_name,classify_type) 
        self.ckpt_base = os.path.join(ckpt_dir, model_name,classify_type)

        self.train_writer = tf.summary.FileWriter(self.summary_base + '/train')
        self.dev_writer = tf.summary.FileWriter(self.summary_base + '/dev')
    
    def add_embedding(self):

        with tf.variable_scope("Embed",reuse = True):
                
            embedding = tf.get_variable("embedding", [self.num_emb, self.emb_dim],
                        initializer = tf.random_uniform_initializer(-0.05, 0.05), 
                        regularizer = tf.contrib.layers.l2_regularizer(0.0))
                

            with tf.name_scope("Embed"):
                tf.summary.histogram('embedding', embedding)

            ix=tf.to_int32(tf.not_equal(self.input,-1))*self.input
            emb = tf.nn.embedding_lookup(embedding,ix)
            emb = emb * tf.to_float(tf.not_equal(tf.expand_dims(self.input,2),-1))
            return emb

    def compute_states(self,emb):

        def unpack_sequence(tensor):
            return tf.unstack(tf.transpose(tensor, perm=[1, 0, 2]))


        with tf.variable_scope("Composition",initializer=tf.contrib.layers.xavier_initializer(),
                regularizer=tf.contrib.layers.l2_regularizer(self.reg)):
            cell = rnn.BasicLSTMCell(self.hidden_dim)
            cell = rnn.DropoutWrapper(cell,output_keep_prob=self.dropout,input_keep_prob=self.dropout)
            outputs, state = rnn.static_rnn(cell,unpack_sequence(emb),sequence_length=self.lngths,dtype=tf.float32)

        #use average of hiddens 
        sum_out=tf.reduce_sum(tf.stack(outputs),[0])
        sent_rep = tf.div(sum_out,tf.expand_dims(tf.to_float(self.lngths),1))
        final_state=sent_rep
        return final_state

    def create_output(self,rnn_out):

        with tf.variable_scope("Projection",regularizer = tf.contrib.layers.l2_regularizer(self.reg)):
            Wp = tf.get_variable("Wp",[self.output_dim,self.hidden_dim],
                    initializer = tf.random_uniform_initializer(-0.05,0.05))
            bp = tf.get_variable("bp",[self.output_dim],initializer = tf.constant_initializer(0.0),
                    regularizer = tf.contrib.layers.l2_regularizer(0.0))

            logits=tf.matmul(rnn_out,Wp,transpose_b=True)+bp
            
            with tf.name_scope("Projection"):
                tf.summary.histogram("Wp",Wp)
                tf.summary.histogram("bp",bp)

            return logits

    def calc_loss(self,logits):
    
        l1=tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits = logits, labels = self.labels)
        loss=tf.reduce_sum(l1,[0])
        reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        regpart=tf.add_n(reg_losses)
        total_loss=loss + self.config.lmda *regpart
        return loss,total_loss

    def accuracy(self):
        y = tf.squeeze(tf.argmax(self.pred, axis = 1,output_type = tf.int32))
        
        right = tf.to_float(tf.equal(y, self.labels))
        _num = tf.to_float(tf.ones_like(y))
        acc = tf.reduce_sum(right) / tf.reduce_sum(_num) 
        
        return acc


    def add_training_op(self):
        loss=self.total_loss
        opt1=tf.train.AdagradOptimizer(self.lr)
        opt2=tf.train.AdagradOptimizer(self.config.emb_lr)

        ts=tf.trainable_variables()
        gs=tf.gradients(loss,ts)
        gs_ts=zip(gs,ts)

        gt_emb,gt_nn=[],[]
        for g,t in gs_ts:
            if "embedding" in t.name:
                gt_emb.append((g,t))
            else:
                gt_nn.append((g,t))

        train_op1=opt1.apply_gradients(gt_nn)
        
        train_op2=opt2.apply_gradients(gt_emb)
        train_op=[train_op1,train_op2]

        return train_op

    def cal_mask(self, seqdata, num):
        mask_idx = []
        for seq in seqdata:
            idx = [0 if element==num else 1 for element in seq]
            mask_idx.append(idx)

        return mask_idx

    def train(self,train_data,test_data, sess,saver):
        
        previous_acc = 0
        for epoch in range(self.config.num_epochs):
            print "epoch", epoch

            avg_loss = self.train_epoch(train_data, sess)
            print "average loss:", avg_loss
            
            if epoch % self.config.test_every_epoch == 0:
                test_acc = self.evaluate(test_data, sess, isDev = False)
                print "test accuracy:", test_acc
                
                if epoch >= self.config.begin_decay_epoch and test_acc <= previous_acc:
                    self.config.lr = self.config.lr * 0.5 
                print "learning rate:", self.config.lr
                previous_acc = test_acc

                if test_acc > self.config.best_test_score:
                    self.config.best_test_score = test_acc
                    self.config.best_test_epoch = epoch
                    if not os.path.isdir(self.ckpt_base):
                        os.makedirs(self.ckpt_base)
                    
                    saver.save(sess, os.path.join(self.ckpt_base, "lstm_weights"))
                    with open(os.path.join(self.ckpt_base, "best_test_data.txt"), 'w') as f:
                        f.write('best_test_score:%s best_test_epoch:%s\n' % (self.config.best_test_score, self.config.best_test_epoch))
                print "best_test_score:", self.config.best_test_score


    def train_epoch(self, train_data, sess):
        shuffle(train_data)
        losses=[]
        for i in range(0,len(train_data),self.batch_size):
            batch_size = min(i+self.batch_size,len(train_data))-i
            batch_data=train_data[i:i+batch_size]
            
            seqdata,seqlabels,seqlngths,max_len=extract_seq_data(batch_data
                                                         ,self.internal,self.config.maxseqlen)

            mask_idx = self.cal_mask(seqdata,-1)
            
            feed={self.input:seqdata,
                    self.mask_idx:mask_idx,
                    self.labels:seqlabels,
                    self.dropout:self.config.dropout,
                    self.lr:self.config.lr,
                    self.lngths:seqlngths,
                    self.batch_len:len(seqdata),
                    self.max_time:max_len}
            
            summary, loss,_,_=sess.run([self.merged, self.loss,self.train_op1,self.train_op2],feed_dict=feed)
            self.train_writer.add_summary(summary,self.config.global_step)
            self.config.global_step += 1

            losses.append(loss)
            avg_loss=np.mean(losses)
            sstr='avg loss %.2f at example %d of %d\r' % (avg_loss, i, len(train_data))
            sys.stdout.write(sstr)
            sys.stdout.flush()

        return np.mean(losses)
    
    def evaluate(self,data,sess,isDev = False):
        num_correct=0
        total_data=0
        for i in range(0,len(data),self.batch_size):
            batch_size = min(i+self.batch_size,len(data))-i
            batch_data=data[i:i+batch_size]

            seqdata,seqlabels,seqlngths,max_len=extract_seq_data(batch_data, 0, self.config.maxseqlen)
            
            mask_idx = self.cal_mask(seqdata, -1)

            feed={self.input:seqdata,
                    self.mask_idx:mask_idx,
                    self.labels:seqlabels,
                    self.dropout:1.0,
                    self.lr: self.config.lr,
                    self.lngths:seqlngths,
                    self.batch_len:len(seqdata),
                    self.max_time:max_len}
            if isDev: 
                summary, pred=sess.run([self.merged,self.pred],feed_dict=feed)
                self.dev_writer.add_summary(summary, self.config.global_step)
            else:
                pred = sess.run(self.pred, feed_dict = feed)

            y=np.argmax(pred,axis=1)
            
            for i,v in enumerate(y):
                if seqlabels[i]==v:
                    num_correct+=1
                total_data+=1
            

        acc=float(num_correct)/float(total_data)
        return acc

    
    def plot_attention(self,data, voc, sess):
        
        idx = np.random.randint(0,len(data))
        one_data = data[idx:idx+1]

        seqdata,seqlabels,seqlngths,max_len=extract_seq_data(one_data, 0, self.config.maxseqlen)
        
        mask_idx = self.cal_mask(seqdata, -1)

        feed = {self.input: seqdata,
                self.mask_idx: mask_idx,
                self.labels: seqlabels,
                self.dropout: 1.0,
                self.lngths: seqlngths,
                self.batch_len: len(seqdata),
                self.max_time: max_len}

        predlabel, alphas = sess.run([self.predlabel,self.alphas], feed_dict = feed)
        
        #context = sess.run(self.context, feed_dict = feed)
        #final_hidden = sess.run(self.final_hidden, feed_dict = feed)
        #masked_betas = sess.run(self.masked_betas, feed_dict = feed)
        #print("context:", context)
        #print("final_hidden:", final_hidden)
        #print("masked_betas:", masked_betas)

        seq_index = seqdata[0]
         
        sentence = []
        for idx in seq_index:
            if idx != -1:
                word = voc.decode(idx)
                sentence.append(word)
            else:
                break

        return alphas, sentence, seqlabels[0], predlabel


# seq_LSTM + Attention layer
class tf_seqLSTMAtt(tf_seqLSTM):

    def unpack_sequence(self,tensor):
        return tf.unstack(tf.transpose(tensor, perm = [1,0,2]))
    
    def attention(self, emb,final_hidden,mt):
         
        with tf.variable_scope("Attention", regularizer = tf.contrib.layers.l2_regularizer(self.reg)):
            Wa = tf.get_variable("Wa", [self.emb_dim + self.hidden_dim, self.attention_dim],
                    initializer = tf.random_uniform_initializer(-0.05, 0.05)) 
            ba = tf.get_variable("ba", [self.attention_dim],
                    initializer = tf.constant_initializer(0.0),
                    regularizer = tf.contrib.layers.l2_regularizer(0.0))
            Ua = tf.get_variable("Ua", [self.attention_dim],
                    initializer = tf.random_uniform_initializer(-0.05, 0.05))
        
            with tf.name_scope("Attention"):
                tf.summary.histogram("Wa", Wa)
                #tf.summary.histogram("ba", ba)
                tf.summary.histogram("Ua", Ua)

        _hs = tf.stack([final_hidden for i in range(mt)])
        hs = tf.multiply(_hs, tf.to_float(tf.expand_dims(tf.transpose(self.mask_idx), -1)))
        _x_h = tf.concat([self.unpack_sequence(emb), hs], axis = 2)
        x_h = tf.reshape(_x_h, [-1, self.emb_dim + self.hidden_dim])

        _v_att = tf.tanh(tf.matmul(x_h, Wa) + tf.reshape(ba, [1, -1]))
        _betas = tf.matmul(_v_att, tf.reshape(Ua, [-1,1]))
        betas = tf.reshape(_betas, [mt, -1])
        
        exp_betas = tf.exp(betas)
        self.masked_betas = tf.multiply(exp_betas, tf.transpose(self.mask_idx))
        _alphas = tf.div(self.masked_betas ,tf.reduce_sum(self.masked_betas, 0))
        alphas = tf.expand_dims(_alphas, -1)
        return alphas

    def score(self, context, target, mt, method = "concat"):
        #general
        #transpose(context) * Ws * target
        if method == "general":
            with tf.variable_scope("Attention_g", regularizer = tf.contrib.layers.l2_regularizer(self.reg)):
                Wa = tf.get_variable("Wa", [self.hidden_dim, self.hidden_dim],
                    initializer = tf.random_uniform_initializer(-0.05, 0.05))

                tf.summary.histogram("Wa", Wa)

                t_wa = tf.matmul(target, Wa)
                t_wa_c = tf.multiply(context, t_wa)
                betas = tf.reduce_sum(t_wa_c, axis = 2)

        #dot
        #transpose(context) * target
        elif method == "dot":
            dot = tf.multiply(context, target)
            betas = tf.reduce_sum(dot, axis = 2)
        
        #location
        #only use hiddens to self attention
        elif method == "location":
            with tf.variable_scope("Attention_l", regularizer = tf.contrib.layers.l2_regularizer(self.reg)):
                Wa = tf.get_variable("Wa", [self.hidden_dim, self.attention_dim],
                    initializer = tf.random_uniform_initializer(-0.05, 0.05))
                 
                ba = tf.get_variable("ba",[self.attention_dim],
                    initializer=tf.constant_initializer(0.0),
                    regularizer=tf.contrib.layers.l2_regularizer(0.0))
                Ua = tf.get_variable("Ua",[self.attention_dim],
                    initializer=tf.random_uniform_initializer(-0.05,0.05))

                tf.summary.histogram("Wa",Wa)
                tf.summary.histogram("Ua",Ua)

                v_att = tf.tanh(tf.matmul(tf.reshape(context,[-1, self.hidden_dim]), Wa) + tf.reshape(ba, [1,-1]))
                _betas = tf.matmul(v_att, tf.reshape(Ua, [-1,1]))
                betas = tf.reshape(_betas, [mt,-1])

        #default concat
        #transpose(Vs) * tanh(Ws[c:h])
        else:
            with tf.variable_scope("Attention_c",regularizer = tf.contrib.layers.l2_regularizer(self.reg)):
                Wa = tf.get_variable("Wa", [2 * self.hidden_dim, self.attention_dim],
                    initializer = tf.random_uniform_initializer(-0.05, 0.05)) 
                Ua = tf.get_variable("Ua", [self.attention_dim],
                    initializer = tf.random_uniform_initializer(-0.05, 0.05),
                    regularizer = tf.contrib.layers.l2_regularizer(0.0))

                tf.summary.histogram("Wa", Wa)
                tf.summary.histogram("Ua", Ua)

                _hs = tf.stack([target for i in range(mt)])
                hs = tf.multiply(_hs, tf.to_float(tf.expand_dims(tf.transpose(self.mask_idx), -1)))
                _x_h = tf.concat([context, hs], axis = 2)
                x_h = tf.reshape(_x_h, [-1, 2*self.hidden_dim])

                _v_att = tf.tanh(tf.matmul(x_h, Wa))
                _betas = tf.matmul(_v_att, tf.reshape(Ua, [-1,1]))
                betas = tf.reshape(_betas, [mt, -1])
        
    
        exp_betas = tf.exp(betas)
        self.masked_betas = tf.multiply(exp_betas, tf.transpose(self.mask_idx))
        _alphas = tf.div(self.masked_betas ,tf.reduce_sum(self.masked_betas, 0))
        alphas = tf.expand_dims(_alphas, -1)
        
        return alphas

    def self_attention(self, context, target, mt, method):
        return self.score(context, target, mt, method)


    def compute_states(self, emb):
        with tf.variable_scope("Composition", 
                initializer = tf.contrib.layers.xavier_initializer(),
                regularizer = tf.contrib.layers.l2_regularizer(self.reg)):
            
            cell = rnn.BasicLSTMCell(self.hidden_dim)
            cell = rnn.DropoutWrapper(cell,
                    output_keep_prob = self.dropout,
                    input_keep_prob = self.dropout)

            outputs, state = rnn.static_rnn(cell, self.unpack_sequence(emb), 
                    sequence_length = self.lngths, dtype = tf.float32)
            
            #output shape:list of [batch_size, output_size],list length=max_time
            #state shape:[2, batch_size, output_size], in which 2 is SLTMStateTuple,(c,h)
        
        #the final hidden of the sentence
        final_hidden = state.h

        #average of hiddens
        sum_out = tf.reduce_sum(tf.stack(outputs), [0])
        average_hidden = tf.div(sum_out, tf.expand_dims(tf.to_float(self.lngths), 1))
        
        #attenions
        mt = len(outputs)
        method = self.config.method
        alphas = self.self_attention(outputs, average_hidden, mt, method)

        context = tf.reduce_sum(outputs * alphas, 0)
        final_outputs = tf.concat([context, final_hidden], axis = 1)
        
        self.alphas = tf.squeeze(alphas)

        #self.context = context
        #self.final_hidden = final_hidden
    
        return final_outputs


    
    def create_output(self,rnn_out):
        #project the concatenation into same space 
        with tf.variable_scope("Concatenation", regularizer = tf.contrib.layers.l2_regularizer(self.reg)):
            Wc = tf.get_variable("Wc", [self.config.concat_dim, 2*self.hidden_dim],
                    initializer = tf.random_uniform_initializer(-0.05, 0.05))
            bc = tf.get_variable("bc", [self.config.concat_dim], initializer = tf.constant_initializer(0.0),
                    regularizer = tf.contrib.layers.l2_regularizer(0.0))

            concat_rnn_out = tf.matmul(rnn_out, Wc, transpose_b=True) + bc
            
            with tf.name_scope("Concatenation"):
                tf.summary.histogram("Wc", Wc)
                #tf.summary.histogram("bc", bc)

        with tf.variable_scope("Projection",regularizer = tf.contrib.layers.l2_regularizer(self.reg)):
            Wp = tf.get_variable("Wp",[self.output_dim ,self.config.concat_dim],
                                initializer = tf.random_uniform_initializer(-0.05,0.05))
            bp = tf.get_variable("bp",[self.output_dim],initializer = tf.constant_initializer(0.0),
                                 regularizer=tf.contrib.layers.l2_regularizer(0.0))

            logits=tf.matmul(concat_rnn_out, Wp, transpose_b=True) + bp
            
            with tf.name_scope("Projection"):
                tf.summary.histogram("Wp",Wp)
                #tf.summary.histogram("bp",bp)

            return logits


class tf_seqbiLSTM(tf_seqLSTM):

    def compute_states(self,emb):
        def unpack_sequence(tensor):
            return tf.unstack(tf.transpose(tensor, perm=[1, 0, 2]))


        with tf.variable_scope("Composition",initializer=tf.contrib.layers.xavier_initializer(),
                regularizer = tf.contrib.layers.l2_regularizer(self.reg)):
            cell_fw = rnn.BasicLSTMCell(self.hidden_dim)
            cell_bw = rnn.BasicLSTMCell(self.hidden_dim)
            
            cell_fw = rnn.DropoutWrapper(cell_fw,output_keep_prob=self.dropout,input_keep_prob=self.dropout)
            cell_bw = rnn.DropoutWrapper(cell_bw, output_keep_prob=self.dropout,input_keep_prob=self.dropout)

            outputs,_,_=rnn.static_bidirectional_rnn(cell_fw,cell_bw,unpack_sequence(emb),sequence_length=self.lngths,dtype=tf.float32)

        #use average of hiddens
        sum_out=tf.reduce_sum(tf.stack(outputs),[0])
        sent_rep = tf.div(sum_out,tf.expand_dims(tf.to_float(self.lngths),1))

        final_state=sent_rep
        return final_state


    def create_output(self,rnn_out):

        with tf.variable_scope("Projection",regularizer=
                               tf.contrib.layers.l2_regularizer(self.reg)):
            Wp= tf.get_variable("Wp",[self.output_dim,2*self.hidden_dim],
                                initializer=tf.random_uniform_initializer(-0.05,0.05))
            bp = tf.get_variable("bp",[self.output_dim],initializer=tf.constant_initializer(0.0),
                                 regularizer=tf.contrib.layers.l2_regularizer(0.0))

            logits=tf.matmul(rnn_out,Wp,transpose_b=True)+bp

            return logits

class tf_seqbiLSTMAtt(tf_seqLSTMAtt):


    def attention(self, emb,final_hidden,mt):
         
        with tf.variable_scope("Attention", regularizer = tf.contrib.layers.l2_regularizer(self.reg)):
            Wa = tf.get_variable("Wa", [self.emb_dim + self.hidden_dim*2, self.attention_dim],
                    initializer = tf.random_uniform_initializer(-0.05, 0.05)) 
            ba = tf.get_variable("ba", [self.attention_dim],
                    initializer = tf.constant_initializer(0.0),
                    regularizer = tf.contrib.layers.l2_regularizer(0.0))
            Ua = tf.get_variable("Ua", [self.attention_dim],
                    initializer = tf.random_uniform_initializer(-0.05, 0.05))
        
            with tf.name_scope("Attention"):
                tf.summary.histogram("Wa", Wa)
                #tf.summary.histogram("ba", ba)
                tf.summary.histogram("Ua", Ua)

        _hs = tf.stack([final_hidden for i in range(mt)])
        hs = tf.multiply(_hs, tf.to_float(tf.expand_dims(tf.transpose(self.mask_idx), -1)))
        _x_h = tf.concat([self.unpack_sequence(emb), hs], axis = 2)
        x_h = tf.reshape(_x_h, [-1, self.emb_dim + self.hidden_dim*2])

        _v_att = tf.tanh(tf.matmul(x_h, Wa) + tf.reshape(ba, [1, -1]))
        _betas = tf.matmul(_v_att, tf.reshape(Ua, [-1,1]))
        betas = tf.reshape(_betas, [mt, -1])
        
        exp_betas = tf.exp(betas)
        self.masked_betas = tf.multiply(exp_betas, tf.transpose(self.mask_idx))
        _alphas = tf.div(self.masked_betas ,tf.reduce_sum(self.masked_betas, 0))
        alphas = tf.expand_dims(_alphas, -1)
        
        return alphas
    
    def score(self, context, target, mt, method = "concat"):
        #general
        #transpose(context) * Ws * target
        if method == "general":
            with tf.variable_scope("Attention_g", regularizer = tf.contrib.layers.l2_regularizer(self.reg)):
                Wa = tf.get_variable("Wa", [2*self.hidden_dim, 2*self.hidden_dim],
                    initializer = tf.random_uniform_initializer(-0.05, 0.05))

                tf.summary.histogram("Wa", Wa)

                t_wa = tf.matmul(target, Wa)
                t_wa_c = tf.multiply(context, t_wa)
                betas = tf.reduce_sum(t_wa_c, axis = 2)

        #dot
        #transpose(context) * target
        elif method == "dot":
            dot = tf.multiply(context, target)
            betas = tf.reduce_sum(dot, axis = 2)
        
        #location
        #only use hiddens to self attention
        elif method == "location":
            with tf.variable_scope("Attention_l", regularizer = tf.contrib.layers.l2_regularizer(self.reg)):
                Wa = tf.get_variable("Wa", [2*self.hidden_dim, self.attention_dim],
                    initializer = tf.random_uniform_initializer(-0.05, 0.05))
                 
                ba = tf.get_variable("ba",[self.attention_dim],
                    initializer=tf.constant_initializer(0.0),
                    regularizer=tf.contrib.layers.l2_regularizer(0.0))
                Ua = tf.get_variable("Ua",[self.attention_dim],
                    initializer=tf.random_uniform_initializer(-0.05,0.05))

                tf.summary.histogram("Wa",Wa)
                tf.summary.histogram("Ua",Ua)

                v_att = tf.tanh(tf.matmul(tf.reshape(context,[-1, 2*self.hidden_dim]),Wa) + tf.reshape(ba, [1,-1]))
                _betas = tf.matmul(v_att, tf.reshape(Ua, [-1,1]))
                betas = tf.reshape(_betas, [mt,-1])

        #default concat
        #transpose(Vs) * tanh(Ws[c:h])
        else:
            with tf.variable_scope("Attention_c",regularizer = tf.contrib.layers.l2_regularizer(self.reg)):
                Wa = tf.get_variable("Wa", [4 * self.hidden_dim, self.attention_dim],
                    initializer = tf.random_uniform_initializer(-0.05, 0.05)) 
                Ua = tf.get_variable("Ua", [self.attention_dim],
                    initializer = tf.random_uniform_initializer(-0.05, 0.05),
                    regularizer = tf.contrib.layers.l2_regularizer(0.0))

                tf.summary.histogram("Wa", Wa)
                tf.summary.histogram("Ua", Ua)

                _hs = tf.stack([target for i in range(mt)])
                hs = tf.multiply(_hs, tf.to_float(tf.expand_dims(tf.transpose(self.mask_idx), -1)))
                _x_h = tf.concat([context, hs], axis = 2)
                x_h = tf.reshape(_x_h, [-1, 4*self.hidden_dim])

                _v_att = tf.tanh(tf.matmul(x_h, Wa))
                _betas = tf.matmul(_v_att, tf.reshape(Ua, [-1,1]))
                betas = tf.reshape(_betas, [mt, -1])
        
    
        exp_betas = tf.exp(betas)
        self.masked_betas = tf.multiply(exp_betas, tf.transpose(self.mask_idx))
        _alphas = tf.div(self.masked_betas ,tf.reduce_sum(self.masked_betas, 0))
        alphas = tf.expand_dims(_alphas, -1)
        
        return alphas
     
    def compute_states(self, emb):

        with tf.variable_scope("Composition", 
                initializer = tf.contrib.layers.xavier_initializer(),
                regularizer = tf.contrib.layers.l2_regularizer(self.reg)):
            
            cell_fw = rnn.BasicLSTMCell(self.hidden_dim)
            cell_bw = rnn.BasicLSTMCell(self.hidden_dim)

            cell_fw = rnn.DropoutWrapper(cell_fw, output_keep_prob = self.dropout, input_keep_prob = self.dropout)
            cell_bw = rnn.DropoutWrapper(cell_bw, output_keep_prob = self.dropout, input_keep_prob = self.dropout)

            outputs, output_state_fw, output_state_bw = rnn.static_bidirectional_rnn(cell_fw, cell_bw, self.unpack_sequence(emb), sequence_length = self.lngths, dtype = tf.float32)

        #final hidden
        final_hidden = outputs[-1]

        #use average of hiddens
        sum_out=tf.reduce_sum(tf.stack(outputs),[0])
        average_hidden = tf.div(sum_out,tf.expand_dims(tf.to_float(self.lngths),1))
        
        mt = len(outputs)
        method = self.config.method
        alphas = self.self_attention(outputs, average_hidden, mt,method)

        context = tf.reduce_sum(outputs * alphas, 0)
        final_outputs = tf.concat([context, average_hidden], axis = 1)

        self.alphas = tf.squeeze(alphas)
        
        return final_outputs

    
    def create_output(self,rnn_out):
        #project the concatenation into same space 
        with tf.variable_scope("Concatenation", regularizer = tf.contrib.layers.l2_regularizer(self.reg)):
            Wc = tf.get_variable("Wc", [self.config.concat_dim, 4*self.hidden_dim],
                    initializer = tf.random_uniform_initializer(-0.05, 0.05))
            bc = tf.get_variable("bc", [self.config.concat_dim], initializer = tf.constant_initializer(0.0),
                    regularizer = tf.contrib.layers.l2_regularizer(0.0))

            concat_rnn_out = tf.matmul(rnn_out, Wc, transpose_b=True) + bc
            
            with tf.name_scope("Concatenation"):
                tf.summary.histogram("Wc", Wc)
                #tf.summary.histogram("bc", bc)

        with tf.variable_scope("Projection",regularizer = tf.contrib.layers.l2_regularizer(self.reg)):
            Wp = tf.get_variable("Wp",[self.output_dim ,self.config.concat_dim],
                                initializer = tf.random_uniform_initializer(-0.05,0.05))
            bp = tf.get_variable("bp",[self.output_dim],initializer = tf.constant_initializer(0.0),
                                 regularizer=tf.contrib.layers.l2_regularizer(0.0))

            logits=tf.matmul(concat_rnn_out, Wp, transpose_b=True) + bp
            
            with tf.name_scope("Projection"):
                tf.summary.histogram("Wp",Wp)
                #tf.summary.histogram("bp",bp)

            return logits

