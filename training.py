#!/usr/bin/env python
# coding: utf-8

# In[34]:


import importlib
import Preprocessing
importlib.reload(Preprocessing)

DEVICE= torch.device('cpu')
trainloader = Preprocessing.TrainLoader(DEVICE).trainloader


# In[35]:


import Encoder
import Decoder


# In[38]:


import importlib
importlib.reload(Decoder)
importlib.reload(Encoder)
import os
import pickle
import time
import numpy as np

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision import transforms

# from utils.data_loader import get_loader, Vocabulary
# from utils.model import *
# from utils.logger import Logger
EncoderCNN = Encoder.EncoderCNN
SentenceRNN = Decoder.SentenceRNN
WordRNN = Decoder.WordRNN

EPOCHS = 2
LEARNING_RATE = 0.1
BATCH_SIZE=2

class Im2pGenerator(object):
    def __init__(self):
        self.min_loss = 100000
        self.train_data_loader = trainloader
        print('in constructor')
#         self.val_data_loader = self.__init_data_loader(self.args.val_images_json)

        self.encoderCNN = EncoderCNN()
        self.sentenceRNN = SentenceRNN()
        self.wordRNN = WordRNN(hidden_size=256, vocab_size=2000, att_dim=256, embed_size=256, encoded_dim=256, device=DEVICE)
        self.criterion = nn.BCELoss(size_average=False, reduce=False)
        self.optimizer = torch.optim.Adam(params=(
            list(self.encoderCNN.parameters()) + list(self.sentenceRNN.parameters()) + list(self.wordRNN.parameters())
        ), lr=LEARNING_RATE)

#         self.scheduler = self.__init_scheduler()

#         self.logger = self.__init_logger()

    def train(self):
        for epoch in range(EPOCHS):
            train_loss = self.__epoch_train()
            # val_loss = self.__epoch_val()
            val_loss = 0
#             self.scheduler.step(train_loss)
            print("[{}] Epoch-{} - train loss:{} - val loss:{} - lr:{}".format(self.__get_now(),
                                                                               epoch + 1,
                                                                               train_loss,
                                                                               val_loss,
#                                                                                self.optimizer.param_groups[0]['lr']
                                                                               0
                                                                              ))
            # self.__save_model(train_loss, self.args.saved_model_name)
            # self.__log(train_loss, val_loss, epoch + 1)

    def __epoch_train(self):
        print('in epoch train')
        train_loss = 0
        self.encoderCNN.train()
        self.wordRNN.train()
        self.sentenceRNN.train()
        
        for i, (images, findings, sentenceVectors, word2d, wordsLengths) in enumerate(self.train_data_loader):
            print('in for loop', i)
            """"""
            
            featureMap, globalFeatures = self.encoderCNN.forward(images)
            sentence_states = None

            sentence_loss = 0
            word_loss = 0
            word2d = word2d.permute(1, 0, 2) #(sentenceIndex, batchSize, maxWordsInSentence)
            for sentenceIndex, sentence_value in enumerate(sentenceVectors):
                print('-----SI-----', sentenceIndex)
                endToken, topic_vec, sentence_states = self.sentenceRNN.forward(globalFeatures, sentence_states)
                endToken = endToken.squeeze(1).squeeze(1)
                """***TODO*** Should stop calculating loss for sentences once they're done."""
                loss = self.criterion(endToken, sentence_value.type(torch.float)).sum()
                sentence_loss += loss
                
#                 print('words2d', word2d.size())
#                 print('wordsLengths', len(wordsLengths), wordsLengths)
                captions=word2d[sentenceIndex]
                captionLengths=wordsLengths[sentenceIndex]
#                 print('captions', captions.size())
                if(any(captionLengths)):
                    predictions, alphas, betas, encoded_captions, decode_lengths, sort_ind = self.wordRNN.forward(
                        enc_image=featureMap,
                        global_features=globalFeatures,
                        encoded_captions=captions,
                        caption_lengths=captionLengths
                    )
                    
            break
#                     for word_index in range(1, word2d.shape[2] - 1):
#                         words_pred = self.wordRNN.forward(topic_vec=topic_vec,
#                                                           captions=captions[:, sentence_index, :word_index])
#                         caption_mask = (captions[:, sentence_index, word_index] > 1).view(-1,).float().cuda()
#                         t_loss = self.criterion(words_pred, self.__to_var(captions[:, sentence_index, word_index]))
#                         t_loss = t_loss * caption_mask
#                         word_loss += t_loss.sum()

#             loss = self.args.lambda_word * word_loss
#             loss.backward()
#             self.optimizer.step()
#             train_loss += loss.data[0]

        return train_loss


    def __get_date(self):
        return str(time.strftime('%Y%m%d', time.gmtime()))

    def __get_now(self):
        return str(time.strftime('%y%m%d-%H:%M:%S', time.gmtime()))

im2p = Im2pGenerator()
im2p.train()

