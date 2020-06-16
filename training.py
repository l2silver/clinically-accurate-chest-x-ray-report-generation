#!/usr/bin/env python
# coding: utf-8

# In[27]:


import importlib
import preprocessing
import torch
importlib.reload(preprocessing)

DEVICE= torch.device('cpu')
trainloader = preprocessing.TrainLoader(DEVICE).trainloader
dic = preprocessing.dic


# In[28]:


import encoder
import decoder


# In[42]:


import importlib
importlib.reload(decoder)
importlib.reload(encoder)
import os
import pickle
import time
import numpy as np


import torch.nn as nn
from torch.autograd import Variable
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision import transforms
from torch.nn.utils.rnn import pack_padded_sequence

# from utils.data_loader import get_loader, Vocabulary
# from utils.model import *
# from utils.logger import Logger
EncoderCNN = encoder.EncoderCNN
SentenceRNN = decoder.SentenceRNN
WordRNN = decoder.WordRNN

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
        self.wordRNN = WordRNN(hidden_size=256, vocab_size=len(dic), att_dim=256, embed_size=256, encoded_dim=256, device=DEVICE)
        self.criterionSentence = nn.BCELoss(size_average=False, reduce=False)
        self.criterionWord = nn.CrossEntropyLoss().to(DEVICE)
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
            featureMap, globalFeatures = self.encoderCNN.forward(images)
            sentence_states = None

            word_loss = 0
            word2d = word2d.permute(1, 0, 2) #(sentenceIndex, batchSize, maxWordsInSentence)
            for sentenceIndex, sentence_value in enumerate(sentenceVectors):
                endToken, topic_vec, sentence_states = self.sentenceRNN.forward(globalFeatures, sentence_states)
                endToken = endToken.squeeze(1).squeeze(1)
                """***TODO*** Should stop calculating loss for sentences once they're done."""
                sentenceLoss = self.criterionSentence(endToken, sentence_value.type(torch.float)).sum()
                
                captions=word2d[sentenceIndex]
                captionLengths=wordsLengths[sentenceIndex]
                if(any(captionLengths)):
                    predictions, alphas, betas, encoded_captions, decode_lengths, sort_ind = self.wordRNN.forward(
                        enc_image=featureMap,
                        global_features=globalFeatures,
                        encoded_captions=captions,
                        caption_lengths=captionLengths
                    )
                    # predictions: (batch_size, largest_sentence_in_batch_size, vocab_size)
                    targets = captions

                    # Remove timesteps that we didn't decode at, or are pads
                    # pack_padded_sequence is an easy trick to do this

                    greaterThan0LengthIndeces = list() #remove length 0 sentences
                    greaterThan0Lengths = list()
                    for i, length in enumerate(decode_lengths):
                        if(length > 0):
                            greaterThan0LengthIndeces.append(i)
                            greaterThan0Lengths.append(length)
                    targets = targets[greaterThan0LengthIndeces]
                    predictions = predictions[greaterThan0LengthIndeces]

                    targets = pack_padded_sequence(targets, greaterThan0Lengths, batch_first=True).data
                    scores = pack_padded_sequence(predictions, greaterThan0Lengths, batch_first=True).data

                    # Calculate loss
                    wordLoss = self.criterionWord(scores, targets)
                    loss = sentenceLoss + wordLoss
                    
                else:
                    loss = sentenceLoss
                
                self.optimizer.zero_grad()

                # Update weights
                loss.backward()
                self.optimizer.step()

                    
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


# In[25]:


loss = nn.CrossEntropyLoss()
input = torch.randn(3, 5, requires_grad=True)
target = torch.empty(3, dtype=torch.long).random_(5)
print(input.size())
print(target.size())
# output = loss(input, target)
# output.backward()


# In[ ]:




