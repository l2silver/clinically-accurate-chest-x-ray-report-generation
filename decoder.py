#!/usr/bin/env python
# coding: utf-8

# In[2]:


import torch.nn as nn
import torch

class SentenceRNN(nn.Module):
    def __init__(self,
                 hidden_size=256,
                 lstm_layer=1,
                 pooling_dim=256,
                 topic_dim=256):
        super(SentenceRNN, self).__init__()
        self.hidden_size = hidden_size
        self.lstm_layer = lstm_layer
        self.pooling_dim = pooling_dim
        self.topic_dim = topic_dim

        self.lstm = nn.LSTM(self.topic_dim, hidden_size=self.hidden_size, num_layers=self.lstm_layer, batch_first=True)
        self.endTokenLinear = nn.Linear(self.hidden_size, 1)
        self.endTokenActivation = nn.Sigmoid()
        self.topicVectorLinear = nn.Linear(self.hidden_size, self.topic_dim)
        self.topicVectorActivation = nn.ReLU()
        self.__init_weights()

    def __init_weights(self):
        self.endTokenLinear.weight.data.uniform_(-0.1, 0.1)
        self.endTokenLinear.bias.data.fill_(0)
        self.topicVectorLinear.weight.data.uniform_(-0.1, 0.1)
        self.topicVectorLinear.bias.data.fill_(0)
    
    def forward(self, pooling_vector, states=None):
        pooling_vector = pooling_vector.unsqueeze(1)

        hiddens, states = self.lstm(pooling_vector, states)
        endToken = self.endTokenActivation(self.endTokenLinear(hiddens))
#         endToken = endToken >= 0.5
        endToken = endToken.type(torch.float)
        topicVector = self.topicVectorActivation(self.topicVectorLinear(hiddens))
        return endToken, topicVector, states

#     def sample(self, pooling_vector, states=None):
#         pooling_vector = pooling_vector.unsqueeze(1)

#         hiddens, states = self.lstm(pooling_vector, states)
#         p = self.activation(self.logistic(hiddens))
#         topic = self.fc2(self.fc1(hiddens))
#         return p, topic, states


# In[3]:


import torch.nn.functional as F

# https://github.com/fawazsammani/knowing-when-to-look-adaptive-attention/blob/master/models.py
class AdaptiveLSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(AdaptiveLSTMCell, self).__init__()
        self.lstm_cell = nn.LSTMCell(input_size, hidden_size)
        self.x_gate = nn.Linear(input_size, hidden_size)
        self.h_gate = nn.Linear(hidden_size, hidden_size)
        
    def forward(self, inp, states):
        h_old, c_old = states
        ht, ct = self.lstm_cell(inp, (h_old, c_old))
        sen_gate = F.sigmoid(self.x_gate(inp) + self.h_gate(h_old))
        st =  sen_gate * F.tanh(ct)
        return ht, ct, st

# https://github.com/fawazsammani/knowing-when-to-look-adaptive-attention/blob/master/models.py
class AdaptiveAttention(nn.Module):
    def __init__(self, hidden_size, att_dim):
        super(AdaptiveAttention,self).__init__()
        self.sen_affine = nn.Linear(hidden_size, hidden_size)  
        self.sen_att = nn.Linear(hidden_size, att_dim)
        self.h_affine = nn.Linear(hidden_size, hidden_size)   
        self.h_att = nn.Linear(hidden_size, att_dim)
        self.v_att = nn.Linear(hidden_size, att_dim)
        self.alphas = nn.Linear(att_dim, 1)
        self.context_hidden = nn.Linear(hidden_size, hidden_size)

    def forward(self, spatial_image, decoder_out, st):
        """
        spatial_image: the spatial image of size (batch_size,num_pixels,hidden_size)
        decoder_out: the decoder hidden state of shape (batch_size, hidden_size)
        st: visual sentinal returned by the Sentinal class, of shape: (batch_size, hidden_size)
        """
        num_pixels = spatial_image.shape[1]
        visual_attn = self.v_att(spatial_image)           # (batch_size,num_pixels,att_dim)
        sentinel_affine = F.relu(self.sen_affine(st))     # (batch_size,hidden_size)
        sentinel_attn = self.sen_att(sentinel_affine)     # (batch_size,att_dim)

        hidden_affine = F.tanh(self.h_affine(decoder_out))    # (batch_size,hidden_size)
        hidden_attn = self.h_att(hidden_affine)               # (batch_size,att_dim)

        hidden_resized = hidden_attn.unsqueeze(1).expand(hidden_attn.size(0), num_pixels + 1, hidden_attn.size(1))

        concat_features = torch.cat([spatial_image, sentinel_affine.unsqueeze(1)], dim = 1)   # (batch_size, num_pixels+1, hidden_size)
        attended_features = torch.cat([visual_attn, sentinel_attn.unsqueeze(1)], dim = 1)     # (batch_size, num_pixels+1, att_dim)

        attention = F.tanh(attended_features + hidden_resized)    # (batch_size, num_pixels+1, att_dim)
        
        alpha = self.alphas(attention).squeeze(2)                   # (batch_size, num_pixels+1)
        att_weights = F.softmax(alpha, dim=1)                              # (batch_size, num_pixels+1)

        context = (concat_features * att_weights.unsqueeze(2)).sum(dim=1)       # (batch_size, hidden_size)     
        beta_value = att_weights[:,-1].unsqueeze(1)                       # (batch_size, 1)

        out_l = F.tanh(self.context_hidden(context + hidden_affine))

        return out_l, att_weights, beta_value


# In[4]:


# https://github.com/fawazsammani/knowing-when-to-look-adaptive-attention/blob/master/models.py
class WordRNN(nn.Module):
    def __init__(self, hidden_size, vocab_size, att_dim, embed_size, encoded_dim, device):
        super(WordRNN,self).__init__()
        self.hidden_size = hidden_size
        self.fc = nn.Linear(hidden_size, vocab_size)
        self.encoded_to_hidden = nn.Linear(encoded_dim, hidden_size)
        self.global_features = nn.Linear(encoded_dim, embed_size)
        self.LSTM = AdaptiveLSTMCell(embed_size * 2,hidden_size)
        self.adaptive_attention = AdaptiveAttention(hidden_size, att_dim)
        # input to the LSTMCell should be of shape (batch, input_size). Remember we are concatenating the word with
        # the global image features, therefore out input features should be embed_size * 2
        self.embedding = nn.Embedding(vocab_size, embed_size)  
        self.vocab_size = vocab_size
        self.dropout = nn.Dropout(p=0.5)
        self.init_weights()
        self.device=device
        
    def init_weights(self):
        self.fc.weight.data.uniform_(-0.1, 0.1)
        self.fc.bias.data.fill_(0)
        self.embedding.weight.data.uniform_(-0.1, 0.1)
    
    def init_hidden_state(self, enc_image):
        h = torch.zeros(enc_image.shape[0], self.hidden_size).to(self.device)
        c = torch.zeros(enc_image.shape[0], self.hidden_size).to(self.device)
        return h, c
    
    def forward(self, enc_image, global_features, encoded_captions, caption_lengths):
        
        """
        enc_image: the encoded images from the encoder, of shape (batch_size, num_pixels, 2048)
        global_features: the global image features returned by the Encoder, of shape: (batch_size, 2048)
        encoded_captions: encoded captions, a tensor of dimension (batch_size, max_caption_length)
        caption_lengths: caption lengths, a tensor of dimension (batch_size, 1)
        """

        spatial_image = F.relu(self.encoded_to_hidden(enc_image))  # (batch_size,num_pixels,hidden_size)
        global_image = F.relu(self.global_features(global_features))      # (batch_size,embed_size)
        batch_size = spatial_image.shape[0]
        num_pixels = spatial_image.shape[1]
        
        # Sort input data by decreasing lengths
        # caption_lenghts will contain the sorted lengths, and sort_ind contains the sorted elements indices 
        caption_lengths, sort_ind = caption_lengths.sort(dim=0, descending=True)
        #The sort_ind contains elements of the batch index of the tensor encoder_out. For example, if sort_ind is [3,2,0],
        #then that means the descending order starts with batch number 3,then batch number 2, and finally batch number 0. 
        spatial_image = spatial_image[sort_ind]           # (batch_size,num_pixels,hidden_size) with sorted batches
        global_image = global_image[sort_ind]             # (batch_size, embed_size) with sorted batches
        encoded_captions = encoded_captions[sort_ind]     # (batch_size, max_caption_length) with sorted batches 
        enc_image = enc_image[sort_ind]                   # (batch_size, num_pixels, 2048)

        # Embedding. Each batch contains a caption. All batches have the same number of rows (words), since we previously
        # padded the ones shorter than max_caption_length, as well as the same number of columns (embed_dim)
        embeddings = self.embedding(encoded_captions)     # (batch_size, max_caption_length, embed_dim)

        # Initialize the LSTM state
        h,c = self.init_hidden_state(enc_image)          # (batch_size, hidden_size)

#         # We won't decode at the <end> position, since we've finished generating as soon as we generate <end>
        decode_lengths = (caption_lengths).tolist()
            
        # Create tensors to hold word predicion scores,alphas and betas
        predictions = torch.zeros(batch_size, max(decode_lengths), self.vocab_size).to(self.device)
        alphas = torch.zeros(batch_size, max(decode_lengths), num_pixels+1).to(self.device)
        betas = torch.zeros(batch_size, max(decode_lengths),1).to(self.device) 
        
        # Concatenate the embeddings and global image features for input to LSTM 
        global_image = global_image.unsqueeze(1).expand_as(embeddings)
        inputs = torch.cat((embeddings,global_image), dim = 2)    # (batch_size, max_caption_length, embed_dim * 2)

        #Start decoding
        for timestep in range(max(decode_lengths)):
            # Create a Packed Padded Sequence manually, to process only the effective batch size N_t at that timestep. Note
            # that we cannot use the pack_padded_seq provided by torch.util because we are using an LSTMCell, and not an LSTM
            batch_size_t = sum([l > timestep for l in decode_lengths])
            current_input = inputs[:batch_size_t, timestep, :]             # (batch_size_t, embed_dim * 2)
            h, c, st = self.LSTM(current_input, (h[:batch_size_t], c[:batch_size_t]))  # (batch_size_t, hidden_size)
            # Run the adaptive attention model
            out_l, alpha_t, beta_t = self.adaptive_attention(spatial_image[:batch_size_t],h,st)
            # Compute the probability over the vocabulary
            pt = self.fc(self.dropout(out_l))                  # (batch_size, vocab_size)
            predictions[:batch_size_t, timestep, :] = pt
            alphas[:batch_size_t, timestep, :] = alpha_t
            betas[:batch_size_t, timestep, :] = beta_t
        return predictions, alphas, betas, encoded_captions, decode_lengths, sort_ind 

