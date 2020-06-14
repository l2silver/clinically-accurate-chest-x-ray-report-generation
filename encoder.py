#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
from torchvision import models
import torch.nn as nn


model_urls = {
    'densenet121': 'https://download.pytorch.org/models/densenet121-a639ec97.pth',
    'densenet169': 'https://download.pytorch.org/models/densenet169-b2777c0a.pth',
    'densenet201': 'https://download.pytorch.org/models/densenet201-c1103571.pth',
    'densenet161': 'https://download.pytorch.org/models/densenet161-8d451a50.pth',
}

def _densenet(arch, growth_rate, block_config, num_init_features, pretrained, progress,
              **kwargs):
    model = AdaptedDenseNet(growth_rate, block_config, num_init_features, **kwargs)
    if pretrained:
        models.densenet._load_state_dict(model, model_urls[arch], progress)
    return model

def adaptedDensenet121(pretrained=False, progress=True, **kwargs):
    r"""Densenet-121 model from
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
        memory_efficient (bool) - If True, uses checkpointing. Much more memory efficient,
          but slower. Default: *False*. See `"paper" <https://arxiv.org/pdf/1707.06990.pdf>`_
    """
    return _densenet('densenet121', 32, (6, 12, 24, 16), 64, pretrained, progress,
                     **kwargs)

    
class AdaptedDenseNet(models.DenseNet):
    def __init__(self, growth_rate, block_config, num_init_features, **kwargs):
        super().__init__(growth_rate, block_config, num_init_features, **kwargs)
    
    def forward(self, x):
        return self.features(x)
#         out = F.relu(features, inplace=True)
#         out = F.adaptive_avg_pool2d(out, (1, 1))
#         out = torch.flatten(out, 1)
#         out = self.classifier(out)


# https://github.com/fawazsammani/knowing-when-to-look-adaptive-attention/blob/master/models.py#L32
# Use sentinel paper to create correct results
class EncoderCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.densenet = adaptedDensenet121(pretrained=True)
        self.dropout = nn.Dropout(p=0.5)
        self.avgpool = nn.AvgPool2d(7)
        self.featuresProjection = nn.Linear(1024, 256)
        self.globalProjection = nn.Linear(1024, 256)
    
    def forward(self, x):
        features = self.densenet(x)
        out = self.dropout(features)
        batch_size = out.shape[0]
        features = out.shape[1]
        num_pixels = out.shape[2] * out.shape[3]
        # Get the global features of the image
        global_features = self.avgpool(out).view(batch_size, -1)   # (batch_size, 1024)
        featureMap = out.permute(0, 2, 3, 1)  #  (batch_size,7,7,1024)
        featureMap = featureMap.view(batch_size,num_pixels,features)          # (batch_size,num_pixels,1024)
        featureMap = self.featuresProjection(featureMap) # (batch_size,num_pixels,256)
        global_features = self.globalProjection(global_features) # (batch_size,256)
        return featureMap, global_features


# In[1]:


# import Preprocessing
# trainloader = Preprocessing.TrainLoader().trainloader


# In[2]:


# encoder = EncoderCNN()
# encoder.train()
# for  i, (images, findings, sentenceVectors, word2d, wordLengths) in enumerate(trainloader):
#     if(i == 0):
#         featureMap, globalFeatures = encoder.forward(images)
#         print('globalFeaturesShape', globalFeatures.shape)
#         print('featureMapShape', featureMap.shape)
#         break
        
# # # TEST OUTPUT
# # input_image = Image.open("./chest.png")
# # input_tensor = preprocess(input_image)
# # input_batch = input_tensor.unsqueeze(0) # create a mini-batch as expected by the model

# # # move the input and model to GPU for speed if available
# # # if torch.cuda.is_available():
# # #     input_batch = input_batch.to('cuda')
# # #     model.to('cuda')
# # densenet.eval()
# # with torch.no_grad():
# #     output = densenet(input_batch)
# #     print(output.size())


# In[ ]:




