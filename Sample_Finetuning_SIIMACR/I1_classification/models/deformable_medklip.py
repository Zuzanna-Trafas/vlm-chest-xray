# modified from https://github.com/tensorflow/models/blob/master/research/slim/nets/s3dg.py
from sklearn.metrics import log_loss
import torch.nn as nn
import torch
import math
import numpy as np  
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F
from .transformer import *
from .deformable_transformer import *
from .util.misc import NestedTensor
from .position_encoding import PositionEmbeddingSine
from .backbone import build_backbone
import torchvision.models as models
from einops import rearrange
from transformers import AutoModel
from timm.models import create_model
'''
args.N
args.d_model
args.res_base_model
args.H 
args.num_queries
args.dropout
args.attribute_set_size
'''




class Deformable_MedKLIP(nn.Module):

    def __init__(self, config, ana_book, disease_book, mode='train'):
        super(Deformable_MedKLIP, self).__init__()

        self.mode = mode
        self.config = config
        self.d_model = config['d_model']
        # ''' book embedding'''
        with torch.no_grad():
            bert_model = self._get_bert_basemodel(config['text_encoder'],freeze_layers = None).to(ana_book['input_ids'].device)
            self.ana_book = bert_model(input_ids = ana_book['input_ids'],attention_mask = ana_book['attention_mask'])#(**encoded_inputs)
            self.ana_book = self.ana_book.last_hidden_state[:,0,:]
            self.disease_book = bert_model(input_ids = disease_book['input_ids'],attention_mask = disease_book['attention_mask'])#(**encoded_inputs)
            self.disease_book = self.disease_book.last_hidden_state[:,0,:]
        self.disease_embedding_layer = nn.Linear(768,config['d_model'])
        self.cl_fc = nn.Linear(config['d_model'],768)
        
        self.disease_name = [
            'normal', 'clear', 'sharp', 'sharply', 'unremarkable', 'intact', 'stable', 'free',
            'effusion', 'opacity', 'pneumothorax', 'edema', 'atelectasis', 'tube', 'consolidation', 'process', 'abnormality', 'enlarge', 'tip', 'low',
            'pneumonia', 'line', 'congestion', 'catheter', 'cardiomegaly', 'fracture', 'air', 'tortuous', 'lead', 'disease', 'calcification', 'prominence',
            'device', 'engorgement', 'picc', 'clip', 'elevation', 'expand', 'nodule', 'wire', 'fluid', 'degenerative', 'pacemaker', 'thicken', 'marking', 'scar',
            'hyperinflate', 'blunt', 'loss', 'widen', 'collapse', 'density', 'emphysema', 'aerate', 'mass', 'crowd', 'infiltrate', 'obscure', 'deformity', 'hernia',
            'drainage', 'distention', 'shift', 'stent', 'pressure', 'lesion', 'finding', 'borderline', 'hardware', 'dilation', 'chf', 'redistribution', 'aspiration',
            'tail_abnorm_obs', 'excluded_obs'
        ]
        
        self.excluded_disease = [
            'pneumonia',
            'infiltrate',
            'mass',
            'nodule',
            'emphysema',
            'fibrosis',
            'thicken',
            'hernia'
        ]
        
        self.keep_class_dim = [self.disease_name.index(i) for i in self.disease_name if i not in self.excluded_disease ]
        ''' visual backbone'''
        self.backbone = build_backbone(config)
        self.input_proj = nn.ModuleList([
                nn.Sequential(
                    nn.Conv2d(self.backbone.num_channels[0], config['d_model'], kernel_size=1),
                    nn.GroupNorm(32, config['d_model']),
                )])

        ###################################
        ''' Query Decoder'''
        ###################################

        self.transformer = DeformableTransformer()            

        # Learnable Queries
        #self.query_embed = nn.Embedding(config['num_queries'] ,self.d_model)
        self.dropout_feas = nn.Dropout(config['dropout'] )

        # Attribute classifier
        # self.cls_classifier = nn.Linear(self.d_model,config['num_classes'])

        # # Class classifier
        self.cls_classifier = nn.Linear(self.d_model,config['num_classes'])

        self.apply(self._init_weights)

    def _get_bert_basemodel(self, bert_model_name, freeze_layers):
        try:
            model = AutoModel.from_pretrained(bert_model_name)#, return_dict=True)
            print("text feature extractor:", bert_model_name)
        except:
            raise ("Invalid model name. Check the config file and pass a BERT model from transformers lybrary")

        if freeze_layers is not None:
            for layer_idx in freeze_layers:
                for param in list(model.encoder.layer[layer_idx].parameters()):
                    param.requires_grad = False
        return model
    
    
    def forward(self, images,smaple_index = None, is_train = True, no_cl= False, exclude_class= False):

        # labels batch,51,75 binary_label batch,75 sample_index batch,index
        B = images.shape[0]
        device = images.device
        ''' Visual Backbone '''
        samples = NestedTensor(images, torch.ones(B, images.size(2), images.size(3))).to(device)
        features, pos = self.backbone(samples)

        srcs = []
        masks = []
        for l, feat in enumerate(features):
            src, mask = feat.decompose()
            srcs.append(self.input_proj[l](src))
            masks.append(mask)
            assert mask is not None

        # features = x.transpose(0,1) #patch_num b dim
        # print(features.shape)
        #query_embed = self.query_embed.weight.unsqueeze(1).repeat(1, B, 1) # query_number, batch, dim
        query_embed = self.disease_embedding_layer(self.disease_book)
        query_embed = query_embed.unsqueeze(1).repeat(1, B, 1)

        features, _, _, _, _ = self.transformer(srcs,masks,pos,query_embed)

        features = features.permute(1, 0, 2)
        
        out = self.dropout_feas(features)   
        x= self.cls_classifier(out).transpose(0,1) #B query Atributes

        x = x.mean(dim=1) 
        
        return x



    @staticmethod
    def _init_weights(module):
        r"""Initialize weights like BERT - N(0.0, 0.02), bias = 0."""

        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)

        elif isinstance(module, nn.MultiheadAttention):
            module.in_proj_weight.data.normal_(mean=0.0, std=0.02)
            module.out_proj.weight.data.normal_(mean=0.0, std=0.02)

        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()