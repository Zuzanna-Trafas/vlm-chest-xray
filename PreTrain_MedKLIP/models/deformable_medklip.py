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

    def __init__(self, config, ana_book, disease_book, mode='train', num_feature_levels=4):
        super(Deformable_MedKLIP, self).__init__()

        self.mode = mode
        self.config = config
        self.d_model = config['d_model']
        self.num_feature_levels = num_feature_levels
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
        num_backbone_outs = len(self.backbone.strides)
        input_proj_list = []
        for _ in range(num_backbone_outs):
            in_channels = self.backbone.num_channels[_]
            input_proj_list.append(nn.Sequential(
                nn.Conv2d(in_channels, config['d_model'], kernel_size=1),
                nn.GroupNorm(32, config['d_model']),
            ))
        for _ in range(num_feature_levels - num_backbone_outs):
            input_proj_list.append(nn.Sequential(
                nn.Conv2d(in_channels, config['d_model'], kernel_size=3, stride=2, padding=1),
                nn.GroupNorm(32, config['d_model']),
            ))
            in_channels = config['d_model']
        self.input_proj = nn.ModuleList(input_proj_list)
        ###################################
        ''' Query Decoder'''
        ###################################

        self.transformer = DeformableTransformer()            

        # Learnable Queries
        #self.query_embed = nn.Embedding(config['num_queries'] ,self.d_model)
        self.dropout_feas = nn.Dropout(config['dropout'] )

        # Attribute classifier
        self.classifier = nn.Linear(self.d_model,config['attribute_set_size'])

        # # Class classifier
        # self.cls_classifier = nn.Linear(self.d_model,args.num_classes)

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
    
    
    def forward(self, images,labels,smaple_index = None, is_train = True, no_cl= False, exclude_class= False):

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
        if self.num_feature_levels > len(srcs):
            _len_srcs = len(srcs)
            for l in range(_len_srcs, self.num_feature_levels):
                if l == _len_srcs:
                    src = self.input_proj[l](features[-1].tensors)
                else:
                    src = self.input_proj[l](srcs[-1])
                m = samples.mask
                mask = F.interpolate(m[None].float(), size=src.shape[-2:]).to(torch.bool)[0]
                pos_l = self.backbone[1](NestedTensor(src, mask)).to(src.dtype)
                srcs.append(src)
                masks.append(mask)
                pos.append(pos_l)

        # features = x.transpose(0,1) #patch_num b dim
        # print(features.shape)
        #query_embed = self.query_embed.weight.unsqueeze(1).repeat(1, B, 1) # query_number, batch, dim
        query_embed = self.disease_embedding_layer(self.disease_book)
        query_embed = query_embed.unsqueeze(1).repeat(1, B, 1)

        features, _, _, _, _ = self.transformer(srcs,masks,pos,query_embed)

        features = features.permute(1, 0, 2)
        

        out = self.dropout_feas(features)
        if is_train == True and no_cl == False:
            anatomy_query = self.ana_book[smaple_index,:] # batch, Q , position_num ,dim
            # [Q,B,A]
            ll = out.transpose(0,1) # B Q A
            Q = ll.shape[1]
            ll = ll.reshape(ll.shape[0]*ll.shape[1],-1)
            ll = self.cl_fc(ll)
            ll = ll.unsqueeze(dim =-1)
            #ll = ll.reshape(B,Q,-1)
            anatomy_query = anatomy_query.reshape(B*Q,8,768)
            ll = torch.bmm(anatomy_query, ll ).squeeze()  # B Q position_num
            cl_labels = torch.zeros((ll.shape[0])).to(device)
            if exclude_class == True:
                cl_labels = cl_labels.reshape(B,Q)
                cl_labels = cl_labels[:,self.keep_class_dim]
                cl_labels = cl_labels.reshape(-1)
                ll = ll.reshape(B,Q,-1)
                ll = ll[:,self.keep_class_dim,:]
                ll = ll.reshape(B*(len(self.keep_class_dim)),-1)
        
        
        x= self.classifier(out).transpose(0,1) #B query Atributes
         
        if exclude_class == True:
            labels = labels[:,self.keep_class_dim]
            x = x[:,self.keep_class_dim,:]
        
        
        labels = labels.reshape(-1,1)
        logits = x.reshape(-1, x.shape[-1])
        Mask = ((labels != -1) & (labels != 2)).squeeze()
        
        cl_mask = (labels == 1).squeeze()
        if is_train == True:
            labels = labels[Mask].long()
            logits = logits[Mask]
            loss_ce = F.cross_entropy(logits,labels[:,0])
            if no_cl == False:
                cl_labels = cl_labels[cl_mask].long()
                ll = ll[cl_mask]
                loss_cl = F.cross_entropy(ll,cl_labels)
                loss = loss_ce +loss_cl
            else:
                loss_cl = torch.tensor(0)
                loss = loss_ce
        else:
            loss = 0
        if is_train==True:
            return loss,loss_ce,loss_cl
        else:
            return loss,x,ws
        



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