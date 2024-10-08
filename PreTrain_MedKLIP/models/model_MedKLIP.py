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




class MedKLIP(nn.Module):

    def __init__(self, config, ana_book, disease_book, mode='train'):
        super(MedKLIP, self).__init__()

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
        self.is_resnet = config['res_base_model'] in ['resnet18','resnet50']
        self.model_type = config['res_base_model'] 
        if self.is_resnet:
            self.resnet_dict = {"resnet18": models.resnet18(pretrained=False),
                                "resnet50": models.resnet50(pretrained=False)}
            resnet = self._get_res_basemodel(config['res_base_model'])
            num_ftrs = int(resnet.fc.in_features/2)
            self.res_features = nn.Sequential(*list(resnet.children())[:-3])
            self.res_l1 = nn.Linear(num_ftrs, num_ftrs)
            self.res_l2 = nn.Linear(num_ftrs, self.d_model)
        elif config['res_base_model'] == 'dino_vit':
            self.encoder = AutoModel.from_pretrained(config['res_base_model'])
            for param in self.encoder.parameters():
                param.requires_grad = False

        elif config['res_base_model'] == 'sam':
            self.encoder = create_model('timm/samvit_base_patch16.sa1b', pretrained=True, num_classes=0)
            for param in self.encoder.parameters():
                param.requires_grad = False

        ###################################
        ''' Query Decoder'''
        ###################################

        self.H = config['H'] 
        decoder_layer = TransformerDecoderLayer(self.d_model, config['H'] , 1024,
                                        0.1, 'relu',normalize_before=True)
        decoder_norm = nn.LayerNorm(self.d_model)
        self.decoder = TransformerDecoder(decoder_layer, config['N'] , decoder_norm,
                                return_intermediate=False)
            

        # Learnable Queries
        #self.query_embed = nn.Embedding(config['num_queries'] ,self.d_model)
        self.dropout_feas = nn.Dropout(config['dropout'] )

        # Attribute classifier
        self.classifier = nn.Linear(self.d_model,config['attribute_set_size'])

        # # Class classifier
        # self.cls_classifier = nn.Linear(self.d_model,args.num_classes)

        self.apply(self._init_weights)

    def _get_res_basemodel(self, res_model_name):
        try:
            res_model = self.resnet_dict[res_model_name]
            print("Image feature extractor:", res_model_name)
            return res_model
        except:
            raise ("Invalid model name. Check the config file and pass one of: resnet18 or resnet50")

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
    
    def image_encoder(self, xis):
        #patch features
        """
        16 torch.Size([16, 1024, 14, 14])
        torch.Size([16, 196, 1024])
        torch.Size([3136, 1024])
        torch.Size([16, 196, 256])
        """
        batch_size = xis.shape[0]
        res_fea = self.res_features(xis) #batch_size,feature_size,patch_num,patch_num
        res_fea = rearrange(res_fea,'b d n1 n2 -> b (n1 n2) d')
        h = rearrange(res_fea,'b n d -> (b n) d')
        #batch_size,num,feature_size
        # h = h.squeeze()
        x = self.res_l1(h)
        x = F.relu(x)
        
        
        x = self.res_l2(x)
        out_emb = rearrange(x,'(b n) d -> b n d',b=batch_size)
        return out_emb
    
    def vit_encoder_forward(self, images):
        # Extract features using DINO encoder
        x = self.encoder(images)
        x = x.last_hidden_state  
        # x = self.dino_projector(x)  # Project to the correct dimension
        return x

    def forward(self, images,labels,smaple_index = None, is_train = True, no_cl= False, exclude_class= False):

        # labels batch,51,75 binary_label batch,75 sample_index batch,index
        B = images.shape[0]
        device = images.device
        ''' Visual Backbone '''
        if self.is_resnet:
            x = self.image_encoder(images) #batch_size,patch_num,dim
            # print(x.shape) torch.Size([48, 196, 256])
        elif self.model_type == 'sam':
            x = self.encoder.forward_features(images)
            # print(x.shape) torch.Size([4, 256, 64, 64])
            # Flatten the spatial dimensions: [B, C, H, W] -> [B, C, H*W]
            B, C, H, W = x.shape
            x = x.view(B, C, H * W)  # Shape: [4, 256, 4096]

            # Permute to [B, H*W, C] -> [4, 4096, 256]
            x = x.permute(0, 2, 1)  # Shape: [4, 4096, 256]

        else:
            x = self.vit_encoder_forward(images)
            # print(x.shape)
            # 785, 768
    
        features = x.transpose(0,1) #patch_num b dim
        # print(features.shape)
        #query_embed = self.query_embed.weight.unsqueeze(1).repeat(1, B, 1) # query_number, batch, dim
        query_embed = self.disease_embedding_layer(self.disease_book)
        query_embed = query_embed.unsqueeze(1).repeat(1, B, 1)


        features,ws = self.decoder(query_embed, features, 
        memory_key_padding_mask=None, pos=None, query_pos=None)

        # print("BBBBBBBBBBBBBBBBBBBBBB")
        # print(features.shape)
        # torch.Size([75, 48, 256])

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