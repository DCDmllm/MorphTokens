import os
from PIL import Image
from omegaconf import OmegaConf
import contextlib
import numpy as np
import logging
import torch.nn.functional as F

import torch
import time
import torch.nn as nn
from transformers import LlamaTokenizer
from typing import Any, Dict, List, Optional, Union
from einops import rearrange
from morph.common.dist_utils import download_cached_file
from morph.common.utils import get_abs_path, is_url
from morph.models.eva_vit import create_eva_vit_g
from morph.models.base_model import LayerNorm, disabled_train
from morph.models.Qformer import BertConfig, BertLMHeadModel
from morph.models.quantize import VectorQuantizer
from morph.models.taming.modules.transformer.mingpt import sample_with_past_new
from morph.models.taming.util import instantiate_from_config

rescale = lambda x: (x + 1.) / 2.

def chw_to_pillow(x):
    return Image.fromarray((255*rescale(x.detach().cpu().numpy().transpose(1,2,0))).clip(0,255).astype(np.uint8))

def load_model_from_config(config, sd, gpu=True, eval_mode=True):
    model = instantiate_from_config(config)
    if sd is not None:
        model.load_state_dict(sd, strict=False)
    if gpu:
        model.cuda()
    if eval_mode:
        model.eval()
    return {"model": model}

def save_from_logs(logs, logdir, key="samples"):
    xx = logs[key]
    all_x = []
    for i, x in enumerate(xx):
        x = chw_to_pillow(x)
        all_x.append(x)
        if logdir is not None:
            x.save(os.path.join(logdir, f"{i:06}.png"))
    return all_x

class ImageTokenizer(nn.Module):
    def __init__(self,
                 q_former_model=None,
                 vit_model="eva_clip_g",
                 img_size=224,
                 drop_path_rate=0,
                 use_grad_checkpoint=False,
                 vit_precision="fp16",
                 freeze_vit=True,
                 freeze_qformer=True,
                 num_query_token=32,
                 codebook_embed_dim=32):
        
        super().__init__()
        self.visual_encoder, self.ln_vision = self.init_vision_encoder(
            vit_model, img_size, drop_path_rate, use_grad_checkpoint, vit_precision, freeze_vit
        )
        self.Qformer, self.query_tokens = self.init_Qformer(
            num_query_token, self.visual_encoder.num_features, freeze_qformer
        )
        
        # codebook_embed_dim = embedding.weight.size(-1)
        self.encode_task_layer = nn.Sequential(
            nn.Linear(self.Qformer.config.hidden_size, self.Qformer.config.hidden_size),
            nn.Tanh(),
            nn.Linear(self.Qformer.config.hidden_size, codebook_embed_dim)  # for quantize
        )
        
        self.quantizer = VectorQuantizer(hidden_size=codebook_embed_dim)
        if q_former_model is not None and q_former_model!='':
            self.load_from_pretrained(url_or_filename=q_former_model)
        
        
    def load_from_pretrained(self, url_or_filename):
        if is_url(url_or_filename):
            cached_file = download_cached_file(
                url_or_filename, check_hash=False, progress=True
            )
            checkpoint = torch.load(cached_file, map_location="cpu")
        elif os.path.isfile(url_or_filename):
            checkpoint = torch.load(url_or_filename, map_location="cpu")
        else:
            raise RuntimeError("checkpoint url or path is invalid")

        state_dict = checkpoint

        msg = self.load_state_dict(state_dict, strict=False)

        logging.info("load checkpoint from %s" % url_or_filename)
        return msg
    
    
    def vit_to_cpu(self):
        self.ln_vision.to("cpu")
        self.ln_vision.float()
        self.visual_encoder.to("cpu")
        self.visual_encoder.float()
    
    
    def maybe_autocast(self, dtype=torch.float16, device='cuda:0'):
        enable_autocast = device != torch.device("cpu")

        if enable_autocast:
            return torch.cuda.amp.autocast(dtype=dtype)
        else:
            return contextlib.nullcontext()
    
    
    @classmethod
    def init_vision_encoder(
        cls, model_name, img_size, drop_path_rate, use_grad_checkpoint, precision, freeze
    ):
        logging.info('Loading VIT')

        assert model_name == "eva_clip_g", "vit model must be eva_clip_g for current version of Morph-MLLM"
        if not freeze:
            precision = "fp32"  # fp16 is not for training

        visual_encoder = create_eva_vit_g(
            img_size, drop_path_rate, use_grad_checkpoint, precision
        )

        ln_vision = LayerNorm(visual_encoder.num_features)

        if freeze:
            for name, param in visual_encoder.named_parameters():
                param.requires_grad = False
            visual_encoder = visual_encoder.eval()
            visual_encoder.train = disabled_train
            for name, param in ln_vision.named_parameters():
                param.requires_grad = False
            ln_vision = ln_vision.eval()
            ln_vision.train = disabled_train
            logging.info("freeze vision encoder")

        logging.info('Loading VIT Done')
        return visual_encoder, ln_vision
    
    
    @classmethod
    def init_Qformer(cls, num_query_token, vision_width, freeze):
        encoder_config = BertConfig.from_pretrained("bert-base-uncased")
        encoder_config.encoder_width = vision_width
        # insert cross-attention layer every other block
        encoder_config.add_cross_attention = True
        encoder_config.cross_attention_freq = 2
        encoder_config.query_length = num_query_token
        Qformer = BertLMHeadModel(config=encoder_config)
        print(num_query_token, encoder_config.hidden_size)
        query_tokens = nn.Parameter(
            torch.zeros(1, num_query_token, encoder_config.hidden_size)
        )
        query_tokens.data.normal_(mean=0.0, std=encoder_config.initializer_range)

        Qformer.cls = None
        Qformer.bert.embeddings.word_embeddings = None
        Qformer.bert.embeddings.position_embeddings = None
        for layer in Qformer.bert.encoder.layer:
            layer.output = None
            layer.intermediate = None

        if freeze:
            for name, param in Qformer.named_parameters():
                if "dic" not in name and "prior" not in name and "slotattention" not in name:
                    param.requires_grad = False

            query_tokens.requires_grad = False
            logging.info("freeze Qformer")

        return Qformer, query_tokens
    
    
            
    def get_codebook_indices(self, image):
        with self.maybe_autocast():
            image_embeds = self.ln_vision(self.visual_encoder(image))
            image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(image.device)
        
        query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)
        query_output = self.Qformer.bert(
            query_embeds=query_tokens,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_atts,
            return_dict=True,
        )

        query_output_down = self.encode_task_layer(query_output.last_hidden_state)
        return self.quantizer(query_output_down)



class VisualTokenizer(nn.Module):
    def __init__(self, 
                 # image tokenize 
                 q_former_model,
                 taming_config, 
                 taming_ckpt,
                 vit_model="eva_clip_g",
                 img_size=224,
                 drop_path_rate=0,
                 use_grad_checkpoint=False,
                 vit_precision="fp16",
                 freeze_vit=True,
                 freeze_qformer=True,
                 num_query_token=32,
                 codebook_embed_dim=32,
                 embed_dim=4096,
                 set_taming=True,
                 code_project=False,):
        
        super().__init__()
        self.image_tokenizer = ImageTokenizer(codebook_embed_dim=codebook_embed_dim,
                                              q_former_model=q_former_model,
                                              vit_model=vit_model,
                                              img_size=img_size,
                                              drop_path_rate=drop_path_rate,
                                              use_grad_checkpoint=use_grad_checkpoint,
                                              vit_precision=vit_precision,
                                              freeze_vit=freeze_vit,
                                              freeze_qformer=freeze_qformer,
                                              num_query_token=num_query_token,)      
        
        # config, ckpt, gpu, eval_mode
        taming_model, _ = self.load_taming_model(config=taming_config, 
                                                 ckpt=taming_ckpt, 
                                                 gpu=True, 
                                                 eval_mode=self.training)
        
        codebook_embed_dim = codebook_embed_dim
        decoder_dim = taming_model.transformer.n_embd
        
        if code_project:
            self.code_project = nn.Linear(codebook_embed_dim, embed_dim)
        else:
            self.code_project = None
        
        if set_taming:
            self.taming_model = taming_model
            
            self.decode_task_layer = nn.Sequential(
                nn.Linear(codebook_embed_dim, codebook_embed_dim),
                nn.Tanh(),
                nn.Linear(codebook_embed_dim, decoder_dim)  
            )  
    
    
    def load_taming_model(self, config, ckpt, gpu, eval_mode):
        if ckpt:
            pl_sd = torch.load(ckpt, map_location="cpu")
            # global_step = pl_sd["global_step"]
            global_step = None
            print(f"loaded taming model from global step {global_step}.")
        else:
            pl_sd = {"state_dict": None}
            global_step = None
        config = OmegaConf.load(config)
        print(config)
        model = load_model_from_config(config.model, pl_sd["state_dict"], gpu=gpu, eval_mode=eval_mode)["model"]
        return model, global_step

    
    def freeze(self):
        for name, param in self.image_tokenizer.named_parameters():
                param.requires_grad = False
        self.image_tokenizer = self.image_tokenizer.eval()
        self.image_tokenizer.train = disabled_train


    def encode_image(self, images):
        quant, embed_ind = self.image_tokenizer.get_codebook_indices(images)
        if self.code_project is not None:
            quant = self.code_project(quant)
        img_att = torch.ones(quant.size()[:-1], dtype=torch.long).to(images.device)
        
        return quant, embed_ind, img_att
        
    
    def forward(self, vqgan_image=None, quant=None):
        query_output_up = self.decode_task_layer(quant)
        logits, target = self.taming_model(x=vqgan_image, c=query_output_up)
        loss_regressive = F.cross_entropy(logits.reshape(-1, logits.size(-1)), target.reshape(-1))
        return loss_regressive
    

    def decode_image(self, embed=None, steps=256, temperature=1.0, top_k=250, callback=None, dim_z=256, h=16, w=16, verbose_time=False, top_p=1.0):
        
        batch_size = embed.size(0)
        qzshape = [batch_size, dim_z, h, w]
        
        quant_embedding = embed
        query_output_up = self.decode_task_layer(quant_embedding)
        
        t1 = time.time()
        index_sample = sample_with_past_new(query_output_up, 
                                            self.taming_model.transformer, steps=steps,
                                            sample_logits=True, top_k=top_k, callback=callback,
                                            temperature=temperature, top_p=top_p)
        if verbose_time:
            sampling_time = time.time() - t1
            print(f"Full sampling takes about {sampling_time:.2f} seconds.")
        x_sample = self.taming_model.decode_to_img(index_sample, qzshape)
        return x_sample
            