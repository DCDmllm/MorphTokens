import logging
import random
import numpy as np
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_int8_training,
)
from PIL import Image
import torch
from morph.common.registry import registry
from torch.cuda.amp import autocast as autocast
import torch.nn as nn
from morph.models.base_model import BaseModel
from morph.models.llama_xformer import LlamaForCausalLM
from transformers import LlamaTokenizer
from transformers import StoppingCriteria, StoppingCriteriaList
from morph.models.visual_tokenizer import VisualTokenizer

IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "</s>"
DEFAULT_UNK_TOKEN = "</s>"
IMG_TOKEN = "[IMG]"
IMG_FLAG = "<ImageHere>"

rescale = lambda x: (x + 1.) / 2.

def chw_to_pillow(x):
    return Image.fromarray((255*rescale(x.detach().cpu().numpy().transpose(1,2,0))).clip(0,255).astype(np.uint8))

class StoppingCriteriaSub(StoppingCriteria):
    def __init__(self, stops=[], encounters=1):
        super().__init__()
        self.stops = stops

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        for stop in self.stops:
            if torch.all((stop == input_ids[0][-len(stop):])).item():
                return True

        return False

@registry.register_model("morph_mllm")
class MorphMLLM(BaseModel):
    
    PRETRAINED_MODEL_CONFIG_DICT = {
        "morph": "configs/models/morph_mllm.yaml",
    }
    
    def __init__(
        self,
        taming_config='',
        taming_ckpt=None,
        use_grad_checkpoint_llm=False,
        q_former_model=None,
        set_taming=True,
        freeze_qformer=True,
        num_query_token=32,
        prompt_path="",
        vit_model="eva_clip_g",
        img_size=224,
        drop_path_rate=0,
        use_grad_checkpoint=False,
        vit_precision="fp16",
        freeze_vit=True,
        llama_model_path="",
        llama_tokenizer_path="",
        torch_dtype="fp16",
        max_txt_len=32,
        max_context_len=3800,
        prompt_template="",
        cembed_dim=4096,
        code_project=False,
        end_sym='\n',
        low_resource=False,  # use 8 bit and put vit in cpu
        device_8bit=0,  # the device of 8bit model should be set when loading and cannot be changed anymore.
        lora_r=0,  # lora_r means lora is not used
        lora_target_modules=["q_proj", "v_proj"],
        lora_alpha=16,
        lora_dropout=0.05,
    ):
        super().__init__()
        # loading llama 
        self.llama_model, self.llama_tokenizer = self.init_llm(
            llama_model_path=llama_model_path,
            llama_tokenizer_path=llama_tokenizer_path,
            low_resource=low_resource,
            torch_dtype=torch_dtype,
            low_res_device=device_8bit,
            lora_r=lora_r,
            lora_target_modules=lora_target_modules,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
        )
        
        
        self.max_txt_len = max_txt_len
        self.max_context_len = max_context_len
        self.end_sym = end_sym
        
        self.NUM_IMG_TOKENS = num_query_token
        
        self.output_img_id = self.llama_tokenizer.convert_tokens_to_ids(IMG_TOKEN)
        embed_dim = self.llama_model.get_input_embeddings().weight.size(-1)
        
        self.autoencoder = VisualTokenizer(q_former_model,
                                            taming_config=taming_config,
                                            taming_ckpt=taming_ckpt,
                                            vit_model=vit_model,
                                            img_size=img_size,
                                            drop_path_rate=drop_path_rate,
                                            use_grad_checkpoint=use_grad_checkpoint,
                                            vit_precision=vit_precision,
                                            freeze_vit=freeze_vit,
                                            freeze_qformer=freeze_qformer,
                                            codebook_embed_dim=cembed_dim,
                                            embed_dim=embed_dim,
                                            code_project=code_project,
                                            set_taming=set_taming)

        # self.autoencoder.freeze()
        
        if use_grad_checkpoint_llm:
            self.llama_model.gradient_checkpointing_enable()
        
        self.prompt_template = prompt_template
        if prompt_path:
            with open(prompt_path, 'r') as f:
                raw_prompts = f.read().splitlines()
            filted_prompts = [raw_prompt for raw_prompt in raw_prompts if "<ImageHere>" in raw_prompt]
            self.prompt_list = [prompt_template.format(p) for p in filted_prompts]
            print('Load {} training prompts'.format(len(self.prompt_list)))
            print('Prompt Example \n{}'.format(random.choice(self.prompt_list)))
        else:
            self.prompt_list = []
        
        
    
    def init_llm(cls, llama_model_path, llama_tokenizer_path, low_resource=False, torch_dtype='fp16', low_res_device=0, lora_r=0,
                 lora_target_modules=["q_proj","v_proj"], **lora_kargs):
        logging.info('Loading LLAMA')
        llama_tokenizer = LlamaTokenizer.from_pretrained(llama_tokenizer_path, use_fast=False)
        llama_tokenizer.pad_token = "$$"
        
        
        if torch_dtype == 'fp16' or torch_dtype == 'float16':
            torch_dtype = torch.float16
        elif torch_dtype == 'bf16' or torch_dtype == 'bfloat16':
            torch_dtype = torch.bfloat16
        else:
            torch_dtype == torch.float32
            
        if low_resource:
            llama_model = LlamaForCausalLM.from_pretrained(
                llama_model_path,
                torch_dtype=torch_dtype,
                load_in_8bit=True,
                device_map={'': low_res_device}
            )
        else:
            llama_model = LlamaForCausalLM.from_pretrained(
                llama_model_path,
                torch_dtype=torch_dtype,
            )

        if lora_r > 0:
            llama_model = prepare_model_for_int8_training(llama_model)
            loraconfig = LoraConfig(
                r=lora_r,
                bias="none",
                task_type="CAUSAL_LM",
                target_modules=lora_target_modules,
                **lora_kargs
            )
            llama_model = get_peft_model(llama_model, loraconfig)
            ###
            for name, param in llama_model.named_parameters():
                if 'lm_head.0.weight' in name:
                    param.requires_grad = True
                    param.data = param.data.float()
            llama_model.print_trainable_parameters()

        else:
            for name, param in llama_model.named_parameters():
                param.requires_grad = False
        logging.info('Loading LLAMA Done')
        return llama_model, llama_tokenizer
    
    
    def encode_image(self, image):
        return self.autoencoder.encode_image(image)        
           
    def embed_tokens(self, token_ids):
        embeds = self.llama_model.get_input_embeddings()(token_ids)
        return embeds
    
    
    def concat_emb_input_output(self, input_embs, input_atts, output_embs, output_atts):
        """
        Concatenate the batched input embedding and batched output embedding together.
        Both the input and the output embedding should be right padded.
        """
        input_lens = []
        cat_embs = []
        cat_atts = []
        for i in range(input_embs.size(0)):
            input_len = input_atts[i].sum()
            input_lens.append(input_len)
            cat_embs.append(
                torch.cat([
                    input_embs[i][:input_len],
                    output_embs[i],
                    input_embs[i][input_len:]
                ])
            )
            cat_atts.append(
                torch.cat([
                    input_atts[i][:input_len],
                    output_atts[i],
                    input_atts[i][input_len:]
                ])
            )
        cat_embs = torch.stack(cat_embs)
        cat_atts = torch.stack(cat_atts)
        return cat_embs, cat_atts, input_lens
    
    
    def prompt_wrap(self, img_embeds, img_ids, atts_img, prompts, device, output_target_id=False):
        if prompts is None or len(prompts) == 0:
            # prompts is not provided, just return the original image embedding
            img_embeds = torch.cat(img_embeds, dim=1)
            atts_img = torch.cat(atts_img, dim=1)
            if not output_target_id:
                return img_embeds, atts_img
            else:
                img_ids = torch.cat(img_ids, dim=1)
                return img_embeds, atts_img, img_ids

        elif img_embeds is None:
            self.llama_tokenizer.padding_side = "right"
            prompt_tokens = self.llama_tokenizer(
                prompts,
                return_tensors="pt",
                padding="longest",
                add_special_tokens=False
            ).to(device)
            prompt_embeds = self.embed_tokens(prompt_tokens.input_ids)
            atts_prompt = prompt_tokens.attention_mask
            if not output_target_id:
                return prompt_embeds, atts_prompt
            else:
                target_ids = prompt_tokens.input_ids
                mask_target_ids = target_ids.masked_fill(
                    target_ids == self.llama_tokenizer.pad_token_id, -100
                )
                return prompt_embeds, atts_prompt, mask_target_ids
            
        else:
            emb_list = []
            id_list = []
            if isinstance(prompts, str):
                prompts = [prompts] * img_embeds[0].size(0)

            for idx, each_prompt in enumerate(prompts):
                interleave_emb, interleave_id = [], []
                p_segs = each_prompt.split('<ImageHere>')
                assert len(p_segs) > 1, "there is at least one image"
                assert len(p_segs) == len(img_embeds) + 1, "error1: Unmatched numbers of image placeholders and images."
                assert len(p_segs) == len(img_ids) + 1, "error2: Unmatched numbers of image placeholders and images."
                for iddx, seg in enumerate(p_segs[:-1]):
                    if seg == '':
                        interleave_emb.append(img_embeds[iddx][idx:idx+1, :, :])
                        interleave_id.append(img_ids[iddx][idx:idx+1, :])
                        continue
                    p_tokens = self.llama_tokenizer(
                        seg, return_tensors="pt", add_special_tokens=False).to(device)
                    p_embed = self.embed_tokens(p_tokens.input_ids)
                    interleave_emb.append(torch.cat([p_embed, img_embeds[iddx][idx:idx+1, :, :]], dim=1))
                    interleave_id.append(torch.cat([p_tokens.input_ids, img_ids[iddx][idx:idx+1, :]], dim=1))
                wrapped_emb = torch.cat(interleave_emb, dim=1)
                wrapped_id = torch.cat(interleave_id, dim=1)
                
                if p_segs[-1]!='':
                    p_tokens = self.llama_tokenizer(
                        p_segs[-1], return_tensors="pt", add_special_tokens=False).to(device)
                    p_embed = self.embed_tokens(p_tokens.input_ids)
                    wrapped_emb = torch.cat([wrapped_emb, p_embed], dim=1)
                    wrapped_id = torch.cat([wrapped_id, p_tokens.input_ids], dim=1)
                    
                emb_list.append(wrapped_emb)
                id_list.append(wrapped_id)
            
            emb_lens = [emb.shape[1] for emb in emb_list]
            pad_emb = self.embed_tokens(torch.tensor(self.llama_tokenizer.pad_token_id, device=device))
            
            max_length = max(emb_lens) if max(emb_lens) < self.max_context_len else self.max_context_len
            wrapped_embs = pad_emb.expand(len(emb_lens), max_length, -1).clone()
            wrapped_atts = torch.zeros([len(emb_lens), max_length], dtype=torch.long, device=device)
            wrapped_ids = torch.ones([len(emb_lens), max_length], dtype=torch.long).to(device).fill_(-100)
            
            for i, (emb, iddd) in enumerate(zip(emb_list, id_list)):
                length = emb_lens[i] if emb_lens[i] < self.max_context_len else self.max_context_len
                wrapped_embs[i, :length] = emb[:, :length]
                wrapped_atts[i, :length] = 1
                wrapped_ids[i, :length] = iddd[:, :length]
            if not output_target_id:
                return wrapped_embs, wrapped_atts
            else:
                return wrapped_embs, wrapped_atts, wrapped_ids

            
    
    def preparing_embedding(self, samples):
        input_img_embeds, input_img_ids, input_img_atts = [], [], []
        output_img_embeds, output_img_ids, output_img_atts = [], [], []
        device = self.device
        if 'input_image' in samples:
            input_image = samples["input_image"]
            bs = input_image.size(0)
            device = input_image.device
            if input_image.dim() == 5:
                for j in range(input_image.size(2)):
                    this_image = input_image[:,:,j,:,:]
                    img_embed, img_id, img_att = self.encode_image(this_image)
                    input_img_embeds.append(img_embed)
                    input_img_ids.append(img_id)
                    input_img_atts.append(img_att)

            else:
                img_embed, img_id, img_att = self.encode_image(input_image)
                input_img_embeds.append(img_embed)
                input_img_ids.append(img_id)
                input_img_atts.append(img_att)

        else:
            input_img_embeds, input_img_ids, input_img_atts = None, None, None    
        
        if 'output_image' in samples:
            assert 'output_image_vqgan' in samples, "no image information for vqgan generation!"
            output_image = samples["output_image"]
            output_image_vqgan = samples["output_image_vqgan"]
            
            bs = output_image.size(0)
            device = output_image.device
            if output_image.dim() == 5:
                 for j in range(output_image.size(2)):
                    this_image = output_image[:,:,j,:,:]
                    this_image_vqgan = output_image_vqgan[:,:,j,:,:]
                    this_image_vqgan = this_image_vqgan.permute(0, 3, 1, 2).to(memory_format=torch.contiguous_format)
                    if this_image_vqgan.dtype == torch.double:
                        this_image_vqgan = this_image_vqgan.float()
                
                    img_embed, img_id, img_att = self.encode_image(this_image)
                    output_img_embeds.append(img_embed)
                    output_img_ids.append(img_id)
                    output_img_atts.append(img_att)

            else:
                output_image_vqgan = output_image_vqgan.permute(0, 3, 1, 2).to(memory_format=torch.contiguous_format)
                if output_image_vqgan.dtype == torch.double:
                    output_image_vqgan = output_image_vqgan.float()
                
                img_embed, img_id, img_att = self.encode_image(output_image)
                output_img_embeds.append(img_embed)
                output_img_ids.append(img_id)
                output_img_atts.append(img_att)

        else:
            output_img_embeds, output_img_ids, output_img_atts = None, None, None
        
        
        return  (input_img_embeds, input_img_ids, input_img_atts), (output_img_embeds, output_img_ids, output_img_atts), (device, bs)

        

    # Stage1: initialization
    def forward(self, samples):
        
        input_embedding, output_embedding, device_bs = self.preparing_embedding(samples)
        input_img_embeds, input_img_ids, input_img_atts = input_embedding
        output_img_embeds, output_img_ids, output_img_atts = output_embedding
        device, bs = device_bs

        self.llama_tokenizer.padding_side = "right"
       
        output_text = [t + self.end_sym for t in samples["text_output"]]
        
        if 'text_input' in samples:  # VQA dataset
            prompt = samples["text_input"]
            if "no_temp" not in samples:
                prompt_list = []
                for p in prompt:
                    if p.endswith('<img>')==False:
                        p += '\n'
                    prompt_list.append(self.prompt_template.format(p))
                prompt=prompt_list
        else:
            prompt=None
        
        cond_embeds, cond_atts = self.prompt_wrap(input_img_embeds, input_img_ids, 
                                                  input_img_atts, prompt, 
                                                  device=device, output_target_id=False)
        
        regress_embeds, regress_atts, part_targets = self.prompt_wrap(output_img_embeds, output_img_ids, 
                                                                      output_img_atts, output_text, 
                                                                      device=device, output_target_id=True)
        
        # concat the embedding to condition and the embedding to regress
        inputs_embeds, attention_mask, input_lens = \
            self.concat_emb_input_output(cond_embeds, cond_atts, regress_embeds, regress_atts)
        
        # get bos token embedding
        bos = torch.ones_like(part_targets[:, :1]) * self.llama_tokenizer.bos_token_id
        bos_embeds = self.embed_tokens(bos)
        bos_atts = cond_atts[:, :1]
        
        # add bos token at the begining
        inputs_embeds = torch.cat([bos_embeds, inputs_embeds], dim=1)
        attention_mask = torch.cat([bos_atts, attention_mask], dim=1)

        targets = torch.ones([inputs_embeds.shape[0], inputs_embeds.shape[1]],
                             dtype=torch.long).to(self.device).fill_(-100)
        
        for i, target in enumerate(part_targets):
            targets[i, input_lens[i]+1:input_lens[i]+len(target)+1] = target  # plus 1 for bos
        
        with self.maybe_autocast():
            outputs = self.llama_model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                return_dict=True,
                labels=targets,
            )
        loss_llm = outputs.loss

        return {"loss": loss_llm}
    
    def image_generation(self, img_ids, device, fns):
        img_ids = img_ids[:, :32]
        bz, num = img_ids.size(0), img_ids.size(1)
        img_ids = img_ids.view(-1, 1)
        min_encodings = torch.zeros(img_ids.shape[0], self.autoencoder.image_tokenizer.quantizer.n_e).to(device)
        min_encodings.scatter_(1, img_ids, 1)
        embed = torch.matmul(min_encodings, self.autoencoder.image_tokenizer.quantizer.embedding_weight).view(bz, num, -1)
        x_sample = self.autoencoder.decode_image(embed = embed)[0]
        x = chw_to_pillow(x_sample)
        x.save(fns)
        return fns
        
    
    def generate(self, 
                 images,
                 text,
                 device='cuda:0', 
                 max_new_tokens=100,
                 num_beams=1,
                 min_length=1, 
                 top_p=0.9,
                 repetition_penalty=1.0, 
                 length_penalty=1,
                 do_sample=False,
                 temperature=1.0,
                 max_length=2000,
                 fns = './1.jpg'):
        if text.endswith('<img>')==False:
            text += '\n'
        text = self.prompt_template.format(text)
        texts = text.split('<ImageHere>')
        
        img_prompts = []
        for cur_image in images:
            _, img_id, _ = self.encode_image(cur_image)
            img_id = img_id[0,:]
            cur_prompt = self.llama_tokenizer.decode(img_id)
            img_prompts.append(cur_prompt)
            
        assert len(texts) == len(img_prompts)+1

        prompt = texts[0]
        for pid in range(len(img_prompts)):
            prompt += img_prompts[pid]
            prompt += texts[pid+1]
        
        inputs = self.llama_tokenizer(prompt, return_tensors="pt", add_special_tokens=True).to(device)
        input_ids = inputs.input_ids
        
        stop_words_ids = [torch.tensor([835]).to(device), torch.tensor([2277, 29937]).to(device)]
        stopping_criteria = StoppingCriteriaList([StoppingCriteriaSub(stops=stop_words_ids)])
        
        with self.maybe_autocast():  
            outputs = self.llama_model.generate(
                input_ids = input_ids,
                max_new_tokens=max_new_tokens,
                num_beams=num_beams,
                length_penalty=length_penalty,
                temperature=temperature,
                do_sample=do_sample,
                min_length=min_length,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                stopping_criteria=stopping_criteria,
            )
        outputs = outputs[0][input_ids.shape[1]:]
        boi_list = torch.where(outputs == self.llama_tokenizer('<img>', add_special_tokens=False).input_ids[0])[0]
        eoi_list = torch.where(outputs == self.llama_tokenizer('</img>', add_special_tokens=False).input_ids[0])[0]
        if len(boi_list) == 0 and len(eoi_list) == 0:
            text_ids = outputs
            texts = self.llama_tokenizer.decode(text_ids, skip_special_tokens=True)
            texts = texts.split('###')[0].strip()
            return texts, None
        else:
            boi_index = boi_list[0]
            eoi_index = eoi_list[0]
            image_ids = (outputs[boi_index+1:eoi_index] - 32000).reshape(1,-1)
            fns = self.image_generation(image_ids, device, fns)
            
            text_ids_1 = outputs[:boi_index+1]
            text_ids_2 = outputs[eoi_index:]
            text_ids = torch.cat([text_ids_1, text_ids_2])
            if len(text_ids)!=0:
                texts = self.llama_tokenizer.decode(text_ids, skip_special_tokens=True)
                texts = texts.split('###')[0].strip()
            else:
                texts = None
            
            return texts, fns
            
        
    @classmethod
    def from_config(cls, cfg):
        taming_config = cfg.get("taming_config", None)
        taming_ckpt = cfg.get("taming_ckpt", None)
        set_taming = cfg.get("set_taming", True)
        
        q_former_model=cfg.get("q_former_model", None)
        # has_qformer=cfg.get("has_qformer", True),
        freeze_qformer=cfg.get("freeze_qformer", True)
        num_query_token=cfg.get("num_query_token", 32)
        prompt_path=cfg.get("prompt_path", None)
        
        vit_model = cfg.get("vit_model", "eva_clip_g")
        img_size = cfg.get("image_size")
        llama_model_path = cfg.get("llama_model_path", "")
        llama_tokenizer_path = cfg.get("llama_tokenizer_path", "")
        torch_dtype = cfg.get("torch_dtype", "fp16")

        drop_path_rate = cfg.get("drop_path_rate", 0)
        use_grad_checkpoint = cfg.get("use_grad_checkpoint", False)
        vit_precision = cfg.get("vit_precision", "fp16")
        freeze_vit = cfg.get("freeze_vit", True)
        low_resource = cfg.get("low_resource", False)

        prompt_template = cfg.get("prompt_template", '[INST] {} [/INST]')
        max_txt_len = cfg.get("max_txt_len", 300)
        end_sym = cfg.get("end_sym", '\n')

        lora_r = cfg.get("lora_r", 64)
        lora_alpha = cfg.get("lora_alpha", 16)

        use_grad_checkpoint_llm = cfg.get("use_grad_checkpoint_llm", False)
        max_context_len = cfg.get("max_context_len", 3800)
        print(num_query_token, lora_r, max_txt_len)
        model = cls(
            taming_config=taming_config,
            taming_ckpt=taming_ckpt,
            q_former_model = q_former_model,
            set_taming = set_taming,
            # has_qformer = has_qformer,
            freeze_qformer = freeze_qformer,
            num_query_token = num_query_token,
            prompt_path = prompt_path,
            vit_model=vit_model,
            img_size=img_size,
            drop_path_rate=drop_path_rate,
            use_grad_checkpoint=use_grad_checkpoint,
            vit_precision=vit_precision,
            freeze_vit=freeze_vit,
            llama_model_path=llama_model_path,
            llama_tokenizer_path=llama_tokenizer_path,
            torch_dtype=torch_dtype,
            
            prompt_template=prompt_template,
            max_txt_len=max_txt_len,
            low_resource=low_resource,
            end_sym=end_sym,
            lora_r=lora_r,
            lora_alpha=lora_alpha,
            use_grad_checkpoint_llm=use_grad_checkpoint_llm,
            max_context_len=max_context_len,
        )

        ckpt_path = cfg.get("ckpt", "")  # load weights of Morph
        if ckpt_path:
            ckpt = torch.load(ckpt_path, map_location="cpu")
            msg = model.load_state_dict(ckpt['model'], strict=False)
            print("Load Morph-MLLM Checkpoint (ckpt): {}".format(ckpt_path))
        else:
            print("no ckpt")
        
        return model
    