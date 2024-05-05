import os
import json
from typing import Iterable
import linecache
from torch.utils.data import Dataset, ConcatDataset
from torch.utils.data.dataloader import default_collate
from PIL import Image
import webdataset as wds
import random
import torch
from morph.datasets.datasets.base_dataset import BaseDataset

from torchvision import transforms

def get_transform(type='clip', keep_ratio=False, image_size=224):
    if type == 'clip':
        transform = []
        if keep_ratio:
            transform.extend([
                transforms.Resize(image_size),
                transforms.CenterCrop(image_size),
            ])
        else:
            transform.append(transforms.Resize((image_size, image_size)))
        transform.extend([
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))
        ])

        return transforms.Compose(transform)
    else:
        raise NotImplementedError

class NewImgDataset(Dataset):
    def __init__(
        self, vis_processor=None, text_processor=None, vis_root=None, 
        ann_paths=[], dataset_type = "caption", vq_vis_processor=None, no_end=False, no_temp=False,config=None
    ):
        """
        vis_root (string): Root directory of images (e.g. coco/images/)
        ann_root (string): directory to store the annotation file
        """
        self.vis_root = vis_root
        self.ann_paths = ann_paths
        self.annotation = []
        for ann_path in ann_paths:
            nq = len(linecache.getlines(ann_path))
            for i in range(nq):
                json_line = linecache.getline(ann_path, i+1)
                content = json.loads(json_line)
                self.annotation.append(content)

        self.vis_processor = vis_processor
        self.text_processor = text_processor
        self.vq_vis_processor = vq_vis_processor

        self.no_end = no_end
        self.no_temp = no_temp
        self.config = config

        self.transform = get_transform()
        # self._add_instance_ids()

    def __len__(self):
        return len(self.annotation)

    def __getitem__(self, index):
        ann = self.annotation[index]
        input_image = ann["input_img"]
        if input_image is None:
            input_images = None
            
        elif isinstance(input_image, str):
            input_image_paths = os.path.join(self.vis_root, input_image)
            cur_image = Image.open(input_image_paths).convert("RGB")
            input_images = self.transform(cur_image)
            
        elif isinstance(input_image, list):
            input_images = []
            input_image_paths = [os.path.join(self.vis_root, img) for img in input_image]
            for image_path in input_image_paths:
                cur_image = Image.open(image_path).convert("RGB")
                cur_image = self.transform(cur_image).unsqueeze(1)
                input_images.append(cur_image)
            input_images = torch.cat(input_images, 1)
        
        output_image = ann["output_img"]
        if output_image is None:
            output_images = None
            output_images_vqgan = None
            
        elif isinstance(output_image, str):
            output_image_paths = os.path.join(self.vis_root, output_image)
            cur_image = Image.open(output_image_paths).convert("RGB")
            output_images = self.transform(cur_image)
            output_images_vqgan = self.vq_vis_processor(cur_image)
            
        elif isinstance(output_image, list):
            output_images, output_images_vqgan = [], []
            output_image_paths = [os.path.join(self.vis_root, img) for img in output_image]
            for image_path in output_image_paths:
                cur_image = Image.open(image_path).convert("RGB")
                cur_output_image = self.transform(cur_image).unsqueeze(1)
                cur_image_vqgan = self.vq_vis_processor(cur_image).unsqueeze(1)
                output_images.append(cur_output_image)
                output_images_vqgan.append(cur_image_vqgan)
            output_images = torch.cat(output_images, 1)
            output_images_vqgan = torch.cat(output_images, 1)
        
        text_input = str(ann["input_text"])
        text_output = str(ann["output_text"])
        
        result =  {
            "text_input": text_input,
            "text_output": text_output,
            "config":self.config,
        }
        if input_images is not None:
            result.update({"input_image": input_images})
            result.update({"input_image_path": input_image_paths})
        if output_image is not None:
            result.update({"output_image": output_images})
            result.update({"output_image_path": output_image_paths})
        if output_images_vqgan is not None:
            result.update({"output_image_vqgan": output_images_vqgan})

        return result

    def collater(self, samples):
        return default_collate(samples)

    def set_processors(self, vis_processor, text_processor, vq_vis_processor=None):
        self.vis_processor = vis_processor
        self.vq_vis_processor = vq_vis_processor
        self.text_processor = text_processor


class LaionI2TDataset(BaseDataset):
    def __init__(self, vis_processor, text_processor, location, vq_vis_processor=None):
        super().__init__(vis_processor=vis_processor, text_processor=text_processor, vq_vis_processor=vq_vis_processor)
        self.transform = get_transform()
        self.inner_dataset = wds.DataPipeline(
            wds.ResampledShards(location),
            wds.tarfile_to_samples(handler=wds.warn_and_continue),
            wds.shuffle(1000, handler=wds.warn_and_continue),
            wds.decode("pilrgb", handler=wds.warn_and_continue),
            wds.to_tuple("jpg", "json", handler=wds.warn_and_continue),
            wds.map_tuple(self.transform, handler=wds.warn_and_continue),
            wds.map(self.to_dict, handler=wds.warn_and_continue),
        )

    def to_dict(self, sample):
        return {
            "input_image": sample[0],
            "text_output": self.text_processor(sample[1]["caption"]),
        }

class LaionT2IDataset(BaseDataset):
    def __init__(self, vis_processor, text_processor, location, vq_vis_processor=None):
        super().__init__(vis_processor=vis_processor, text_processor=text_processor, vq_vis_processor=vq_vis_processor)

        self.inner_dataset = wds.DataPipeline(
            wds.ResampledShards(location),
            wds.tarfile_to_samples(handler=wds.warn_and_continue),
            wds.shuffle(1000, handler=wds.warn_and_continue),
            wds.decode("pilrgb", handler=wds.warn_and_continue),
            wds.to_tuple("jpg", "json", handler=wds.warn_and_continue),
            wds.map(self.to_dict, handler=wds.warn_and_continue),
        )
        self.transform = get_transform()

    def to_dict(self, sample):
        return {
            "output_image": self.transform(sample[0]),
            "text_input": self.text_processor(sample[1]["caption"]) + " <img>",
            "text_output": "<ImageHere> </img>",
            "output_image_vqgan": self.vq_vis_processor(sample[0]),
        }