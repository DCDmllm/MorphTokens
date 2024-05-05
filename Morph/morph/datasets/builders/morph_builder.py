import os
import logging
import warnings

from morph.common.registry import registry
from morph.datasets.builders.base_dataset_builder import BaseDatasetBuilder
from morph.datasets.datasets.morph_dataset import LaionI2TDataset, LaionT2IDataset, NewImgDataset


@registry.register_builder("laion_i2t")
class LaionI2TBuilder(BaseDatasetBuilder):
    train_dataset_cls = LaionI2TDataset

    DATASET_CONFIG_DICT = {"default": "configs/datasets/laion_ccsbu/i2t.yaml"}

    def _download_ann(self):
        pass

    def _download_vis(self):
        pass

    def build(self):
        self.build_processors()

        build_info = self.config.build_info
        datasets = dict()
        split = "train"

        # create datasets
        dataset_cls = self.train_dataset_cls
        datasets[split] = dataset_cls(
            vis_processor=self.vis_processors[split],
            text_processor=self.text_processors[split],
            location=build_info.storage,
            vq_vis_processor=self.vq_vis_processors[split]
        ).inner_dataset

        return datasets


@registry.register_builder("laion_t2i")
class LaionT2IBuilder(LaionI2TBuilder):
    train_dataset_cls = LaionT2IDataset

    DATASET_CONFIG_DICT = {"default": "configs/datasets/laion_ccsbu/t2i.yaml"}



@registry.register_builder("coco")
class CocoBuilder(BaseDatasetBuilder):
	train_dataset_cls = NewImgDataset
	DATASET_CONFIG_DICT = {"default": "configs/datasets/coco.yaml"}


    

