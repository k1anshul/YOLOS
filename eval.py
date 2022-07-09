import torch
import torchvision
import os
import yaml
import argparse
import numpy as np
from transformers import AutoFeatureExtractor
from transformers import DetrConfig, AutoModelForObjectDetection
import pytorch_lightning as pl
from utils.coco_eval import CocoEvaluator
from utils import get_coco_api_from_dataset


# Loading inference config file
parser = argparse.ArgumentParser()
parser.add_argument('--hyp', type=str, default=r"config/hyp_eval.yaml",
                    help="path of eval config file")  # parameter to add hyp.yaml file
opt = parser.parse_args()
hyp = yaml.load(open(opt.hyp), Loader=yaml.FullLoader)

feature_extractor = AutoFeatureExtractor.from_pretrained(
    hyp['model_name'], size=hyp['imgz'])

id2label={k:v for k,v in enumerate(hyp['id2label'])}

class CocoDetection(torchvision.datasets.CocoDetection):
    def __init__(self, img_folder, feature_extractor, train=True):
        ann_file = os.path.join(img_folder, "_annotations.coco.json")
        super(CocoDetection, self).__init__(img_folder, ann_file)
        self.feature_extractor = feature_extractor

    def __getitem__(self, idx):
        # read in PIL image and target in COCO format
        img, target = super(CocoDetection, self).__getitem__(idx)

        # preprocess image and target (converting target to DETR format, resizing + normalization of both image and target)
        image_id = self.ids[idx]
        target = {'image_id': image_id, 'annotations': target}
        encoding = self.feature_extractor(
            images=img, annotations=target, return_tensors="pt")
        # remove batch dimension
        pixel_values = encoding["pixel_values"].squeeze()
        target = encoding["labels"][0]  # remove batch dimension

        return pixel_values, target


def collate_fn(batch):
    pixel_values = [item[0] for item in batch]
    encoding = feature_extractor.pad(pixel_values, return_tensors="pt")
    labels = [item[1] for item in batch]
    batch = {}
    batch['pixel_values'] = encoding['pixel_values']
    batch['labels'] = labels
    return batch


# we wrap our model around pytorch lightning for training
class YoloS(pl.LightningModule):

    def __init__(self, lr, weight_decay, model_name):
        super().__init__()
        # replace COCO classification head with custom head
        self.model = AutoModelForObjectDetection.from_pretrained(
            model_name, num_labels=len(id2label), ignore_mismatched_sizes=True)
        self.lr = lr
        self.weight_decay = weight_decay
        self.save_hyperparameters()  # adding this will save the hyperparameters to W&B too

    def forward(self, pixel_values):
        outputs = self.model(pixel_values=pixel_values)

        return outputs

    def common_step(self, batch, batch_idx):
        pixel_values = batch["pixel_values"]
        labels = [{k: v.to(self.device) for k, v in t.items()}
                  for t in batch["labels"]]

        outputs = self.model(pixel_values=pixel_values, labels=labels)

        loss = outputs.loss
        loss_dict = outputs.loss_dict

        return loss, loss_dict

    def training_step(self, batch, batch_idx):
        loss, loss_dict = self.common_step(batch, batch_idx)
        # logs metrics for each training_step,
        # and the average across the epoch
        # logging metrics with a forward slash will ensure the train and validation metrics as split into 2 separate sections in the W&B workspace
        self.log("train/loss", loss)
        for k, v in loss_dict.items():
            # logging metrics with a forward slash will ensure the train and validation metrics as split into 2 separate sections in the W&B workspace
            self.log("train/" + k, v.item())

        return loss

    def validation_step(self, batch, batch_idx):
        loss, loss_dict = self.common_step(batch, batch_idx)
        # logging metrics with a forward slash will ensure the train and validation metrics as split into 2 separate sections in the W&B workspace
        self.log("validation/loss", loss)
        for k, v in loss_dict.items():
            # logging metrics with a forward slash will ensure the train and validation metrics as split into 2 separate sections in the W&B workspace
            self.log("validation/" + k, v.item())

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        return optimizer

    def train_dataloader(self):
        return train_dataloader

    def val_dataloader(self):
        return val_dataloader

val_dataset = CocoDetection(img_folder=(
    hyp['test_data']), feature_extractor=feature_extractor, train=False)

val_dataloader = DataLoader(
    val_dataset, collate_fn=collate_fn, batch_size=hyp['validation_bz'])

# Loading trained model
model = torch.load(hyp['model_path'])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

print("Running evaluation...")

for idx, batch in enumerate(tqdm(val_dataloader)):
    # get the inputs
    pixel_values = batch["pixel_values"].to(device)
    # these are in DETR format, resized + normalized
    labels = [{k: v.to(device) for k, v in t.items()} for t in batch["labels"]]

    # forward pass
    outputs = model.model(pixel_values=pixel_values)

    orig_target_sizes = torch.stack(
        [target["orig_size"] for target in labels], dim=0)
    # convert outputs of model to COCO api
    results = feature_extractor.post_process(outputs, orig_target_sizes)
    res = {target['image_id'].item(): output for target,
           output in zip(labels, results)}
    coco_evaluator.update(res)

coco_evaluator.synchronize_between_processes()
coco_evaluator.accumulate()
coco_evaluator.summarize()
# the evaluation here prints out mean average precision details
