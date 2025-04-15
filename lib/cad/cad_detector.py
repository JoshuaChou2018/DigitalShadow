# -*- coding: utf-8 -*-
"""
@Time ： 2024/6/6 17:32
@Auth ： Juexiao Zhou
@File ：infer.py
@IDE ：PyCharm
@Page: www.joshuachou.ink
"""

import sys
from timm.models import create_model
import torch
from collections import OrderedDict
from beit2.optim_factory import create_optimizer, get_parameter_groups, LayerDecayValueAssigner
from beit2 import imagenet_a_r_indices, utils, modeling_finetune
from scipy import interpolate
import numpy as np
import argparse
from PIL import Image
from torchvision import transforms
import torch.nn.functional as F
import cv2

def load_model(args):
    model = create_model(
        args.diff_arch,
        pretrained=False,
        num_classes=args.diff_nb_classes,
        drop_rate=args.diff_drop,
        drop_path_rate=args.diff_drop_path,
        attn_drop_rate=args.diff_attn_drop_rate,
        drop_block_rate=None,
        use_mean_pooling=True,
        init_scale=args.diff_init_scale,
        use_rel_pos_bias=True,
        use_abs_pos_emb=False,
        init_values=args.diff_layer_scale_init_value,
        qkv_bias=True,
    )

    patch_size = model.patch_embed.patch_size
    print("Patch size = %s" % str(patch_size))
    args.window_size = (args.diff_input_size // patch_size[0], args.diff_input_size // patch_size[1])
    args.patch_size = patch_size

    if args.diff_finetune:
        if args.diff_finetune.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.diff_finetune, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.diff_finetune, map_location='cpu')

        print("Load ckpt from %s" % args.diff_finetune)
        checkpoint_model = None
        for model_key in args.diff_model_key.split('|'):
            if model_key in checkpoint:
                checkpoint_model = checkpoint[model_key]
                print("Load state_dict by model_key = %s" % model_key)
                break
        if checkpoint_model is None:
            checkpoint_model = checkpoint
        if (checkpoint_model is not None) and (args.diff_model_filter_name != ''):
            all_keys = list(checkpoint_model.keys())
            new_dict = OrderedDict()
            for key in all_keys:
                if key.startswith('encoder.'):
                    new_dict[key[8:]] = checkpoint_model[key]
                else:
                    pass
            checkpoint_model = new_dict

        state_dict = model.state_dict()
        for k in ['head.weight', 'head.bias']:
            if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
                if args.diff_robust_test == 'imagenet_r':
                    mask = torch.tensor(imagenet_a_r_indices.imagenet_r_mask)
                    checkpoint_model[k] = checkpoint_model[k][mask]
                elif args.diff_robust_test == 'imagenet_a':
                    mask = torch.tensor(imagenet_a_r_indices.imagenet_a_mask)
                    checkpoint_model[k] = checkpoint_model[k][mask]
                else:
                    print(f"Removing key {k} from pretrained checkpoint")
                    del checkpoint_model[k]

        if getattr(model, 'use_rel_pos_bias',
                   False) and "rel_pos_bias.relative_position_bias_table" in checkpoint_model:
            print("Expand the shared relative position embedding to each transformer block. ")
            num_layers = model.get_num_layers()
            rel_pos_bias = checkpoint_model["rel_pos_bias.relative_position_bias_table"]
            for i in range(num_layers):
                checkpoint_model["blocks.%d.attn.relative_position_bias_table" % i] = rel_pos_bias.clone()

            checkpoint_model.pop("rel_pos_bias.relative_position_bias_table")

        all_keys = list(checkpoint_model.keys())
        for key in all_keys:
            if "relative_position_index" in key:
                checkpoint_model.pop(key)

            if "relative_position_bias_table" in key:
                rel_pos_bias = checkpoint_model[key]
                src_num_pos, num_attn_heads = rel_pos_bias.size()
                dst_num_pos, _ = model.state_dict()[key].size()
                dst_patch_shape = model.patch_embed.patch_shape
                if dst_patch_shape[0] != dst_patch_shape[1]:
                    raise NotImplementedError()
                num_extra_tokens = dst_num_pos - (dst_patch_shape[0] * 2 - 1) * (dst_patch_shape[1] * 2 - 1)
                src_size = int((src_num_pos - num_extra_tokens) ** 0.5)
                dst_size = int((dst_num_pos - num_extra_tokens) ** 0.5)
                if src_size != dst_size:
                    print("Position interpolate for %s from %dx%d to %dx%d" % (
                        key, src_size, src_size, dst_size, dst_size))
                    extra_tokens = rel_pos_bias[-num_extra_tokens:, :]
                    rel_pos_bias = rel_pos_bias[:-num_extra_tokens, :]

                    def geometric_progression(a, r, n):
                        return a * (1.0 - r ** n) / (1.0 - r)

                    left, right = 1.01, 1.5
                    while right - left > 1e-6:
                        q = (left + right) / 2.0
                        gp = geometric_progression(1, q, src_size // 2)
                        if gp > dst_size // 2:
                            right = q
                        else:
                            left = q

                    # if q > 1.090307:
                    #     q = 1.090307

                    dis = []
                    cur = 1
                    for i in range(src_size // 2):
                        dis.append(cur)
                        cur += q ** (i + 1)

                    r_ids = [-_ for _ in reversed(dis)]

                    x = r_ids + [0] + dis
                    y = r_ids + [0] + dis

                    t = dst_size // 2.0
                    dx = np.arange(-t, t + 0.1, 1.0)
                    dy = np.arange(-t, t + 0.1, 1.0)

                    print("Original positions = %s" % str(x))
                    print("Target positions = %s" % str(dx))

                    all_rel_pos_bias = []

                    for i in range(num_attn_heads):
                        z = rel_pos_bias[:, i].view(src_size, src_size).float().numpy()
                        f = interpolate.interp2d(x, y, z, kind='cubic')
                        all_rel_pos_bias.append(
                            torch.Tensor(f(dx, dy)).contiguous().view(-1, 1).to(rel_pos_bias.device))

                    rel_pos_bias = torch.cat(all_rel_pos_bias, dim=-1)

                    new_rel_pos_bias = torch.cat((rel_pos_bias, extra_tokens), dim=0)
                    checkpoint_model[key] = new_rel_pos_bias
        # interpolate position embedding
        if ('pos_embed' in checkpoint_model) and (model.pos_embed is not None):
            pos_embed_checkpoint = checkpoint_model['pos_embed']
            embedding_size = pos_embed_checkpoint.shape[-1]
            num_patches = model.patch_embed.num_patches
            num_extra_tokens = model.pos_embed.shape[-2] - num_patches
            # height (== width) for the checkpoint position embedding
            orig_size = int((pos_embed_checkpoint.shape[-2] - num_extra_tokens) ** 0.5)
            # height (== width) for the new position embedding
            new_size = int(num_patches ** 0.5)
            # class_token and dist_token are kept unchanged
            if orig_size != new_size:
                print("Position interpolate from %dx%d to %dx%d" % (orig_size, orig_size, new_size, new_size))
                extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
                # only the position tokens are interpolated
                pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
                pos_tokens = pos_tokens.reshape(-1, orig_size, orig_size, embedding_size).permute(0, 3, 1, 2)
                pos_tokens = torch.nn.functional.interpolate(
                    pos_tokens, size=(new_size, new_size), mode='bicubic', align_corners=False)
                pos_tokens = pos_tokens.permute(0, 2, 3, 1).flatten(1, 2)
                new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)
                checkpoint_model['pos_embed'] = new_pos_embed

        utils.load_state_dict(model, checkpoint_model, prefix=args.diff_model_prefix)
        # model.load_state_dict(checkpoint_model, strict=False)

    return model

@torch.no_grad()
def evaluate_one_image(model, image, device, class_labels):
    if isinstance(image, str):  # is a image path
        raw_image = Image.open(image).convert('RGB')
    elif isinstance(image, Image.Image):
        raw_image = image
    elif isinstance(image, torch.Tensor):
        if len(image.shape) == 3:
            raw_image = image.unsqueeze(0)

    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),  # 调整大小
        transforms.ToTensor(),  # 转换为张量
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    image = preprocess(raw_image).unsqueeze(0).to('cuda:{}'.format(device))
    model.eval()
    with torch.no_grad():
        outputs = model(image)
    _, predicted_class_idx = torch.max(outputs, 1)
    probabilities = F.softmax(outputs, dim=1)[0]
    top_probs, top_indices = torch.topk(probabilities, k=1)
    predictions = [(class_labels[idx.item()], prob.item()) for idx, prob in zip(top_indices, top_probs)]
    #print("diff_model output: ", outputs.shape, predicted_class_idx)
    diff_model_text = ''
    for label, prob in predictions:
        diff_model_text += f"Category: {label} (Probability: {prob:.4f})\n"
    predictions = [(class_labels[1], probabilities[1].item())]
    return diff_model_text, predictions

@torch.no_grad()
def evaluate_one_cv2_frame(model, image, device, class_labels):
    # image: cv2.frame
    raw_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    raw_image = Image.fromarray(raw_image)

    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),  # 调整大小
        transforms.ToTensor(),  # 转换为张量
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    image = preprocess(raw_image).unsqueeze(0).to('cuda:{}'.format(device))
    model.eval()
    with torch.no_grad():
        outputs = model(image)
    _, predicted_class_idx = torch.max(outputs, 1)
    probabilities = F.softmax(outputs, dim=1)[0]
    top_probs, top_indices = torch.topk(probabilities, k=1)
    predictions = [(class_labels[idx.item()], prob.item()) for idx, prob in zip(top_indices, top_probs)]
    #print("diff_model output: ", outputs.shape, predicted_class_idx)
    diff_model_text = ''
    for label, prob in predictions:
        diff_model_text += f"Category: {label} (Probability: {prob:.4f})\n"
    predictions = [(class_labels[1], probabilities[1].item())]
    return diff_model_text, predictions

def parse_args(diff_model_root = None):
    parser = argparse.ArgumentParser()
    args = parser.parse_args('')
    args.diff_arch = "beit_base_patch16_224"
    args.diff_drop = 0
    args.diff_drop_path = 0.1
    args.diff_attn_drop_rate = 0
    args.diff_init_scale = 0.001
    args.diff_layer_scale_init_value = 0.1
    args.diff_input_size = 224
    args.diff_model_root = diff_model_root
    args.diff_finetune = f"{args.diff_model_root}/checkpoint-best.pth"
    args.diff_model_key = "model|module"
    args.diff_model_filter_name = ""
    args.diff_robust_test = None   
    args.diff_model_prefix = ""  
    args.diff_class_labels_path = f"{args.diff_model_root}/classes.txt"
    args.diff_top1_prob_threshold = 0.8
    args.gpu_id = 1
    
    with open(args.diff_class_labels_path, 'r') as f:
        lines = f.readlines()
        args.diff_class_labels = lines[0].rstrip('\n').split(', ')
    args.diff_nb_classes = len(args.diff_class_labels)
    return args

class cad_detector_vit: # face cropped by default
    def __init__(self, diff_model_root = '/home/zhouj0d/Science/PID26.EWS/EWS/dataset/CAD/processed_dataset/finetune_output/cad_beitv2_base_20240822_nc2'):
        self.args = parse_args(diff_model_root)
        self.model = load_model(self.args).to('cuda:{}'.format(self.args.gpu_id))
        
    def predict_single_image(self, img_path = '/home/zhouj0d/Science/PID26.EWS/EWS/DigitalShadow/test_data/people.jpg'):
        diff_model_text, predictions = evaluate_one_image(model = self.model, 
                                                          image = img_path, 
                                                          device = self.args.gpu_id, 
                                                          class_labels=self.args.diff_class_labels)
        return predictions
    
    def predict_single_cv2_frame(self, frame = None):
        diff_model_text, predictions = evaluate_one_cv2_frame(model = self.model, 
                                                          image = frame, 
                                                          device = self.args.gpu_id, 
                                                          class_labels=self.args.diff_class_labels)
        return predictions
    
if __name__ == '__main__':
    cad_detector = cad_detector_vit()
    predictions = cad_detector.predict_single_image(img_path = '/home/zhouj0d/Science/PID26.EWS/EWS/DigitalShadow/test_data/people.jpg')
    print(predictions)
    
    frame = cv2.imread('/home/zhouj0d/Science/PID26.EWS/EWS/DigitalShadow/test_data/people.jpg')
    predictions = cad_detector.predict_single_frame(frame = frame)
    print(predictions)