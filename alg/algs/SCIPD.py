# coding=utf-8
import torch
import torch.nn.functional as F
from alg.algs.ERM import ERM

from alg.modelopera import get_fea
from network import common_network
from alg.algs.base import Algorithm

from network import Adver_network
from utils.daml_util import *
from torchvision import models
import math
from utils.daml_util import (
    get_ratio_mixup_Dirichlet,
    get_sample_mixup_random,
    manual_CE,
    weighted_manual_CE
)

import sys
sys.path.append('..')
import CLIP.clip as clip
import timm


class SCIPD(Algorithm):
    def __init__(self, args):
        super(SCIPD, self).__init__(args)
        self.featurizer = get_fea(args)
        self.classifier = common_network.feat_classifier_new(
            args.num_classes, self.featurizer.in_features, args.classifier
        )
        # self.linear_project = common_network.feat_classifier(512, self.featurizer.in_features, args.classifier)
        # self.domain_classifier = common_network.feat_classifier(
        #     args.domain_num - 1, self.featurizer.in_features, args.classifier
        # )
        self.args = args
        self.device = torch.device(f'cuda:{self.args.gpu_id}' if torch.cuda.is_available() else 'cpu')
        # self.model, _ = clip.load('RN50', self.device)
        self.model, _ = clip.load('ViT-B/16', self.device)
        # self.model, _ = clip.load('ViT-L/14', self.device)
        # self.model, _ = clip.load('RN101', self.device)
        self.bz = args.batch_size
        self.num_classes = args.num_classes
        # PACS
        # known_classnames = ['dog', 'elephant', 'giraffe', 'guitar', 'horse', 'house']
        # MultiDataset
        # known_classnames = ['back_pack', 'bike', 'bike_helmet', 'bookcase', 'bottle', 'calculator', 'desk_chair', 'desk_lamp', 'desktop_computer',
        #                     'file_cabinet', 'headphones', 'keyboard', 'laptop_computer', 'letter_tray',
        #                     'mobile_phone', 'monitor', 'mouse', 'mug', 'paper_notebook', 'pen', 'phone', 'printer', 'projector', 'punchers',
        #                     'ring_binder', 'ruler', 'scissors', 'speaker', 'stapler', 'tape_dispenser', 'trash_can', 'airplane', 'bus', 'car', 'horse', 'knife',
        #                     'motorcycle', 'person', 'plant', 'skateboard', 'train', 'truck', 'bird', 'cat', 'deer', 'dog', 'monkey', 'ship']
        # OfficeHome
        known_classnames = ['Drill', 'Exit_Sign', 'Bottle', 'Glasses', 'Computer', 'File_Cabinet', 'Shelf', 'Toys', 'Sink',
               'Laptop', 'Kettle', 'Folder', 'Keyboard', 'Flipflops', 'Pencil', 'Bed', 'Hammer', 'ToothBrush', 'Couch',
               'Bike', 'Postit_Notes', 'Mug', 'Webcam', 'Desk_Lamp', 'Telephone', 'Helmet', 'Mouse', 'Pen', 'Monitor',
               'Mop', 'Sneakers', 'Notebook', 'Backpack', 'Alarm_Clock', 'Push_Pin', 'Paper_Clip', 'Batteries', 'Radio',
               'Fan', 'Ruler', 'Pan', 'Screwdriver', 'Trash_Can', 'Printer', 'Speaker', 'Eraser', 'Bucket', 'Chair',
               'Calendar', 'Calculator', 'Flowers', 'Lamp_Shade', 'Spoon', 'Candles']
        # DomainNet
        # known_classnames = ['aircraft_carrier', 'airplane', 'alarm_clock', 'ambulance', 'angel', 'animal_migration', 'ant', 'anvil',
        #  'apple', 'arm', 'asparagus', 'axe', 'backpack', 'banana', 'bandage', 'barn', 'baseball', 'baseball_bat',
        #  'basket', 'basketball', 'bat', 'bathtub', 'beach', 'bear', 'beard', 'bed', 'bee', 'belt', 'bench', 'bicycle',
        #  'binoculars', 'bird', 'birthday_cake', 'blackberry', 'blueberry', 'book', 'boomerang', 'bottlecap', 'bowtie',
        #  'bracelet', 'brain', 'bread', 'bridge', 'broccoli', 'broom', 'bucket', 'bulldozer', 'bus', 'bush', 'butterfly',
        #  'cactus', 'cake', 'calculator', 'calendar', 'camel', 'camera', 'camouflage', 'campfire', 'candle', 'cannon',
        #  'canoe', 'car', 'carrot', 'castle', 'cat', 'ceiling_fan', 'cello', 'cell_phone', 'chair', 'chandelier',
        #  'church', 'circle', 'clarinet', 'clock', 'cloud', 'coffee_cup', 'compass', 'computer', 'cookie', 'cooler',
        #  'couch', 'cow', 'crab', 'crayon', 'crocodile', 'crown', 'cruise_ship', 'cup', 'diamond', 'dishwasher',
        #  'diving_board', 'dog', 'dolphin', 'donut', 'door', 'dragon', 'dresser', 'drill', 'drums', 'duck', 'dumbbell',
        #  'ear', 'elbow', 'elephant', 'envelope', 'eraser', 'eye', 'eyeglasses', 'face', 'fan', 'feather', 'fence',
        #  'finger', 'fire_hydrant', 'fireplace', 'firetruck', 'fish', 'flamingo', 'flashlight', 'flip_flops',
        #  'floor_lamp', 'flower', 'flying_saucer', 'foot', 'fork', 'frog', 'frying_pan', 'garden', 'garden_hose',
        #  'giraffe', 'goatee', 'golf_club', 'grapes', 'grass', 'guitar', 'hamburger', 'hammer', 'hand', 'harp', 'hat',
        #  'headphones', 'hedgehog', 'helicopter', 'helmet', 'hexagon', 'hockey_puck', 'hockey_stick', 'horse',
        #  'hospital', 'hot_air_balloon', 'hot_dog', 'hot_tub', 'hourglass', 'house', 'house_plant', 'hurricane',
        #  'ice_cream', 'jacket', 'jail', 'kangaroo', 'key', 'keyboard', 'knee', 'knife', 'ladder', 'lantern', 'laptop',
        #  'leaf', 'leg', 'light_bulb', 'lighter', 'lighthouse', 'lightning', 'line', 'lion', 'lipstick', 'lobster',
        #  'lollipop', 'mailbox', 'map', 'marker', 'matches', 'megaphone', 'mermaid', 'microphone', 'microwave', 'monkey',
        #  'moon', 'mosquito', 'motorbike', 'mountain', 'mouse', 'moustache', 'mouth', 'mug', 'mushroom', 'nail',
        #  'necklace', 'nose', 'ocean', 'octagon', 'octopus', 'onion', 'oven', 'owl', 'paintbrush', 'paint_can',
        #  'palm_tree', 'panda', 'pants', 'paper_clip', 'parachute', 'parrot', 'passport', 'peanut', 'pear', 'peas',
        #  'pencil', 'penguin', 'piano', 'pickup_truck', 'picture_frame', 'pig', 'pillow', 'pineapple', 'pizza', 'pliers',
        #  'police_car', 'pond', 'pool', 'popsicle', 'postcard', 'potato', 'power_outlet', 'purse', 'rabbit', 'raccoon',
        #  'radio', 'rain', 'rainbow', 'rake', 'remote_control', 'rhinoceros', 'rifle', 'river', 'roller_coaster',
        #  'rollerskates', 'sailboat', 'sandwich', 'saw', 'saxophone', 'school_bus', 'scissors', 'scorpion',
        #  'screwdriver', 'sea_turtle', 'see_saw', 'shark', 'sheep', 'shoe', 'shorts', 'shovel', 'sink', 'skateboard',
        #  'skull', 'skyscraper', 'sleeping_bag', 'smiley_face', 'snail', 'snake', 'snorkel', 'snowflake', 'snowman',
        #  'soccer_ball', 'sock', 'speedboat', 'spider', 'spoon', 'spreadsheet', 'square', 'squiggle', 'squirrel',
        #  'stairs', 'star', 'steak', 'stereo', 'stethoscope', 'stitches', 'stop_sign', 'stove', 'strawberry',
        #  'streetlight', 'string_bean', 'submarine', 'suitcase', 'sun', 'swan', 'sweater', 'swing_set', 'sword',
        #  'syringe', 'table', 'teapot', 'teddy-bear', 'telephone', 'television', 'tennis_racquet', 'tent',
        #  'The_Eiffel_Tower', 'The_Great_Wall_of_China', 'The_Mona_Lisa', 'tiger', 'toaster', 'toe', 'toilet', 'tooth',
        #  'toothbrush', 'toothpaste', 'tornado', 'tractor', 'traffic_light', 'train', 'tree', 'triangle', 'trombone',
        #  'truck', 'trumpet', 't-shirt', 'umbrella', 'underwear', 'van', 'vase', 'violin', 'washing_machine',
        #  'watermelon', 'waterslide', 'whale', 'wheel', 'windmill', 'wine_bottle', 'wine_glass', 'wristwatch', 'yoga',
        #  'zebra', 'zigzag']

        # image_input = self.preprocess(minibatches).unsqueeze(0).to(self.device)
        text_inputs = torch.cat([clip.tokenize(f"a photo of a {c}") for c in known_classnames]).to(self.device)
        with torch.no_grad():
            text_features = self.model.encode_text(text_inputs)
            text_features /= text_features.norm(dim=-1, keepdim=True)
        classifier_weight = torch.tensor(text_features, dtype=torch.float32, requires_grad=True)
        self.classifier.fc.weight.data.copy_(classifier_weight)
        self.register_buffer("proto_feats", torch.zeros((args.num_classes, self.featurizer.in_features), requires_grad=False))

    def torch_cosine_similarity(self, features1, features2):
        norm1 = torch.norm(features1, dim=-1).reshape(features1.shape[0], 1)
        norm2 = torch.norm(features2, dim=-1).reshape(1, features2.shape[0])
        end_norm = torch.mm(norm1.to(torch.double), norm2.to(torch.double))
        cos = torch.mm(features1.to(torch.double), features2.T.to(torch.double)) / end_norm
        return cos

    def compute_similarity(self, features, class_features):
        features_norm = torch.nn.functional.normalize(features, dim=1)  # 归一化样本特征
        class_features_norm = torch.nn.functional.normalize(class_features, dim=2)  # 归一化类特征
        similarity_matrix = torch.bmm(features_norm.unsqueeze(1), class_features_norm.permute(0, 2, 1)).squeeze(1)
        return similarity_matrix

    def update(self, minibatches, opt, sch):

        # data preparation
        all_x = torch.cat([data[0].to(self.device).float() for data in minibatches])
        all_y = torch.cat([data[1].to(self.device).long() for data in minibatches])
        all_domain = torch.cat([data[2].to(self.device).long() for data in minibatches])
        thresh = self.args.test_envs[0]
        all_domain = torch.tensor([x.item() - 1 if x > thresh else x.item() for x in all_domain]).to(self.device)
        # original
        # opt.zero_grad()
        # optimizer set
        # sch[0].step()
        # sch[1].step()
        # sch[2].step()

        net_opt = opt[0]
        cls_opt = opt[1]
        # dcls_opt = opt[2]

        net_opt.zero_grad()
        cls_opt.zero_grad()
        # dcls_opt.zero_grad()

        # CLIP initialization
        # known_classnames = ['back_pack', 'bike', 'bike_helmet', 'bookcase', 'bottle', 'calculator', 'desk_chair', 'desk_lamp', 'desktop_computer',
        #                     'file_cabinet', 'headphones', 'keyboard', 'laptop_computer', 'letter_tray',
        #                     'mobile_phone', 'monitor', 'mouse', 'mug', 'paper_notebook', 'pen', 'phone', 'printer', 'projector', 'punchers',
        #                     'ring_binder', 'ruler', 'scissors', 'speaker', 'stapler', 'tape_dispenser', 'trash_can', 'airplane', 'bus', 'car', 'horse', 'knife',
        #                     'motorcycle', 'person', 'plant', 'skateboard', 'train', 'truck', 'bird', 'cat', 'deer', 'dog', 'monkey', 'ship']

        # known_classnames = ['dog', 'elephant', 'giraffe', 'guitar', 'horse', 'house']
        # known_classnames = ['aircraft_carrier', 'airplane', 'alarm_clock', 'ambulance', 'angel', 'animal_migration', 'ant', 'anvil',
        #  'apple', 'arm', 'asparagus', 'axe', 'backpack', 'banana', 'bandage', 'barn', 'baseball', 'baseball_bat',
        #  'basket', 'basketball', 'bat', 'bathtub', 'beach', 'bear', 'beard', 'bed', 'bee', 'belt', 'bench', 'bicycle',
        #  'binoculars', 'bird', 'birthday_cake', 'blackberry', 'blueberry', 'book', 'boomerang', 'bottlecap', 'bowtie',
        #  'bracelet', 'brain', 'bread', 'bridge', 'broccoli', 'broom', 'bucket', 'bulldozer', 'bus', 'bush', 'butterfly',
        #  'cactus', 'cake', 'calculator', 'calendar', 'camel', 'camera', 'camouflage', 'campfire', 'candle', 'cannon',
        #  'canoe', 'car', 'carrot', 'castle', 'cat', 'ceiling_fan', 'cello', 'cell_phone', 'chair', 'chandelier',
        #  'church', 'circle', 'clarinet', 'clock', 'cloud', 'coffee_cup', 'compass', 'computer', 'cookie', 'cooler',
        #  'couch', 'cow', 'crab', 'crayon', 'crocodile', 'crown', 'cruise_ship', 'cup', 'diamond', 'dishwasher',
        #  'diving_board', 'dog', 'dolphin', 'donut', 'door', 'dragon', 'dresser', 'drill', 'drums', 'duck', 'dumbbell',
        #  'ear', 'elbow', 'elephant', 'envelope', 'eraser', 'eye', 'eyeglasses', 'face', 'fan', 'feather', 'fence',
        #  'finger', 'fire_hydrant', 'fireplace', 'firetruck', 'fish', 'flamingo', 'flashlight', 'flip_flops',
        #  'floor_lamp', 'flower', 'flying_saucer', 'foot', 'fork', 'frog', 'frying_pan', 'garden', 'garden_hose',
        #  'giraffe', 'goatee', 'golf_club', 'grapes', 'grass', 'guitar', 'hamburger', 'hammer', 'hand', 'harp', 'hat',
        #  'headphones', 'hedgehog', 'helicopter', 'helmet', 'hexagon', 'hockey_puck', 'hockey_stick', 'horse',
        #  'hospital', 'hot_air_balloon', 'hot_dog', 'hot_tub', 'hourglass', 'house', 'house_plant', 'hurricane',
        #  'ice_cream', 'jacket', 'jail', 'kangaroo', 'key', 'keyboard', 'knee', 'knife', 'ladder', 'lantern', 'laptop',
        #  'leaf', 'leg', 'light_bulb', 'lighter', 'lighthouse', 'lightning', 'line', 'lion', 'lipstick', 'lobster',
        #  'lollipop', 'mailbox', 'map', 'marker', 'matches', 'megaphone', 'mermaid', 'microphone', 'microwave', 'monkey',
        #  'moon', 'mosquito', 'motorbike', 'mountain', 'mouse', 'moustache', 'mouth', 'mug', 'mushroom', 'nail',
        #  'necklace', 'nose', 'ocean', 'octagon', 'octopus', 'onion', 'oven', 'owl', 'paintbrush', 'paint_can',
        #  'palm_tree', 'panda', 'pants', 'paper_clip', 'parachute', 'parrot', 'passport', 'peanut', 'pear', 'peas',
        #  'pencil', 'penguin', 'piano', 'pickup_truck', 'picture_frame', 'pig', 'pillow', 'pineapple', 'pizza', 'pliers',
        #  'police_car', 'pond', 'pool', 'popsicle', 'postcard', 'potato', 'power_outlet', 'purse', 'rabbit', 'raccoon',
        #  'radio', 'rain', 'rainbow', 'rake', 'remote_control', 'rhinoceros', 'rifle', 'river', 'roller_coaster',
        #  'rollerskates', 'sailboat', 'sandwich', 'saw', 'saxophone', 'school_bus', 'scissors', 'scorpion',
        #  'screwdriver', 'sea_turtle', 'see_saw', 'shark', 'sheep', 'shoe', 'shorts', 'shovel', 'sink', 'skateboard',
        #  'skull', 'skyscraper', 'sleeping_bag', 'smiley_face', 'snail', 'snake', 'snorkel', 'snowflake', 'snowman',
        #  'soccer_ball', 'sock', 'speedboat', 'spider', 'spoon', 'spreadsheet', 'square', 'squiggle', 'squirrel',
        #  'stairs', 'star', 'steak', 'stereo', 'stethoscope', 'stitches', 'stop_sign', 'stove', 'strawberry',
        #  'streetlight', 'string_bean', 'submarine', 'suitcase', 'sun', 'swan', 'sweater', 'swing_set', 'sword',
        #  'syringe', 'table', 'teapot', 'teddy-bear', 'telephone', 'television', 'tennis_racquet', 'tent',
        #  'The_Eiffel_Tower', 'The_Great_Wall_of_China', 'The_Mona_Lisa', 'tiger', 'toaster', 'toe', 'toilet', 'tooth',
        #  'toothbrush', 'toothpaste', 'tornado', 'tractor', 'traffic_light', 'train', 'tree', 'triangle', 'trombone',
        #  'truck', 'trumpet', 't-shirt', 'umbrella', 'underwear', 'van', 'vase', 'violin', 'washing_machine',
        #  'watermelon', 'waterslide', 'whale', 'wheel', 'windmill', 'wine_bottle', 'wine_glass', 'wristwatch', 'yoga',
        #  'zebra', 'zigzag']
        known_classnames = ['Drill', 'Exit_Sign', 'Bottle', 'Glasses', 'Computer', 'File_Cabinet', 'Shelf', 'Toys', 'Sink',
               'Laptop', 'Kettle', 'Folder', 'Keyboard', 'Flipflops', 'Pencil', 'Bed', 'Hammer', 'ToothBrush', 'Couch',
               'Bike', 'Postit_Notes', 'Mug', 'Webcam', 'Desk_Lamp', 'Telephone', 'Helmet', 'Mouse', 'Pen', 'Monitor',
               'Mop', 'Sneakers', 'Notebook', 'Backpack', 'Alarm_Clock', 'Push_Pin', 'Paper_Clip', 'Batteries', 'Radio',
               'Fan', 'Ruler', 'Pan', 'Screwdriver', 'Trash_Can', 'Printer', 'Speaker', 'Eraser', 'Bucket', 'Chair',
               'Calendar', 'Calculator', 'Flowers', 'Lamp_Shade', 'Spoon', 'Candles']
        # image_input = self.preprocess(minibatches).unsqueeze(0).to(self.device)
        text_inputs = torch.cat([clip.tokenize(f"a photo of a {c}") for c in known_classnames]).to(self.device)

        # # Calculate features and generate CLIP logits for the image
        # with torch.no_grad():
        #     image_features, x1, x2, x3, x4 = self.model.encode_image(all_x)
        #     text_features = self.model.encode_text(text_inputs)
        #     image_features /= image_features.norm(dim=-1, keepdim=True)
        #     text_features /= text_features.norm(dim=-1, keepdim=True)
        #     similarity = (100 * image_features @ text_features.T).softmax(dim=-1)
        #     print(x1.shape, x2.shape, x3.shape, x4.shape)
        #     similarity = torch.tensor(similarity, dtype=torch.float32, requires_grad=False)
        #
        #     x1 /= x1.norm(dim=-1, keepdim=True)
        #     similarity_x1 = (100 * x1 @ text_features.T).softmax(dim=-1)
        #     similarity_x1 = torch.tensor(similarity_x1, dtype=torch.float32, requires_grad=False)
        #
        #     x2 /= x2.norm(dim=-1, keepdim=True)
        #     similarity_x2 = (100 * x2 @ text_features.T).softmax(dim=-1)
        #     similarity_x2 = torch.tensor(similarity_x2, dtype=torch.float32, requires_grad=False)
        #
        #     x3 /= x1.norm(dim=-1, keepdim=True)
        #     similarity_x3 = (100 * x4 @ text_features.T).softmax(dim=-1)
        #     similarity_x3 = torch.tensor(similarity_x3, dtype=torch.float32, requires_grad=False)
        #
        #     x4 /= x1.norm(dim=-1, keepdim=True)
        #     similarity_x4 = (100 * x4 @ text_features.T).softmax(dim=-1)
        #     similarity_x4 = torch.tensor(similarity_x4, dtype=torch.float32, requires_grad=False)
        #     print(similarity, similarity_x1, similarity_x2, similarity_x3, similarity_x4)
        #     assert 1==2

        # Calculate features and generate CLIP logits for the image
        with torch.no_grad():
            image_features = self.model.encode_image(all_x)
            text_features = self.model.encode_text(text_inputs)
            text_sim = self.torch_cosine_similarity(text_features, text_features)
            # soft_text_sim0.1 = torch.softmax(text_sim / 0.1, dim=1)
            soft_text_sim = torch.softmax(text_sim, dim=1)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            text_features /= text_features.norm(dim=-1, keepdim=True)
            similarity = (100 * image_features @ text_features.T).softmax(dim=-1)
            similarity = torch.tensor(similarity, dtype=torch.float32, requires_grad=False)


        # test on mistakes
        # predict_label = torch.argmax(similarity, dim=1)
        # num1 = torch.nonzero(predict_label[0:32] != all_y[0:32])
        # num2 = torch.nonzero(predict_label[32:64] != all_y[32:64])
        # num3 = torch.nonzero(predict_label[64:96] != all_y[64:96])



        # # Base
        # features = self.featurizer(all_x)
        # class_logits = self.classifier(features)
        #
        # class_loss = F.cross_entropy(class_logits, all_y)
        #
        # # class_loss = F.cross_entropy(class_logits, all_y, reduction='none')
        # # class_loss = class_loss * weight_simple
        # # class_loss = class_loss.mean()
        # cls_logits = class_logits.softmax(dim=-1)
        #
        # # kl_loss = F.kl_div(cls_logits.log(), similarity, reduction='none')
        # # kl_loss = torch.mul(kl_loss, weight_ori.unsqueeze(1))
        # # kl_loss = kl_loss.sum() / len(features)
        # kl_loss = F.kl_div(cls_logits.log(), similarity, reduction='batchmean')
        # loss = class_loss + kl_loss

        # # Base
        # features = self.featurizer(all_x)
        # class_logits = self.classifier(features)
        # class_loss = manual_CE(class_logits, similarity)
        # loss = class_loss

        # # complement wrong samples
        # predict_label = torch.argmax(similarity, dim=1)
        # wrong_index = torch.nonzero(predict_label != all_y).squeeze()
        # mask = torch.zeros(all_y.size(0)).to(self.device)
        # mask[wrong_index] = 1
        # fixed_loss = F.cross_entropy(class_logits, all_y, reduction='none')
        # fixed_loss = mask * fixed_loss
        # fixed_loss = fixed_loss.mean()
        # loss += fixed_loss


        # # feature perturbation
        # max_sim_index = torch.argsort(weight_ori, dim=0, descending=True)
        # # ratio = int(len(max_sim_index) * 0.2)
        # # need_perturb_index = max_sim_index[0:ratio]
        # # epsilon = 1
        # # features[need_perturb_index] = features[need_perturb_index] + epsilon * image_features[need_perturb_index]
        #
        #


        # sample selection standard
        weight_ori = torch.max(similarity, dim=1)[0]
        weight_inverse = 1 / torch.pow(weight_ori, self.args.hyper1)
        # weight_square_inverse = 1 / torch.pow(weight_ori, 2)

        # feature output
        features = self.featurizer(all_x)
        # features = self.featurizer(all_x, ret_feats=False)

        # # fix wrong samples
        class_logits = self.classifier(features)

        predict_label = torch.argmax(similarity, dim=1)
        wrong_index = torch.nonzero(predict_label != all_y)
        # print(torch.softmax(class_logits[wrong_index], dim=-1))
        # print(similarity[wrong_index])
        if len(wrong_index) == 1:
            wrong_index = wrong_index.squeeze(1)
            wrong_onehot = F.one_hot(all_y[wrong_index], num_classes=self.args.num_classes).float()
            wrong_similarity = similarity[wrong_index]
            max_similarity, _ = wrong_similarity.max(dim=1, keepdim=False)
            similarity[wrong_index] = F.softmax((wrong_onehot * max_similarity.unsqueeze(1) + wrong_similarity) / self.args.hyper2,
                                                dim=1)
        elif len(wrong_index) > 1:
            wrong_index = wrong_index.squeeze()
            wrong_onehot = F.one_hot(all_y[wrong_index], num_classes=self.args.num_classes).float()
            wrong_similarity = similarity[wrong_index]
            max_similarity, _ = wrong_similarity.max(dim=1, keepdim=False)
            similarity[wrong_index] = F.softmax((wrong_onehot * max_similarity.unsqueeze(1) + wrong_similarity) / self.args.hyper2, dim=1)
        # print(similarity[wrong_index])
        # assert 1==2
        # overfitting_index = torch.nonzero(similarity.max(1, keepdim=False)[0] > 0.8).squeeze()
        # print(similarity[overfitting_index].max(1)[0], similarity[overfitting_index].max(1)[1])
        # similarity[overfitting_index] += soft_text_sim[all_y[overfitting_index]]
        # similarity[overfitting_index] /= 2
        # print(similarity[overfitting_index].max(1)[0], similarity[overfitting_index].max(1)[1])

        # class_loss = manual_CE(class_logits, similarity)
        class_loss = weighted_manual_CE(class_logits, similarity, weight_inverse)
        loss = class_loss
        #
        #
        #
        #
        # text2cls loss
        # 1) current
        text_features_norm = (text_features / text_features.norm(p=2, dim=-1, keepdim=True)).float()
        cls_weight = self.classifier.fc.weight
        cls_weight_norm = cls_weight / cls_weight.norm(p=2, dim=-1, keepdim=True)
        # textcls_sim = torch.matmul(text_features_norm, cls_weight_norm.t())
        # # textcls_sim = torch.matmul(cls_weight_norm, text_features_norm.t())
        # # textcls_sim = self.torch_cosine_similarity(text_features_norm, cls_weight_norm)
        # textcls_sim = torch.softmax(textcls_sim, dim=1)
        # # print(textcls_sim)
        # print(textcls_sim.t().sum(dim=1))
        # print(textcls_sim.t().sum(dim=0))
        # # print(soft_text_sim)
        # assert 1==2
        # # assert 1==2
        # contrastive_loss = (manual_CE(textcls_sim, soft_text_sim.to(self.device)) + \
        #                     manual_CE(textcls_sim.t(), soft_text_sim.to(self.device).t())) / 2

        textcls_sim = torch.matmul(text_features_norm, cls_weight_norm.t())
        contrastive_loss = (manual_CE(torch.softmax(textcls_sim, dim=1), torch.softmax(text_sim.to(self.device), dim=1)) + \
                            manual_CE(torch.softmax(textcls_sim.t(), dim=1), torch.softmax(text_sim.to(self.device).t(), dim=1))) / 2
        # soft
        # contrastive_loss = (F.cross_entropy(textcls_sim, torch.arange(len(textcls_sim)).to(self.device)) + \
        #                    F.cross_entropy(textcls_sim.t(), torch.arange(len(textcls_sim)).to(self.device))) / 2
        loss += self.args.hyper3 * contrastive_loss

        # # 2) original
        # text_features_norm = (text_features / text_features.norm(p=2, dim=-1, keepdim=True)).float()
        # cls_weight = self.classifier.fc.weight
        # cls_weight_norm = cls_weight / cls_weight.norm(p=2, dim=-1, keepdim=True)
        # textcls_sim = torch.matmul(text_features_norm, cls_weight_norm.t())
        # textcls_sim = torch.softmax(textcls_sim, dim=1)
        # contrastive_loss = (manual_CE(textcls_sim, soft_text_sim.to(self.device)) + \
        #                     manual_CE(textcls_sim.t(), soft_text_sim.to(self.device))) / 2
        # loss += self.args.hyper3 * contrastive_loss


        # image_features = image_features.float()
        # fd_loss = torch.norm((features - image_features), p=2) / len(image_features)
        # loss += 0.2 * fd_loss

        # # 2) fixed
        # cls_weight = self.classifier.fc.weight
        # cls_weight_norm = cls_weight / cls_weight.norm(p=2, dim=-1, keepdim=True)
        # cls_sim = torch.matmul(cls_weight_norm, cls_weight_norm.t())
        # cls_sim = torch.softmax(cls_sim, dim=1)
        # contrastive_loss = (manual_CE(cls_sim, soft_text_sim.to(self.device)) + \
        #                     manual_CE(cls_sim.t(), soft_text_sim.to(self.device))) / 2
        # loss += 0.1 * contrastive_loss

        # print(0, cls_sim)
        # print(1, textcls_sim)
        # print(2, soft_text_sim)
        # print(3, soft_text_sim1)


        # # CLIP-KD
        # # 1) FD
        # # clip_features = self.linear_project(image_features)
        # image_features = image_features.float()
        # fd_loss = torch.norm((features - image_features), p=2) / len(image_features)
        # loss += 0.2 * fd_loss
        #
        # # 2) CRD
        # image_features_norm = (image_features / image_features.norm(p=2, dim=-1, keepdim=True)).float()
        # features_norm = features / features.norm(p=2, dim=-1, keepdim=True)
        # text_features_norm = (text_features / text_features.norm(p=2, dim=-1, keepdim=True)).float()
        # teacher_sim = torch.matmul(text_features_norm, image_features_norm.t())
        # student_sim = torch.matmul(text_features_norm, features_norm.t())
        # teacher_sim, student_sim = torch.softmax(teacher_sim / 0.03, dim=1), torch.softmax(student_sim / 0.03, dim=1)
        # crd_loss = F.kl_div(student_sim.log(), teacher_sim, reduction='batchmean')
        # loss += crd_loss



        # # ent loss
        # clip_entropy = -torch.sum(similarity * torch.log2(similarity), dim=1)
        # dg_entropy = -torch.sum(cls_logits * torch.log2(cls_logits), dim=1)
        # mask = torch.zeros_like(clip_entropy)
        # mask[clip_entropy > (math.log(self.args.num_classes, 2) + 0.5)] = 1
        # mask[clip_entropy < (math.log(self.args.num_classes, 2) - 0.5)] = 1
        # count = torch.sum(mask == 1)
        # ent_loss = dg_entropy * mask
        # ent_loss = ent_loss.sum() / count
        # loss += ent_loss



        # # feature alignment
        # feats_align_loss = torch.dist(features, image_features)
        # loss += feats_align_loss
        # print(feats_align_loss)

        # # Weighted-DANN
        # # 1) max
        # reverse_features = Adver_network.ReverseLayerF.apply(features, self.args.alpha)
        # domain_logits = self.domain_classifier(reverse_features)
        # domain_loss = F.cross_entropy(domain_logits, all_domain, reduction='none')
        # domain_loss = domain_loss * weight_inverse
        # domain_loss = domain_loss.mean()
        # loss += 0.2 * domain_loss

        # 2) max + entropy
        # entropies = -torch.sum(similarity * torch.log2(similarity), dim=1) / math.log(self.args.num_classes, 2)
        # maxs = torch.max(similarity, dim=1)[0]
        # weight = ((1 - entropies) + maxs) / 2
        # weight[torch.isnan(weight)] = 1e-5
        # mask = torch.ones_like(weight)
        # mask[weight >= 0.5] = 0
        # mask[weight <= 0.1] = 0
        # mask[torch.isnan(mask)] = 0
        # weight = 1 / weight
        # reverse_features = Adver_network.ReverseLayerF.apply(features, self.args.alpha)
        # domain_logits = self.domain_classifier(reverse_features)
        # domain_loss = F.cross_entropy(domain_logits, all_domain, reduction='none')
        # domain_loss[torch.isnan(domain_loss)] = 0
        # domain_loss = domain_loss * weight * mask
        # domain_loss = domain_loss.mean()
        # loss += 0.2 * domain_loss


        # # XDED
        # xded_loss = 0.0
        # cls_logits = torch.softmax(class_logits / 4, dim=-1)
        # for i in range(self.args.num_classes):
        #     indices = torch.nonzero(all_y == i)
        #     if len(indices) == 0:
        #         continue
        #     else:
        #         indices = indices.squeeze(dim=1)
        #     proto_logits = class_logits[indices].mean(dim=0)
        #     prob_proto = F.softmax(proto_logits / 4, dim=0)
        #     for j in range(len(indices)):
        #         prob_cls = cls_logits[indices[j]]
        #         xded_loss += F.kl_div(torch.log(prob_cls), prob_proto, reduction='batchmean')
        # loss += 5 * xded_loss

        # # Dirichlet Mixup
        # one_hot_targets = [
        #     F.one_hot(y, num_classes=self.args.num_classes) for y in all_y
        # ]
        # # .detach().cpu().numpy()
        # mixup_dir_list = [1, 1, 1]
        # mix_indeces = [
        #     get_sample_mixup_random(minibatch[2]) for minibatch in minibatches
        # ]  # 3 * N
        #
        # mixup_ratios = get_ratio_mixup_Dirichlet(minibatches[0][2], mixup_dir_list).to(
        #     self.device
        # )  # N * 3
        # mixup_ratios = mixup_ratios.permute(1, 0).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)  # 3 * N * 1
        # new_images = [data[0].to(self.device).float() for data in minibatches]
        # mix_images = [
        #     x[mix_index]
        #     for x, mix_index in zip(new_images, mix_indeces)
        # ]  # 3 * 32 * 3 * 224 * 224
        # mixup_images = torch.stack(mix_images, dim=0)  # 3 * N * 512
        # one_hot_targets_list = [one_hot_targets[i:i + 32] for i in range(0, 96, 32)]
        # new_one_hot_targets = []
        # for one_hot_target, mix_index in zip(one_hot_targets_list, mix_indeces):
        #     one_hot_target = torch.stack(one_hot_target)
        #     new_one_hot_targets.append(
        #         one_hot_target[mix_index]
        #     )  # 3 * N * C
        # mixup_one_hot_targets = torch.stack(new_one_hot_targets)  # 3 * N * C
        # mixup_images = torch.sum((mixup_images * mixup_ratios), dim=0)
        # mixup_one_hot_targets = torch.sum((mixup_one_hot_targets * mixup_ratios), dim=0)
        #
        #
        # with torch.no_grad():
        #     image_features = self.model.encode_image(mixup_images)
        #     text_features = self.model.encode_text(text_inputs)
        #     image_features /= image_features.norm(dim=-1, keepdim=True)
        #     text_features /= text_features.norm(dim=-1, keepdim=True)
        #     similarity = (100 * image_features @ text_features.T).softmax(dim=-1)
        #     similarity = torch.tensor(similarity, dtype=torch.float32, requires_grad=False)
        #
        # # Continue
        # features = self.featurizer(mixup_images)
        # class_logits = self.classifier(features)
        # dclass_loss = manual_CE(class_logits, mixup_one_hot_targets)
        # cls_logits = class_logits.softmax(dim=-1)
        # dkl_loss = F.kl_div(cls_logits.log(), similarity, reduction='batchmean')
        #
        # loss = class_loss + kl_loss + dclass_loss + dkl_loss

        # ### Dir loss Start ###
        #
        # one_hot_targets = [
        #     F.one_hot(y, num_classes=self.args.num_classes) for y in all_y
        # ]
        # mixup_dir_list = [1, 1, 1]
        # mix_indeces = [
        #     get_sample_mixup_random(minibatch[2]) for minibatch in minibatches
        # ]  # 3 * N
        # mixup_ratios = get_ratio_mixup_Dirichlet(minibatches[0][2], mixup_dir_list).to(
        #     self.device
        # )  # N * 3
        # mixup_ratios = mixup_ratios.permute(1, 0).unsqueeze(-1)  # 3 * N * 1
        # features_list = [features[i:i + 32] for i in range(0, 96, 32)]
        # new_features = [
        #     feature[mix_index].detach()
        #     for feature, mix_index in zip(features_list, mix_indeces)
        # ]  # 3 * N * 512
        # mixup_features = torch.stack(new_features)  # 3 * N * 512
        #
        # one_hot_targets_list = [one_hot_targets[i:i + 32] for i in range(0, 96, 32)]
        # new_one_hot_targets = []
        # for one_hot_target, mix_index in zip(one_hot_targets_list, mix_indeces):
        #     one_hot_target = torch.stack(one_hot_target)
        #     new_one_hot_targets.append(
        #         one_hot_target[mix_index]
        #     )  # 3 * N * C
        # mixup_one_hot_targets = torch.stack(new_one_hot_targets)  # 3 * N * C
        # # new_one_hot_targets = [
        # #     one_hot_target[mix_index]
        # #     for one_hot_target, mix_index in zip(one_hot_targets, mix_indeces)
        # # ]  # 3 * N * C
        # # mixup_one_hot_targets = torch.stack(new_one_hot_targets)  # 3 * N * C
        # mixup_features = torch.sum((mixup_features * mixup_ratios), dim=0)
        # mixup_one_hot_targets = torch.sum((mixup_one_hot_targets * mixup_ratios), dim=0)
        # mixup_predictions = self.classifier(mixup_features)
        # weight_list = [weight[i:i + 32] for i in range(0, 96, 32)]
        # weights = torch.stack(weight_list)
        # new_weight_list = []
        # for i in range(3):
        #     new_weight_list.append(weights[i][mix_indeces[i]])
        # weight_mix = torch.sum(torch.stack(new_weight_list), dim=0)  # 3 * N * C
        # dirmixup_loss = weighted_manual_CE(mixup_predictions, mixup_one_hot_targets, weight_mix)
        # loss += dirmixup_loss
        # ### Dir loss End ###

        # # Coral
        # objective = 0
        # penalty = 0
        # nmb = len(minibatches)  # num of domain
        # features = [
        #     self.featurizer(data[0].to(self.device).float()) for data in minibatches
        # ]  # features[0].shape (N, 512)
        # classifs = [self.classifier(fi) for fi in features]  # classifs[0].shape (N, C)
        # targets = [
        #     data[1].to(self.device).long() for data in minibatches
        # ]  # targets[0].shape (C)
        #
        # for i in range(nmb):
        #     objective += F.cross_entropy(classifs[i], targets[i])
        #     for j in range(i + 1, nmb):
        #         penalty += self.coral(features[i], features[j])
        #
        # objective /= nmb
        # if nmb > 1:
        #     penalty /= nmb * (nmb - 1) / 2
        # # Base
        # class_logits = torch.cat([classlogits for classlogits in classifs], dim=0)
        # cls_logits = class_logits.softmax(dim=-1)
        # kl_loss = F.kl_div(cls_logits.log(), similarity, reduction='batchmean')
        # loss = objective + 0.2 * kl_loss + self.args.mmd_gamma * penalty
        # if torch.is_tensor(penalty):
        #     penalty = penalty.item()

        loss.backward()
        # opt.step()

        net_opt.step()
        cls_opt.step()
        # dcls_opt.step()

        if sch:
            sch[0].step()
            sch[1].step()
            # sch[2].step()
        # for o in opt:
        #     for params in o.param_groups:
        #         print(params["lr"])
        #
        # if sch:
        #     sch.step()
        # for params in opt.param_groups:
        #     print(params["lr"])
        # if sch:
        #     sch.step()
        # if isinstance(opt, list):
        # return {"class": class_loss.item(),
        #         "kl": kl_loss.item(),
        #         "aug_class": dclass_loss.item(),
        #         "aug_kl": dkl_loss.item()}
        # return {
        #     "class": objective.item(),
        #     "kl": kl_loss.item(),
        #     "coral": penalty,
        #     "total": loss.item(),
        # }
        return {"class": class_loss.item(),
                "contra": contrastive_loss.item(),
                # "fixed" : fixed_loss.item(),
                # "xded": xded_loss.item(),
                # "fd" : fd_loss.item(),
                # "crd": crd_loss.item(),
                # "kl": kl_loss.item(),
                # "domain": domain_loss.item(),
                # "ent": ent_loss.item()
                # "align": feats_align_loss.item()
                }


    def predict(self, x):
        features = self.featurizer(x)
        output = self.classifier(features)
        return output

    def base(self, x):
        # known_classnames = ['dog', 'elephant', 'giraffe', 'guitar', 'horse', 'house']
        known_classnames = ['Drill', 'Exit_Sign', 'Bottle', 'Glasses', 'Computer', 'File_Cabinet', 'Shelf', 'Toys', 'Sink',
               'Laptop', 'Kettle', 'Folder', 'Keyboard', 'Flipflops', 'Pencil', 'Bed', 'Hammer', 'ToothBrush', 'Couch',
               'Bike', 'Postit_Notes', 'Mug', 'Webcam', 'Desk_Lamp', 'Telephone', 'Helmet', 'Mouse', 'Pen', 'Monitor',
               'Mop', 'Sneakers', 'Notebook', 'Backpack', 'Alarm_Clock', 'Push_Pin', 'Paper_Clip', 'Batteries', 'Radio',
               'Fan', 'Ruler', 'Pan', 'Screwdriver', 'Trash_Can', 'Printer', 'Speaker', 'Eraser', 'Bucket', 'Chair',
               'Calendar', 'Calculator', 'Flowers', 'Lamp_Shade', 'Spoon', 'Candles']
        text_inputs = torch.cat([clip.tokenize(f"a photo of a {c}") for c in known_classnames]).to(self.device)
        image_features = self.model.encode_image(x)
        text_features = self.model.encode_text(text_inputs)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        similarity = (100 * image_features @ text_features.T).softmax(dim=-1)
        return similarity


    def domain_ability_test(self, x):
        # known_classnames = ['dog', 'elephant', 'giraffe', 'guitar', 'horse', 'house']
        known_domainnames = ['Art', 'Clipart', 'Product', 'Real_World']
        text_inputs = torch.cat([clip.tokenize(f"a photo from {c}") for c in known_domainnames]).to(self.device)
        image_features = self.model.encode_image(x)
        text_features = self.model.encode_text(text_inputs)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        similarity = (100 * image_features @ text_features.T).softmax(dim=-1)
        return similarity






