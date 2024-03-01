# coding=utf-8
import torch
import torch.nn.functional as F
from alg.algs.ERM import ERM


class CORAL(ERM):
    def __init__(self, args):
        super(CORAL, self).__init__(args)
        self.args = args
        self.kernel_type = "mean_cov"
        trained_model_path = r'/data0/czn/longtail_workspace/OpenDG-Eval/output/daml_loader/lr_1e-3_dataaug_yes/office-home_different_class_space/Clip/test_envs_1/seed_0_clipvitb16_cedistill_featsdistill0.2_earlystop10/' \
                             r'best_acc_model_and_args_epoch10.pkl'
        self.dg_pretrained_model = torch.load(trained_model_path, map_location=torch.device('cuda:4'))
        self.new_state_dict = self.fix_state_dict(self.dg_pretrained_model)
        self.featurizer.load_state_dict(self.new_state_dict, strict=False)
        # self.register_buffer("proto_feats", torch.zeros((args.num_classes, self.featurizer.in_features), requires_grad=False))

    def fix_state_dict(self, pretrained_model):
        state_dict = pretrained_model['model_dict']
        for name, values in state_dict.copy().items():
            new_name = name.split('.')
            new_name = new_name[1:]
            new_name = '.'.join(new_name)
            state_dict[new_name] = values
            del state_dict[name]
        return state_dict

    def coral(self, x, y):
        mean_x = x.mean(0, keepdim=True)
        mean_y = y.mean(0, keepdim=True)
        cent_x = x - mean_x
        cent_y = y - mean_y
        cova_x = (cent_x.t() @ cent_x) / (len(x) - 1)
        cova_y = (cent_y.t() @ cent_y) / (len(y) - 1)

        mean_diff = (mean_x - mean_y).pow(2).mean()
        cova_diff = (cova_x - cova_y).pow(2).mean()

        return mean_diff + cova_diff

    def update(self, minibatches, opt, sch):
        objective = 0
        penalty = 0
        nmb = len(minibatches)  # num of domain

        features = [
            self.featurizer(data[0].to(self.device).float()) for data in minibatches
        ]  # features[0].shape (N, 512)
        classifs = [self.classifier(fi) for fi in features]  # classifs[0].shape (N, C)
        targets = [
            data[1].to(self.device).long() for data in minibatches
        ]  # targets[0].shape (C)

        for i in range(nmb):
            objective += F.cross_entropy(classifs[i], targets[i])
            for j in range(i + 1, nmb):
                penalty += self.coral(features[i], features[j])

        objective /= nmb
        if nmb > 1:
            penalty /= nmb * (nmb - 1) / 2

        opt.zero_grad()
        (objective + (self.args.mmd_gamma * penalty)).backward()
        opt.step()
        if sch:
            sch.step()
        if torch.is_tensor(penalty):
            penalty = penalty.item()

        return {
            "class": objective.item(),
            "coral": penalty,
            "total": (objective.item() + (self.args.mmd_gamma * penalty)),
        }
