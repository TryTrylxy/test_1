import math
from typing import Tuple, Union

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn

from clip import clip
from sentence_transformers import util
# from x2vlm import X2VLM


class AllGather(torch.autograd.Function):
    """An autograd function that performs allgather on a tensor."""

    @staticmethod
    def forward(ctx, tensor, rank, world_size):
        output = [torch.empty_like(tensor) for _ in range(world_size)]
        dist.all_gather(output, tensor)
        ctx.rank = rank
        ctx.batch_size = tensor.shape[0]
        return torch.cat(output, 0)

    @staticmethod
    def backward(ctx, grad_output):
        return (
            grad_output[ctx.batch_size * ctx.rank: ctx.batch_size * (ctx.rank + 1)],
            None,
            None
        )

allgather = AllGather.apply

class EncoderImage(nn.Module):
    def __init__(self, opt, embed_dim):
        super(EncoderImage, self).__init__()
        self.embed_size = embed_dim
        self.k = opt['mv']['k']
        self.fc_list = nn.ModuleList([nn.Linear(self.embed_size, self.embed_size) for _ in range(opt['mv']['k'])])

    def forward(self, images):
        """Extract image feature vectors."""
        # assuming that the precomputed features are already l2-normalized

        emb_list = []
        for fc in self.fc_list:
            emb = fc(images)
            emb_list.append(emb)

        return emb_list

    # def load_state_dict(self, state_dict):
    #     """Copies parameters. overwritting the default one to
    #     accept state_dict from Full model
    #     """
    #     own_state = self.state_dict()
    #     new_state = OrderedDict()
    #     for name, param in state_dict.items():
    #         if name in own_state:
    #             new_state[name] = param

    #     super(EncoderImage, self).load_state_dict(new_state)

class unire(nn.Module):
    def __init__(self, args, config: dict):
        super().__init__()
        self.args = args
        self.config = config
        self.loss_config = config['loss_config']
        self.device = torch.device(args.gpu)

        # set model backbone
        if config['backbone'] == 'CLIP':
            self.clip_model, self.preprocess = clip.load(config['clip_model'], device=self.device, jit=False)
            self.embed_dim = self.clip_model.embed_dim
        elif config['backbone'] == 'X2VLM':
            import yaml
            self.c = yaml.load(open('/home/jncsnlp3/SSD2/syy/aaai24_itr_cusa-main_06/X2VLM/configs/pretrain/x2vlm_base_1b.yaml', 'r'), Loader=yaml.Loader)
            from X2VLM.models.model_retrieval import XVLMForRetrieval
            self.x2_model = XVLMForRetrieval(config=self.c)
            self.x2_model.load_pretrained('/home/jncsnlp3/SSD2/syy/aaai24_itr_cusa-main_06/X2VLM/pretrained/x2vlm_base_4m.th', self.c, is_eval=False)
            self.x2_model = self.x2_model.to(self.device)
            self.embed_dim = self.c['embed_dim']
            # attributes_and_methods = dir(self.x2_model)
            # custom_attributes_and_methods = [item for item in attributes_and_methods if not item.startswith('__')]
            # for item in custom_attributes_and_methods:
            #     try:
            #         value = getattr(self.x2_model, item)
            #         print(f"{item}: {value}")
            #     except AttributeError:
            #         print(f"Error accessing {item}")
            
        # projection layer for image, one is for cross-modal retrieval, the other is for uni-modal retrieval
        # for corss-modal retrieval

        if self.config['do_mv']:
            self.img_enc = EncoderImage(self.config, self.embed_dim)

        if self.is_mode_on("contrastive") or self.is_mode_on("cross_softlabel"):
            self.ln_cross_image_projection = nn.LayerNorm(self.embed_dim)
            self.ln_cross_text_projection = nn.LayerNorm(self.embed_dim)
            self.cross_image_projection = nn.Linear(self.embed_dim, self.embed_dim)
            self.cross_text_projection = nn.Linear(self.embed_dim, self.embed_dim)
        # for uni-modal retrieval
        if self.is_mode_on("uni_softlabel"):
            self.ln_uni_image_projection = nn.LayerNorm(self.embed_dim)
            self.ln_uni_text_projection = nn.LayerNorm(self.embed_dim)
            self.uni_image_projection = nn.Linear(self.embed_dim, self.embed_dim)
            self.uni_text_projection = nn.Linear(self.embed_dim, self.embed_dim)

        # set tau
        if self.is_mode_on("contrastive"):
            self.__init_tau = self.loss_config['contrastive']['tau']
            self.tau = nn.Parameter(torch.tensor(self.__init_tau, device=self.device))

        if self.is_mode_on("cross_softlabel"):
            self.__init_cross_image_tau = self.loss_config['cross_softlabel']['image_tau']
            self.__init_cross_text_tau = self.loss_config['cross_softlabel']['text_tau']
            self.__init_cross_tau = (self.__init_cross_image_tau + self.__init_cross_text_tau) / 2.0
            self.__init_cross_the_softlabel_image_tau = self.loss_config['cross_softlabel']['the_softlabel_image_tau']
            self.__init_cross_the_softlabel_text_tau = self.loss_config['cross_softlabel']['the_softlabel_text_tau']
            self.__init_cross_the_softlabel_tau = (self.__init_cross_the_softlabel_image_tau + self.__init_cross_the_softlabel_text_tau) / 2.0
            if self.is_each_cross_soft_mode():
                if self.loss_config['cross_softlabel']['use_same_tau']:
                    self.cross_tau = nn.Parameter(torch.tensor(self.__init_cross_tau, device=self.device))
                else:
                    self.cross_tau_image = nn.Parameter(torch.tensor(self.__init_cross_image_tau, device=self.device))
                    self.cross_tau_text = nn.Parameter(torch.tensor(self.__init_cross_text_tau, device=self.device))
                if self.loss_config['cross_softlabel']['use_same_softlabel_tau']:
                    self.cross_the_softlabel_tau = nn.Parameter(torch.tensor(self.__init_cross_the_softlabel_tau, device=self.device))
                else:
                    self.cross_the_softlabel_tau_image = nn.Parameter(torch.tensor(self.__init_cross_the_softlabel_image_tau, device=self.device))
                    self.cross_the_softlabel_tau_text = nn.Parameter(torch.tensor(self.__init_cross_the_softlabel_text_tau, device=self.device))
            else:
                self.cross_tau = nn.Parameter(torch.tensor(self.__init_cross_tau, device=self.device))
                self.cross_the_softlabel_tau = nn.Parameter(torch.tensor(self.__init_cross_the_softlabel_tau, device=self.device))

        if self.is_mode_on("uni_softlabel"):
            self.__init_uni_image_tau = self.loss_config['uni_softlabel']['image_tau']
            self.__init_uni_text_tau = self.loss_config['uni_softlabel']['text_tau']
            self.__init_uni_tau = (self.__init_uni_image_tau + self.__init_uni_text_tau) / 2.0
            self.__init_uni_the_softlabel_image_tau = self.loss_config['uni_softlabel']['the_softlabel_image_tau']
            self.__init_uni_the_softlabel_text_tau = self.loss_config['uni_softlabel']['the_softlabel_text_tau']
            self.__init_uni_the_softlabel_tau = (self.__init_uni_the_softlabel_image_tau + self.__init_uni_the_softlabel_text_tau) / 2.0
            
            if not self.loss_config['uni_softlabel']['use_cross_softlabel_same_tau']:
                if self.loss_config['uni_softlabel']['use_same_tau']:
                    self.uni_tau = nn.Parameter(torch.tensor(self.__init_uni_tau, device=self.device))
                else:
                    self.uni_tau_image = nn.Parameter(torch.tensor(self.__init_uni_image_tau, device=self.device))
                    self.uni_tau_text = nn.Parameter(torch.tensor(self.__init_uni_text_tau, device=self.device))

            if not self.loss_config['uni_softlabel']['use_cross_softlabel_same_softlabel_tau'] or not self.is_mode_on("cross_softlabel"):
                if self.loss_config['uni_softlabel']['use_same_softlabel_tau']:
                    self.uni_the_softlabel_tau = nn.Parameter(torch.tensor(self.__init_uni_the_softlabel_tau, device=self.device))
                else:
                    self.uni_the_softlabel_tau_image = nn.Parameter(torch.tensor(self.__init_uni_the_softlabel_image_tau, device=self.device))
                    self.uni_the_softlabel_tau_text = nn.Parameter(torch.tensor(self.__init_uni_the_softlabel_text_tau, device=self.device))

        self.initialize_parameters()

    def is_all_gather(self):
        """check if all_gather"""
        return "is_all_gather" in self.config and self.config['is_all_gather']

    def is_mode_on(self, modeName: str) -> bool:
        return self.loss_config[modeName]['is_on']

    def is_add_cross_soft_mode(self):
        """check if add softlabel"""
        return self.is_mode_on("cross_softlabel") and self.loss_config['cross_softlabel']['cross_softlabel_mode'] == "add"

    def is_dot_cross_soft_mode(self):
        """check if dot softlabel"""
        return self.is_mode_on("cross_softlabel") and self.loss_config['cross_softlabel']['cross_softlabel_mode'] == "dot"

    def is_each_cross_soft_mode(self):
        """check if each softlabel"""
        return self.is_mode_on("cross_softlabel") and self.loss_config['cross_softlabel']['cross_softlabel_mode'] == "each"

    def is_mean_contrastive_loss_mode(self, lossName):
        return self.loss_config[lossName]['contrastive_loss_mode'] == "mean"

    def is_sum_contrastive_loss_mode(self, lossName):
        return self.loss_config[lossName]['contrastive_loss_mode'] == "sum"

    def encode_image(self, image, cross_modal=True):
        """Returns the image embedding "z" of shape [batch_size, projection_dim]."""
        image_features = self.clip_model.encode_image(image)
        return self._encode_image_features(image_features, cross_modal=cross_modal)

    def encode_text(self, text, cross_modal=True):
        """Returns the text embedding "z" of shape [batch_size, projection_dim]."""
        text_features = self.clip_model.encode_text(text)
        return self._encode_text_features(text_features, cross_modal=cross_modal)

    def _encode_image_features(self, image_features, cross_modal=True):
        """encode from clip model"""
        if cross_modal and (self.is_mode_on("contrastive") or self.is_mode_on("cross_softlabel")):
            image_features = self.ln_cross_image_projection(image_features)
            image_features = self.cross_image_projection(image_features)
        elif (not cross_modal) and self.is_mode_on("uni_softlabel"):
            image_features = self.ln_uni_image_projection(image_features)
            image_features = self.uni_image_projection(image_features)
        return image_features

    def _encode_text_features(self, text_features, cross_modal=True):
        """encode from clip model"""
        if cross_modal and (self.is_mode_on("contrastive") or self.is_mode_on("cross_softlabel")):
            text_features = self.ln_cross_text_projection(text_features)
            text_features = self.cross_text_projection(text_features)
        elif (not cross_modal) and self.is_mode_on("uni_softlabel"):
            text_features = self.ln_uni_text_projection(text_features)
            text_features = self.uni_text_projection(text_features)
        return text_features

    def get_similarity(self, image_features, text_features, cross_modal=True):
        # normalized features
        image_features = image_features / image_features.norm(dim=1, keepdim=True)
        text_features = text_features / text_features.norm(dim=1, keepdim=True)

        if cross_modal:
            """if cross-modal retrieval, return the similarity between image and text"""
            logits_per_image = image_features @ text_features.t()
            logits_per_text = logits_per_image.t()
            return logits_per_image, logits_per_text
        else:
            """if uni-modal retrieval, return the similarity between image and image, text and text"""
            logits_image_image = image_features @ image_features.t()
            logits_text_text = text_features @ text_features.t()
            return logits_image_image, logits_text_text
        
    def get_similarity_dot(self, A, B):
        A = A / A.norm(dim=1, keepdim=True)
        B = B / B.norm(dim=1, keepdim=True)
        return (A * B).sum(dim=1)

    def initialize_parameters(self):
        """Initialize the model parameters."""
        if self.is_mode_on("contrastive") or self.is_mode_on("cross_softlabel"):
            nn.init.normal_(self.cross_image_projection.weight, std=0.02)
            nn.init.normal_(self.cross_text_projection.weight, std=0.02)
        if self.is_mode_on("uni_softlabel"):
            nn.init.normal_(self.uni_image_projection.weight, std=0.02)
            nn.init.normal_(self.uni_text_projection.weight, std=0.02)

        if self.is_mode_on("contrastive"):
            if self.loss_config['contrastive']['is_block_tau']:
                self.tau.requires_grad_(False)

        if self.is_mode_on("cross_softlabel"):
            if self.loss_config['cross_softlabel']['is_block_tau']:
                if hasattr(self, "cross_tau"):
                    self.cross_tau.requires_grad_(False)
                else:
                    self.cross_tau_image.requires_grad_(False)
                    self.cross_tau_text.requires_grad_(False)
            if self.loss_config['cross_softlabel']['is_block_softlabel_tau']:
                if hasattr(self, "cross_the_softlabel_tau"):
                    self.cross_the_softlabel_tau.requires_grad_(False)
                else:
                    self.cross_the_softlabel_tau_image.requires_grad_(False)
                    self.cross_the_softlabel_tau_text.requires_grad_(False)

        if self.is_mode_on("uni_softlabel"):
            if not self.loss_config['uni_softlabel']['use_cross_softlabel_same_tau']:
                if self.loss_config['uni_softlabel']['is_block_tau']:
                    if hasattr(self, "uni_tau"):
                        self.uni_tau.requires_grad_(False)
                    else:
                        self.uni_tau_image.requires_grad_(False)
                        self.uni_tau_text.requires_grad_(False)
            if not self.loss_config['uni_softlabel']['use_cross_softlabel_same_softlabel_tau']:
                if self.loss_config['uni_softlabel']['is_block_softlabel_tau']:
                    if hasattr(self, "uni_the_softlabel_tau"):
                        self.uni_the_softlabel_tau.requires_grad_(False)
                    else:
                        self.uni_the_softlabel_tau_image.requires_grad_(False)
                        self.uni_the_softlabel_tau_text.requires_grad_(False)

    def load_state_dict(self, state_dict, strict=True):
        """load state dict"""
        if state_dict is None:
            return "state_dict is None"
        msg = super().load_state_dict(state_dict, strict)
        return msg

    def ContrastiveLoss(self, logits_per_image, logits_per_text, idx=None):
        # contrastive loss
        if idx is None:
            sim_targets = torch.eye(logits_per_image.shape[0], device=self.device)
        else:
            idx = idx.view(-1, 1)
            pos_idx = torch.eq(idx, idx.t()).float()
            sim_targets = (pos_idx / pos_idx.sum(1, keepdim=True)).to(self.device)
        if self.is_mean_contrastive_loss_mode("contrastive"):
            loss_i2t = -torch.mean(F.log_softmax(logits_per_image / self.tau, dim=1) * sim_targets, dim=1).mean()
            loss_t2i = -torch.mean(F.log_softmax(logits_per_text / self.tau, dim=1) * sim_targets, dim=1).mean()
        elif self.is_sum_contrastive_loss_mode("contrastive"):
            loss_i2t = -torch.sum(F.log_softmax(logits_per_image / self.tau, dim=1) * sim_targets, dim=1).mean()
            loss_t2i = -torch.sum(F.log_softmax(logits_per_text / self.tau, dim=1) * sim_targets, dim=1).mean()
        else:
            raise ValueError("contrastive loss mode error")
        contrastive_loss = loss_i2t + loss_t2i

        return contrastive_loss

    def TripletLoss_mv(self, v_list, t):
        batch_size = t.size(0)
        pos_mask = torch.eye(batch_size)
        pos_mask = pos_mask.to(self.device)
        neg_mask = 1 - pos_mask

        # calculate multi-view similarity score
        scores_list = []
        for v in v_list:
            scores = v.mm(t.t())
            scores_list.append(scores)

        # calculate image embedding similarity
        view_sim = torch.tensor(0)
        if self.config['mv']['k'] > 1:
            view_sim_list = []
            for i in range(self.config['mv']['k']):
                for j in range(i+1, self.config['mv']['k']):
                    sims = v_list[i].mm(v_list[j].t())
                    sim = sims.diag().mean()
                    view_sim_list.append(sim)
            view_sim_list = torch.stack(view_sim_list, dim=0)

        # max score
        comb_scores = torch.stack(scores_list, dim=0)
        (max_scores, max_id) = comb_scores.max(0)

        # multi-view up loss
        loss_list = []
        for scores in scores_list:
            pos_scores = scores.diag().view(batch_size, 1)
            pos_scores_t = pos_scores.expand_as(scores)
            pos_scores_v = pos_scores.t().expand_as(scores)
            loss_t = (max_scores - pos_scores_t + self.config['mv']['margin']).clamp(min=0)
            loss_v = (max_scores - pos_scores_v + self.config['mv']['margin']).clamp(min=0)
            loss_t = loss_t * neg_mask
            loss_v = loss_v * neg_mask
            loss_t = loss_t.max(dim=1)[0]
            loss_v = loss_v.max(dim=0)[0]
            loss_t = loss_t.mean()
            loss_v = loss_v.mean()
            loss = (loss_t + loss_v) / 2
            loss_list.append(loss)

        loss_list = torch.stack(loss_list, dim=0)
        up_loss = loss_list.mean()

        # multi-view low loss
        loss_list = []
        for scores in scores_list:
            max_pos_scores = max_scores.diag().view(batch_size, 1)
            max_pos_scores_t = max_pos_scores.expand_as(scores)
            max_pos_scores_v = max_pos_scores.t().expand_as(scores)
            loss_t = (scores - max_pos_scores_t + self.config['mv']['margin']).clamp(min=0)
            loss_v = (scores - max_pos_scores_v + self.config['mv']['margin']).clamp(min=0)
            loss_t = loss_t * neg_mask
            loss_v = loss_v * neg_mask
            loss_t = loss_t.max(dim=1)[0]
            loss_v = loss_v.max(dim=0)[0]
            loss_t = loss_t.mean()
            loss_v = loss_v.mean()
            loss = (loss_t + loss_v) / 2
            loss_list.append(loss)

        loss_list = torch.stack(loss_list, dim=0)
        low_loss = loss_list.mean()

        loss = self.config['mv']['weight'] * up_loss + (1 - self.config['mv']['weight']) * low_loss

        return loss, up_loss, low_loss

    def UnifiedLoss_mv(self, v_list, t):
        batch_size = t.size(0)
        pos_mask = torch.eye(batch_size)
        pos_mask = pos_mask.to(self.device)
        neg_mask = 1 - pos_mask

        # calculate multi-view similarity score
        scores_list = []
        for v in v_list:
            scores = v.mm(t.t())
            scores_list.append(scores)

        # calculate image embedding similarity
        view_sim = torch.tensor(0)
        if self.config['mv']['k'] > 1:
            view_sim_list = []
            for i in range(self.config['mv']['k']):
                for j in range(i+1, self.config['mv']['k']):
                    sims = v_list[i].mm(v_list[j].t())
                    sim = sims.diag().mean()
                    view_sim_list.append(sim)
            view_sim_list = torch.stack(view_sim_list, dim=0)

        # max score
        comb_scores = torch.stack(scores_list, dim=0)
        (max_scores, max_id) = comb_scores.max(0)

        # multi-view up loss
        loss_list = []
        for scores in scores_list:
            pos_scores = scores.diag().view(batch_size, 1)
            pos_scores_t = pos_scores.expand_as(scores)
            pos_scores_v = pos_scores.t().expand_as(scores)
            loss_t = max_scores - pos_scores_t + self.config['mv']['margin']
            loss_v = max_scores - pos_scores_v + self.config['mv']['margin']
            loss_t = loss_t * neg_mask - pos_mask
            loss_v = loss_v * neg_mask - pos_mask
            loss_t = torch.logsumexp(loss_t / self.config['mv']['tau'], dim=1) * self.config['mv']['tau']
            loss_v = torch.logsumexp(loss_v / self.tau, dim=0) * self.config['mv']['tau']
            loss_t = torch.nn.functional.softplus(loss_t, beta=1 / self.config['mv']['tau'])
            loss_v = torch.nn.functional.softplus(loss_v, beta=1 / self.config['mv']['tau'])
            loss_t = loss_t.mean()
            loss_v = loss_v.mean()
            loss = (loss_t + loss_v) / 2
            loss_list.append(loss)

        loss_list = torch.stack(loss_list, dim=0)
        up_loss = loss_list.mean()

        # multi-view low loss
        loss_list = []
        for scores in scores_list:
            max_pos_scores = max_scores.diag().view(batch_size, 1)
            max_pos_scores_t = max_pos_scores.expand_as(scores)
            max_pos_scores_v = max_pos_scores.t().expand_as(scores)
            loss_t = scores - max_pos_scores_t + self.config['mv']['margin']
            loss_v = scores - max_pos_scores_v + self.config['mv']['margin']
            loss_t = loss_t * neg_mask - pos_mask
            loss_v = loss_v * neg_mask - pos_mask
            loss_t = torch.logsumexp(loss_t / self.config['mv']['tau'], dim=1) * self.config['mv']['tau']
            loss_v = torch.logsumexp(loss_v / self.config['mv']['tau'], dim=0) * self.config['mv']['tau']
            loss_t = torch.nn.functional.softplus(loss_t, beta=1 / self.config['mv']['tau'])
            loss_v = torch.nn.functional.softplus(loss_v, beta=1 / self.config['mv']['tau'])
            loss_t = loss_t.mean()
            loss_v = loss_v.mean()
            loss = (loss_t + loss_v) / 2
            loss_list.append(loss)

        loss_list = torch.stack(loss_list, dim=0)
        low_loss = loss_list.mean()

        loss = self.config['mv']['weight'] * up_loss + (1 - self.config['mv']['weight']) * low_loss

        return loss, up_loss, low_loss
    
    def triplet_loss(self, a, b, margin):
        return torch.maximum(torch.zeros_like(a), b - a + margin).mean()


    def KLContrastiveSimLoss(self, logits, softlabel, tau, softlabel_tau, lossName, use_loss="kl"):
        """
        KL divergence loss
        make logits and softlabel have the same distribution
        logits to softlabel
        """
        # softmax for softlabel
        sim_targets = F.softmax(softlabel / softlabel_tau, dim=1)

        # log softmax
        logit_inputs = F.log_softmax(logits / tau, dim=1)

        if use_loss == "kl":
            # KL divergence
            loss = F.kl_div(logit_inputs, sim_targets, reduction='batchmean')
        elif use_loss == "contrastive":
            # Switch to the same loss as ContrastiveLoss, but sim_targets is soft
            if self.is_mean_contrastive_loss_mode(lossName):
                loss = -torch.mean(logit_inputs * sim_targets, dim=1).mean()
            elif self.is_sum_contrastive_loss_mode(lossName):
                loss = -torch.sum(logit_inputs * sim_targets, dim=1).mean()
            else:
                raise ValueError("contrastive loss mode error")
        else:
            raise ValueError("loss mode error")

        return loss

    @torch.no_grad()
    def clamp_tau(self):
        # clip tau to prevent overflow
        if self.is_mode_on("contrastive"):
            self.tau.clamp_(min=self.loss_config['contrastive']['tau_min'], max=self.loss_config['contrastive']['tau_max'])

        if self.is_mode_on("cross_softlabel"):
            if hasattr(self, "cross_tau"):
                self.cross_tau.clamp_(min=(self.loss_config['cross_softlabel']['image_tau_min']+self.loss_config['cross_softlabel']['text_tau_min'])/2.0,
                                      max=(self.loss_config['cross_softlabel']['image_tau_max']+self.loss_config['cross_softlabel']['text_tau_max'])/2.0)
            else:
                self.cross_tau_image.clamp_(min=self.loss_config['cross_softlabel']['image_tau_min'],
                                            max=self.loss_config['cross_softlabel']['image_tau_max'])
                self.cross_tau_text.clamp_(min=self.loss_config['cross_softlabel']['text_tau_min'],
                                           max=self.loss_config['cross_softlabel']['text_tau_max'])
            if hasattr(self, "cross_the_softlabel_tau"):
                self.cross_the_softlabel_tau.clamp_(min=(self.loss_config['cross_softlabel']['the_softlabel_image_tau_min']+self.loss_config['cross_softlabel']['the_softlabel_text_tau_min'])/2.0,
                                                    max=(self.loss_config['cross_softlabel']['the_softlabel_image_tau_max']+self.loss_config['cross_softlabel']['the_softlabel_text_tau_max'])/2.0)
            else:
                self.cross_the_softlabel_tau_image.clamp_(min=self.loss_config['cross_softlabel']['the_softlabel_image_tau_min'],
                                                          max=self.loss_config['cross_softlabel']['the_softlabel_image_tau_max'])
                self.cross_the_softlabel_tau_text.clamp_(min=self.loss_config['cross_softlabel']['the_softlabel_text_tau_min'],
                                                         max=self.loss_config['cross_softlabel']['the_softlabel_text_tau_max'])

        if self.is_mode_on("uni_softlabel"):
            if not self.loss_config['uni_softlabel']['use_cross_softlabel_same_tau']:
                if hasattr(self, "uni_tau"):
                    self.uni_tau.clamp_(min=(self.loss_config['uni_softlabel']['image_tau_min']+self.loss_config['uni_softlabel']['text_tau_min'])/2.0,
                                        max=(self.loss_config['uni_softlabel']['image_tau_max']+self.loss_config['uni_softlabel']['text_tau_max'])/2.0)
                else:
                    self.uni_tau_image.clamp_(min=self.loss_config['uni_softlabel']['image_tau_min'],
                                            max=self.loss_config['uni_softlabel']['image_tau_max'])
                    self.uni_tau_text.clamp_(min=self.loss_config['uni_softlabel']['text_tau_min'],
                                            max=self.loss_config['uni_softlabel']['text_tau_max'])
            if not self.loss_config['uni_softlabel']['use_cross_softlabel_same_softlabel_tau']:
                if hasattr(self, "uni_the_softlabel_tau"):
                    self.uni_the_softlabel_tau.clamp_(min=(self.loss_config['uni_softlabel']['the_softlabel_image_tau_min']+self.loss_config['uni_softlabel']['the_softlabel_text_tau_min'])/2.0,
                                                      max=(self.loss_config['uni_softlabel']['the_softlabel_image_tau_max']+self.loss_config['uni_softlabel']['the_softlabel_text_tau_max'])/2.0)
                else:
                    self.uni_the_softlabel_tau_image.clamp_(min=self.loss_config['uni_softlabel']['the_softlabel_image_tau_min'],
                                                            max=self.loss_config['uni_softlabel']['the_softlabel_image_tau_max'])
                    self.uni_the_softlabel_tau_text.clamp_(min=self.loss_config['uni_softlabel']['the_softlabel_text_tau_min'],
                                                           max=self.loss_config['uni_softlabel']['the_softlabel_text_tau_max'])

    def forward(self, image, text, softlabel_image_features=None, softlabel_text_features=None, epoch=None, idx=None, softlabel_text_features_r=None, text_neg=None, im_neg=None):
        # rankNum = torch.distributed.get_rank()
        # worldSize = torch.distributed.get_world_size()
        # clip tau to prevent overflow
        self.clamp_tau()

        # use clip model to extract features
        # can be used for both cross-modal and uni-modal retrieval
        if self.config['backbone'] == 'CLIP':
            image_features = self.clip_model.encode_image(image)
            text_features = self.clip_model.encode_text(text)
            if self.config['do_neg']:
                text_features_neg = self.clip_model.encode_text(text_neg)
                image_features_neg = self.clip_model.encode_image(im_neg)
        elif self.config['backbone'] == 'X2VLM':
            image_features, _ = self.x2_model.get_vision_embeds(image)
            text_features = self.x2_model.get_text_embeds(text.input_ids, text.attention_mask)
            image_features, text_features = self.x2_model.get_features(image_features, text_features)
        # if self.is_all_gather() and idx is not None:
        #     idx_all = allgather(idx, rankNum, worldSize)
        # else:
        idx_all = idx

        # use clip model to extract features and similarity
        # for cross-modal retrieval
        if self.is_mode_on("contrastive") or self.is_mode_on("cross_softlabel"):
            cross_image_features, cross_text_features = self._encode_image_features(
                image_features, cross_modal=True), self._encode_text_features(text_features, cross_modal=True)
            if self.config['do_neg']:
                cross_text_features_neg = self._encode_text_features(text_features_neg, cross_modal=True)
                cross_image_features_neg = self._encode_image_features(image_features_neg, cross_modal=True)
            if self.is_all_gather():
                cross_image_features, cross_text_features = allgather(
                    cross_image_features, rankNum, worldSize), allgather(cross_text_features, rankNum, worldSize)
            logits_per_image, logits_per_text = self.get_similarity(cross_image_features, cross_text_features, cross_modal=True)

        # for uni-modal retrieval
        if self.is_mode_on("uni_softlabel"):
            uni_image_features, uni_text_features = self._encode_image_features(
                image_features, cross_modal=False), self._encode_text_features(text_features, cross_modal=False)
            if self.is_all_gather():
                uni_image_features, uni_text_features = allgather(uni_image_features, rankNum, worldSize), allgather(
                    uni_text_features, rankNum, worldSize)
            logits_image_image, logits_text_text = self.get_similarity(uni_image_features, uni_text_features, cross_modal=False)

        # use external softlabel to get similarity
        # only image-image and text-text similarity
        if self.is_mode_on("cross_softlabel") or self.is_mode_on("uni_softlabel"):
            with torch.no_grad():
                if self.is_all_gather():
                    softlabel_image_features, softlabel_text_features = allgather(
                        softlabel_image_features, rankNum, worldSize), allgather(softlabel_text_features, rankNum, worldSize)
                softlabel_image_sim = util.cos_sim(softlabel_image_features, softlabel_image_features)
                softlabel_text_sim = util.cos_sim(softlabel_text_features, softlabel_text_features)

                if self.is_mode_on("cross_softlabel"):
                    if self.is_add_cross_soft_mode():
                        # Average two similarities
                        softlabel_all_sim = (softlabel_image_sim + softlabel_text_sim) / 2.0
                    elif self.is_dot_cross_soft_mode():
                        # Dot two similarities
                        softlabel_image_sim_copy = softlabel_image_sim.clone()
                        softlabel_text_sim_copy = softlabel_text_sim.clone()
                        softlabel_image_sim_copy[softlabel_image_sim_copy < 0.0] = 0.0
                        softlabel_text_sim_copy[softlabel_text_sim_copy < 0.0] = 0.0
                        softlabel_all_sim = softlabel_image_sim_copy * softlabel_text_sim_copy
                        softlabel_all_sim = torch.sqrt(softlabel_all_sim)
                    elif self.is_each_cross_soft_mode():
                        pass
                    else:
                        raise ValueError("softlabel mode error")

        cross_modal_loss, uni_modal_loss, contrastive_loss = torch.tensor(0.0, device=self.device), torch.tensor(
            0.0, device=self.device), torch.tensor(0.0, device=self.device)
        loss_i2t, loss_t2i, loss_i2i, loss_t2t = torch.tensor(0.0, device=self.device), torch.tensor(
            0.0, device=self.device), torch.tensor(0.0, device=self.device), torch.tensor(0.0, device=self.device)
        tri_txt_neg_loss, tri_img_neg_loss = torch.tensor(0.0, device=self.device), torch.tensor(
            0.0, device=self.device)

        if self.is_mode_on("cross_softlabel"):
            # for cross-modal alignment (similarity)
            # image-text and image-image softlabel
            # text-image and text-text softlabel
            softlabel_image_sim_loss = softlabel_image_sim
            softlabel_text_sim_loss = softlabel_text_sim
            if not self.is_each_cross_soft_mode():
                softlabel_image_sim_loss = softlabel_all_sim
                softlabel_text_sim_loss = softlabel_all_sim

            if hasattr(self, "cross_tau"):
                cross_tau_loss_image = self.cross_tau
                cross_tau_loss_text = self.cross_tau
            else:
                cross_tau_loss_image = self.cross_tau_image
                cross_tau_loss_text = self.cross_tau_text

            if hasattr(self, "cross_the_softlabel_tau"):
                cross_the_softlabel_tau_loss_image = self.cross_the_softlabel_tau
                cross_the_softlabel_tau_loss_text = self.cross_the_softlabel_tau
            else:
                cross_the_softlabel_tau_loss_image = self.cross_the_softlabel_tau_image
                cross_the_softlabel_tau_loss_text = self.cross_the_softlabel_tau_text
            
            if not self.config['do_restate_sl']:
                loss_i2t = self.KLContrastiveSimLoss(logits_per_image, softlabel_image_sim_loss, cross_tau_loss_image,
                                                            cross_the_softlabel_tau_loss_image, "cross_softlabel", use_loss=self.loss_config['cross_softlabel']['use_loss'])
                loss_t2i = self.KLContrastiveSimLoss(logits_per_text, softlabel_text_sim_loss, cross_tau_loss_text,
                                                            cross_the_softlabel_tau_loss_text, "cross_softlabel", use_loss=self.loss_config['cross_softlabel']['use_loss'])
            else:
                tmp_sl = util.cos_sim(softlabel_text_features_r, softlabel_text_features)
                loss_i2t = self.KLContrastiveSimLoss(logits_per_image, tmp_sl, cross_tau_loss_image,
                                                            cross_the_softlabel_tau_loss_image, "cross_softlabel", use_loss=self.loss_config['cross_softlabel']['use_loss'])
                tmp_sl = util.cos_sim(softlabel_text_features, softlabel_text_features_r)
                loss_t2i = self.KLContrastiveSimLoss(logits_per_text, tmp_sl, cross_tau_loss_text,
                                                            cross_the_softlabel_tau_loss_text, "cross_softlabel", use_loss=self.loss_config['cross_softlabel']['use_loss'])
            cross_modal_loss = (loss_i2t + loss_t2i) / 2
            cross_modal_loss = cross_modal_loss * self.loss_config['cross_softlabel']['rate']

        if self.is_mode_on("uni_softlabel"):
            # fro uni-modal alignment (similarity)
            # image-image and image-image softlabel
            # text-text and text-text softlabel
            if not self.loss_config['uni_softlabel']['use_cross_softlabel_same_tau']:
                if hasattr(self, "uni_tau"):
                    uni_tau_image_loss = self.uni_tau
                    uni_tau_text_loss = self.uni_tau
                else:
                    uni_tau_image_loss = self.uni_tau_image
                    uni_tau_text_loss = self.uni_tau_text
            else:
                if hasattr(self, "cross_tau"):
                    uni_tau_image_loss = self.cross_tau
                    uni_tau_text_loss = self.cross_tau
                else:
                    uni_tau_image_loss = self.cross_tau_image
                    uni_tau_text_loss = self.cross_tau_text
            if not self.loss_config['uni_softlabel']['use_cross_softlabel_same_softlabel_tau'] or not self.is_mode_on("cross_softlabel"):
                if hasattr(self, "uni_the_softlabel_tau"):
                    uni_the_softlabel_tau_image_loss = self.uni_the_softlabel_tau
                    uni_the_softlabel_tau_text_loss = self.uni_the_softlabel_tau
                else:
                    uni_the_softlabel_tau_image_loss = self.uni_the_softlabel_tau_image
                    uni_the_softlabel_tau_text_loss = self.uni_the_softlabel_tau_text
            else:
                if hasattr(self, "cross_the_softlabel_tau"):
                    uni_the_softlabel_tau_image_loss = self.cross_the_softlabel_tau
                    uni_the_softlabel_tau_text_loss = self.cross_the_softlabel_tau
                else:
                    uni_the_softlabel_tau_image_loss = self.cross_the_softlabel_tau_image
                    uni_the_softlabel_tau_text_loss = self.cross_the_softlabel_tau_text

            if not self.config['do_restate_sl']:
                loss_i2i = self.KLContrastiveSimLoss(logits_image_image, softlabel_image_sim, uni_tau_image_loss, uni_the_softlabel_tau_image_loss,
                                                        "uni_softlabel", use_loss=self.loss_config['uni_softlabel']['use_loss'])
                loss_t2t = self.KLContrastiveSimLoss(logits_text_text, softlabel_text_sim, uni_tau_text_loss, uni_the_softlabel_tau_text_loss,
                                                            "uni_softlabel", use_loss=self.loss_config['uni_softlabel']['use_loss'])
            else:
                tmp_sl = util.cos_sim(softlabel_text_features_r, softlabel_text_features_r)
                loss_i2i = self.KLContrastiveSimLoss(logits_image_image, tmp_sl, uni_tau_image_loss, uni_the_softlabel_tau_image_loss,
                                                        "uni_softlabel", use_loss=self.loss_config['uni_softlabel']['use_loss'])
                tmp_sl = util.cos_sim(softlabel_text_features, softlabel_text_features)
                loss_t2t = self.KLContrastiveSimLoss(logits_text_text, tmp_sl, uni_tau_text_loss, uni_the_softlabel_tau_text_loss,
                                                            "uni_softlabel", use_loss=self.loss_config['uni_softlabel']['use_loss'])
            uni_modal_loss = (loss_i2i + loss_t2t) / 2
            uni_modal_loss = uni_modal_loss * self.loss_config['uni_softlabel']['rate']

        if self.is_mode_on("contrastive"):
            # the simplest contrastive loss
            # image-text and text-image
            contrastive_loss = self.ContrastiveLoss(logits_per_image, logits_per_text, idx_all)
            contrastive_loss /= 2.0
            contrastive_loss = contrastive_loss * self.loss_config['contrastive']['rate']

        def get_low_dist_mean_idx(tensor_2d: torch.Tensor) -> torch.Tensor:
            with torch.no_grad():
                from sklearn.mixture import GaussianMixture
                row_indices = torch.zeros(tensor_2d.shape[0], dtype=torch.long, device=tensor_2d.device)
                tensor_cpu = tensor_2d.detach().cpu()
                
                for i in range(tensor_2d.shape[0]):
                    row = tensor_cpu[i].unsqueeze(1)
                    gmm = GaussianMixture(n_components=2, random_state=42, max_iter=50).fit(row)
                    
                    means = torch.from_numpy(gmm.means_).squeeze()
                    low_mean_idx = torch.argmin(means)
                    labels = torch.from_numpy(gmm.predict(row))
                    
                    low_mask = (labels == low_mean_idx)
                    if low_mask.any():
                        low_vals = tensor_cpu[i][low_mask]
                        dist = torch.abs(low_vals - means[low_mean_idx])
                        min_local_idx = torch.argmin(dist)
                        # 核心修复：保留维度，避免压缩成标量
                        low_indices = torch.nonzero(low_mask).squeeze(-1)
                        row_indices[i] = low_indices[min_local_idx] if low_indices.ndim > 0 else low_indices
                    else:
                        row_indices[i] = 0
                
                return row_indices
            
        if not self.config['do_TSKL']:
            cross_modal_loss, uni_modal_loss = torch.tensor(0.0, device=self.device), torch.tensor(0.0, device=self.device)
        
        if self.config['do_neg']:
            if self.config['do_TxT']:
                if self.config['do_ab'] or self.config['do_ac'] or self.config['do_abc']:
                    a_sims = torch.diag(logits_per_image)
                if self.config['do_ab'] or self.config['do_bc'] or self.config['do_abc']:
                    b_sims = self.get_similarity_dot(cross_image_features, cross_text_features_neg)
                if self.config['do_ac'] or self.config['do_bc'] or self.config['do_abc']:
                    # c_sims = torch.max(logits_per_image.masked_fill(torch.eye(logits_per_image.shape[0], dtype=bool).to(logits_per_image.device), -torch.inf), dim=1)[0]
                    tmp = get_low_dist_mean_idx(logits_per_image)
                    c_sims = logits_per_image[torch.arange(tmp.shape[0]), tmp]
                alpha = 0.3
                a_b_loss = torch.tensor(0.0, device=self.device)
                b_c_loss = torch.tensor(0.0, device=self.device)
                a_c_loss = torch.tensor(0.0, device=self.device)
                if self.config['do_abcmargin_ab']:
                    alpha = self.config['abc_margin_alpha']
                if self.config['do_ab'] or self.config['do_abc']:
                    a_b_loss = self.triplet_loss(a_sims, b_sims, 0.2 * alpha)
                if self.config['do_bc'] or self.config['do_abc']:
                    b_c_loss = self.triplet_loss(b_sims, c_sims, 0.2 * (1 - alpha))
                if self.config['do_ac'] or self.config['do_abc']:
                    a_c_loss = self.triplet_loss(a_sims, c_sims, 0.2)
                tri_txt_neg_loss = (a_b_loss + b_c_loss + a_c_loss) / 3
            else:
                a_sims = torch.diag(logits_per_image)
                b_sims = self.get_similarity_dot(cross_image_features, cross_text_features_neg)
                tri_txt_neg_loss = self.triplet_loss(a_sims, b_sims, 0.2)

            if self.config['do_TxT']:
                if self.config['do_ab'] or self.config['do_ac'] or self.config['do_abc']:
                    a_sims = torch.diag(logits_per_text)
                if self.config['do_ab'] or self.config['do_bc'] or self.config['do_abc']:
                    b_sims = self.get_similarity_dot(cross_text_features, cross_image_features_neg)
                if self.config['do_ac'] or self.config['do_bc'] or self.config['do_abc']:
                # c_sims = torch.max(logits_per_text.masked_fill(torch.eye(logits_per_text.shape[0], dtype=bool).to(logits_per_text.device), -torch.inf), dim=1)[0]
                    tmp = get_low_dist_mean_idx(logits_per_text)
                    c_sims = logits_per_text[torch.arange(tmp.shape[0]), tmp]
                alpha = 0.3
                a_b_loss = torch.tensor(0.0, device=self.device)
                b_c_loss = torch.tensor(0.0, device=self.device)
                a_c_loss = torch.tensor(0.0, device=self.device)
                if self.config['do_abcmargin_ab']:
                    alpha = self.config['abc_margin_alpha']
                if self.config['do_ab'] or self.config['do_abc']:
                    a_b_loss = self.triplet_loss(a_sims, b_sims, 0.2 * alpha)
                if self.config['do_bc'] or self.config['do_abc']:
                    b_c_loss = self.triplet_loss(b_sims, c_sims, 0.2 * (1 - alpha))
                if self.config['do_ac'] or self.config['do_abc']:
                    a_c_loss = self.triplet_loss(a_sims, c_sims, 0.2)
                tri_img_neg_loss = (a_b_loss + b_c_loss + a_c_loss) / 3
            else:
                a_sims = torch.diag(logits_per_text)
                b_sims = self.get_similarity_dot(cross_text_features, cross_image_features_neg)
                tri_img_neg_loss = self.triplet_loss(a_sims, b_sims, 0.2)
            if not self.config['do_img_neg']:
                tri_img_neg_loss = torch.tensor(0.0, device=self.device)

        loss_mv, up_loss, low_loss = torch.tensor(0.0, device=self.device), torch.tensor(0.0, device=self.device), torch.tensor(0.0, device=self.device)
        if self.config['do_mv']:
            cross_image_features_list = self.img_enc(cross_image_features)
            if self.config['mv']['loss'] == 'triplet':
                loss_mv, up_loss, low_loss = self.TripletLoss_mv(cross_image_features_list, cross_text_features)
            elif self.config['mv']['loss'] == 'infonce':
                loss_mv, up_loss, low_loss = self.UnifiedLoss_mv(cross_image_features_list, cross_text_features)
            else:
                raise
        return cross_modal_loss, uni_modal_loss, contrastive_loss, tri_txt_neg_loss, tri_img_neg_loss, loss_mv, up_loss, low_loss, (loss_i2t.item(), loss_t2i.item(), loss_i2i.item(), loss_t2t.item())
