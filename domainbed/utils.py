import torch
import torch.nn as nn
import torch.nn.functional as F


class SupConLossLambda(nn.Module):
    def __init__(self, lamda: float=0.5, temperature: float=0.07):
        super(SupConLossLambda, self).__init__()
        self.temperature = temperature
        self.lamda = lamda

    def forward(self, features: torch.Tensor, labels: torch.Tensor, domain_labels: torch.Tensor) -> torch.Tensor:
        batch_size, _ = features.shape
        normalized_features = F.normalize(features, p=2, dim=1)
        # create a lookup table for pairwise dot prods
        pairwise_dot_prods = torch.matmul(normalized_features, normalized_features.T)/self.temperature
        loss = 0
        nans = 0
        for i, (label, domain_label) in enumerate(zip(labels, domain_labels)):
            # take the positive and negative samples wrt in/out domain            
            cond_pos_in_domain = torch.logical_and(labels==label, domain_labels == domain_label) # take all positives
            cond_pos_in_domain[i] = False # exclude itself
            cond_pos_out_domain = torch.logical_and(labels==label, domain_labels != domain_label)
            cond_neg_in_domain = torch.logical_and(labels!=label, domain_labels == domain_label)
            cond_neg_out_domain = torch.logical_and(labels!=label, domain_labels != domain_label)

            pos_feats_in_domain = pairwise_dot_prods[cond_pos_in_domain]
            pos_feats_out_domain = pairwise_dot_prods[cond_pos_out_domain]
            neg_feats_in_domain = pairwise_dot_prods[cond_neg_in_domain]
            neg_feats_out_domain = pairwise_dot_prods[cond_neg_out_domain]
            

            # calculate nominator and denominator wrt lambda scaling
            scaled_exp_term = torch.cat((self.lamda * torch.exp(pos_feats_in_domain[:, i]), (1 - self.lamda) * torch.exp(pos_feats_out_domain[:, i])))
            scaled_denom_const = torch.sum(torch.cat((self.lamda * torch.exp(neg_feats_in_domain[:, i]), (1 - self.lamda) * torch.exp(neg_feats_out_domain[:, i]), scaled_exp_term))) + 1e-5
            
            # nof positive samples
            num_positives = pos_feats_in_domain.shape[0] + pos_feats_out_domain.shape[0] # total positive samples
            log_fraction = torch.log((scaled_exp_term / scaled_denom_const) + 1e-5) # take log fraction
            loss_i = torch.sum(log_fraction) / num_positives
            if torch.isnan(loss_i):
                nans += 1
                continue
            loss -= loss_i # sum and average over num positives
        return loss/(batch_size-nans+1) # avg over batch

