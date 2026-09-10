import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mvp.attack.attack_utils.attention_with_trust import ScaledDotProductAttentionWithTrust

def compute_attn_score_from_features(x, trust_score=None, temperature=1.0):
    V, C, W, H = x.shape
    out = []
    scores = []
    for i in range(V):
        xx = x[i]
        cav_num = xx.shape[0]
        xx = xx.view(cav_num, C, -1).permute(2, 0, 1)
        att = ScaledDotProductAttentionWithTrust(C, temperature=temperature)
        h = att(xx, xx, xx, trust_score=trust_score)
        h = h.permute(1, 2, 0).reshape(1, C, W, H)[0, ...]
        score = att.attn.permute(1, 2, 0).reshape(1, cav_num, W, H)[0, ...]
        scores.append(score)
        out.append(h)
    return torch.stack(scores), torch.stack(out)

def attFusionLoss(model, spatial_features, attacker_index, center=None, feature_size=10, partial=False,
                  trust_score=None, temperature=1.0, device="cuda:0"):
    x = spatial_features

    attn_maps, _ = compute_attn_score_from_features(
        x, trust_score=trust_score, temperature=temperature
    )

    cav_num = attn_maps.shape[0]
    att_i = attacker_index

    if center is not None:
        aligned_center = center.astype(np.int32)
        scope_w = (aligned_center[1]-feature_size-1, aligned_center[1]+feature_size+1)
        scope_l = (aligned_center[0]-feature_size-1, aligned_center[0]+feature_size+1)
    else:
        scope_w = (0, attn_maps.shape[2])
        scope_l = (0, attn_maps.shape[3])

    loss = - attn_maps[att_i, :, scope_w[0]:scope_w[1], scope_l[0]:scope_l[1]].mean()
    for i in range(cav_num):
        if i != attacker_index:
            loss = loss + attn_maps[i, :, scope_w[0]:scope_w[1], scope_l[0]:scope_l[1]].mean()

    return loss

def v2vamLoss(model, attacker_first=False, device="cuda:0"):
    return torch.mean(model.fusion_net.attn_score_W) + torch.mean(model.fusion_net.attn_score_H).to(device)

def coalignLoss(model, partial=False, attacker_first=False, device="cuda:0"):
    cav_num = model.fusion_net[0].attn_score.shape[1]
    if cav_num <= 2 or partial:
        return torch.mean(model.fusion_net[0].attn_score[0,1,...])-\
            torch.mean(model.fusion_net[0].attn_score[0,0,...]).to(device)
    else:
        loss = torch.mean(model.fusion_net[0].attn_score[0,1,...])-\
            torch.mean(model.fusion_net[0].attn_score[0,0,...]).to(device)
        for i in range(2, cav_num):
            loss -= torch.mean(model.fusion_net[0].attn_score[0,i,...]).to(device)
    return loss

def where2commLoss(model, partial=False, attacker_first=False, device="cuda:0"):
    cav_num = model.fusion_net.fuse_modules[0].attn_score.shape[1]
    if cav_num <= 2 or partial:
        return torch.mean(model.fusion_net.fuse_modules[0].attn_score[0,1,...])-\
            torch.mean(model.fusion_net.fuse_modules[0].attn_score[0,0,...]).to(device)
    else:
        loss = torch.mean(model.fusion_net.fuse_modules[0].attn_score[0,1,...])-\
            torch.mean(model.fusion_net.fuse_modules[0].attn_score[0,0,...]).to(device)
        for i in range(2, cav_num):
            loss -= torch.mean(model.fusion_net.fuse_modules[0].attn_score[0,i,...]).to(device)
        return loss
