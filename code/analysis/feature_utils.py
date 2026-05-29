"""
analysis/feature_utils.py

Shared feature construction for DnR analysis scripts.
Ensures the same feature format is used in training, evaluation,
and all analysis scripts.
"""

import torch


def build_dnr_features(m1, m2, m3, divide_dim, use_cross):
    """
    Build per-modality feature tensors from SMURF outputs.

    Parameters
    ----------
    m1, m2, m3  : tuples (u, r, s) each [seq, batch, divide_dim]
    divide_dim  : int  output dim of each SMURF head
    use_cross   : bool  whether to append cross-modal features
                  (detected from backbone input dim)

    Returns
    -------
    feat_t, feat_a, feat_v : [seq, batch, out_dim]
      out_dim = 3 * divide_dim            if not use_cross
      out_dim = 7 * divide_dim            if use_cross
    """
    u1, r1, s1 = m1
    u2, r2, s2 = m2
    u3, r3, s3 = m3

    if use_cross:
        cross_t = torch.cat([u1-u2, u1-u3, u1*u2, u1*u3], dim=-1)
        cross_a = torch.cat([u2-u1, u2-u3, u2*u1, u2*u3], dim=-1)
        cross_v = torch.cat([u3-u1, u3-u2, u3*u1, u3*u2], dim=-1)
        feat_t  = torch.cat([u1, r1, s1, cross_t], dim=-1)
        feat_a  = torch.cat([u2, r2, s2, cross_a], dim=-1)
        feat_v  = torch.cat([u3, r3, s3, cross_v], dim=-1)
    else:
        feat_t  = torch.cat([u1, r1, s1], dim=-1)
        feat_a  = torch.cat([u2, r2, s2], dim=-1)
        feat_v  = torch.cat([u3, r3, s3], dim=-1)

    return feat_t, feat_a, feat_v


def detect_use_cross(args_dnr):
    """
    Detect whether cross-modal features were used during training
    by inspecting the backbone's expected input dimension.

    3 * divide_dim → no cross features
    7 * divide_dim → with cross features
    """
    emb_dim_t  = args_dnr.embedding_dim[args_dnr.dataset]["t"]
    divide_dim = args_dnr.divide_dim

    if divide_dim == 0:
        return False

    factor = emb_dim_t // divide_dim
    return factor == 7


def apply_smurf_to_data(data, smurf_model, args_dnr, device):
    """
    Run SMURF on one data batch and update data["tensor"] in-place
    with the correct DnR features (with or without cross-modal terms).

    Parameters
    ----------
    data        : dict  one batch from Dataloader
    smurf_model : ThreeModalityModel (frozen)
    args_dnr    : Namespace with embedding_dim and divide_dim
    device      : torch.device

    Returns
    -------
    data        : updated in-place
    m1, m2, m3  : raw SMURF tuples (u, r, s) for further analysis
    """
    x1 = data["tensor"]['t']
    x2 = data["tensor"]['a']
    x3 = data["tensor"]['v']

    textf   = (x1.permute(1, 2, 0)).transpose(1, 2)
    audiof  = (x2.permute(1, 2, 0)).transpose(1, 2)
    visualf = (x3.permute(1, 2, 0)).transpose(1, 2)

    m1, m2, m3, _ = smurf_model(textf, audiof, visualf)

    divide_dim = m1[0].shape[-1]
    use_cross  = detect_use_cross(args_dnr)

    # Override divide_dim from actual SMURF output in case args is stale
    # Re-detect with actual divide_dim
    emb_dim_t = args_dnr.embedding_dim[args_dnr.dataset]["t"]
    if emb_dim_t % divide_dim == 0:
        factor    = emb_dim_t // divide_dim
        use_cross = (factor == 7)

    feat_t, feat_a, feat_v = build_dnr_features(m1, m2, m3, divide_dim, use_cross)

    data["tensor"]['t'] = feat_t.transpose(0, 1)
    data["tensor"]['a'] = feat_a.transpose(0, 1)
    data["tensor"]['v'] = feat_v.transpose(0, 1)

    return data, m1, m2, m3