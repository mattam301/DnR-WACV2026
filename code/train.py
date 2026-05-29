from tqdm import tqdm
import copy
import argparse
import os
import time
from datetime import datetime
from comm_loss import CoMMLoss
from backbone.simple_backbone import SimpleMultimodalModel


try:
    from comet_ml import Experiment
except ImportError:
    Experiment = None

# Create a folder for saving plots
os.makedirs("viz_embeddings", exist_ok=True)


import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn import metrics

from model import Model
from dataloader import base_dataset_name, infer_feature_dims, load_affect_dataset, load_iemocap, load_meld, Dataloader
from optimizer import Optimizer
from utils import set_seed, weight_visualize, info_nce_loss, visualize_embeddings, forward_masked_augmented
import json
# from divide_decomp import ThreeModalityModel, compute_corr_loss
from new_divide_decomp import ThreeModalityModel, compute_corr_loss
from dnr_kld_analysis import run_dnr_kld_analysis

def smurf_pretrain(smurf_model: ThreeModalityModel, train_set: Dataloader, args):
    """
    Pretrain SMURF with two optimizers / two effective learning rates.

    Why this version?
    -----------------
    In practice, the standard NLL head usually converges much faster than:
      - disentanglement loss (corr)
      - synergy loss

    If all parameters share one optimizer / one LR, the model often finds
    a good task solution first, while u/r/s disentanglement keeps moving
    very slowly.  This function separates the parameter groups:

      main optimizer:
        - branch LayerNorm
        - unique heads
        - shared heads
        - fusion classifier

      synergy optimizer:
        - synergy heads in each branch
        - synergy module itself

    This makes the synergy-related subnetwork learn faster without forcing
    the whole SMURF model to use a too-large LR.

    Assumptions
    -----------
    - smurf_model has attributes:
        branch_t, branch_a, branch_v, fusion, synergy
    - each branch has:
        norm, fc_unique, fc_shared, fc_synergy
    - labels in data["label_tensor"] are already flattened to match
      the masked utterance logits used by NLLLoss.
    """

    m1, m2, m3, final_repr = None, None, None, None
    device = args.device

    if not (args.use_divide and args.use_refine):
        return m1, m2, m3, final_repr, smurf_model

    print("Pretraining SMURF module...")

    # ------------------------------------------------------------------
    # Hyperparameters / fallbacks
    # ------------------------------------------------------------------
    base_lr      = getattr(args, "learning_rate", 2e-4)
    weight_decay = getattr(args, "weight_decay", 1e-8)
    lambda_corr  = getattr(args, "lambda_corr", 5.0)
    lambda_syn   = getattr(args, "lambda_syn", 0.5)

    # Main idea of the "first fix":
    # give the synergy-related params a larger LR
    syn_lr_mult  = getattr(args, "smurf_syn_lr_mult", 3.0)

    main_lr = base_lr
    syn_lr  = base_lr * syn_lr_mult

    # ------------------------------------------------------------------
    # Build parameter groups
    # ------------------------------------------------------------------
    branches = [smurf_model.branch_t, smurf_model.branch_a, smurf_model.branch_v]

    main_params = []
    syn_params  = []

    for branch in branches:
        # Main params: norm + unique + shared
        main_params.extend(list(branch.norm.parameters()))
        main_params.extend(list(branch.fc_unique.parameters()))
        main_params.extend(list(branch.fc_shared.parameters()))

        # Synergy params: synergy head only
        syn_params.extend(list(branch.fc_synergy.parameters()))

    # Fusion head belongs to the "main" path
    main_params.extend(list(smurf_model.fusion.parameters()))

    # Synergy module gets the higher-LR optimizer
    syn_params.extend(list(smurf_model.synergy.parameters()))

    # Safety check: no overlap between optimizers
    main_param_ids = {id(p) for p in main_params}
    syn_param_ids  = {id(p) for p in syn_params}
    overlap = main_param_ids.intersection(syn_param_ids)
    if len(overlap) > 0:
        raise RuntimeError("SMURF pretrain optimizers have overlapping parameter groups.")

    # ------------------------------------------------------------------
    # Build optimizers
    # ------------------------------------------------------------------
    def make_torch_optimizer(params, lr):
        opt_name = args.optimizer.lower()
        if opt_name == "adam":
            return torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
        elif opt_name == "adamw":
            return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
        elif opt_name == "sgd":
            return torch.optim.SGD(params, lr=lr, weight_decay=weight_decay, momentum=0.9)
        elif opt_name == "rmsprop":
            return torch.optim.RMSprop(params, lr=lr, weight_decay=weight_decay)
        else:
            raise ValueError(f"Unsupported optimizer for SMURF pretrain: {args.optimizer}")

    optim_main = make_torch_optimizer(main_params, main_lr)
    optim_syn  = make_torch_optimizer(syn_params,  syn_lr)

    criterion = nn.NLLLoss()
    smurf_model.to(device)

    for epoch in range(args.pretrain_epochs):
        smurf_model.train()

        for idx in (pbar := tqdm(range(len(train_set)), desc=f"SMURF Epoch {epoch+1}")):
            optim_main.zero_grad(set_to_none=True)
            optim_syn.zero_grad(set_to_none=True)

            data = train_set[idx]
            for k, v in data.items():
                if k == "utterance_texts":
                    continue
                if k == "tensor":
                    for m, feat in data[k].items():
                        data[k][m] = feat.to(device)
                else:
                    data[k] = v.to(device)

            labels = data["label_tensor"]

            # Raw modality features from dataloader
            textf   = data["tensor"]['t']
            audiof  = data["tensor"]['a']
            visualf = data["tensor"]['v']

            # Match SMURF expected format: [seq, batch, dim]
            textf   = (textf.permute(1, 2, 0)).transpose(1, 2)
            audiof  = (audiof.permute(1, 2, 0)).transpose(1, 2)
            visualf = (visualf.permute(1, 2, 0)).transpose(1, 2)

            # ----------------------------------------------------------
            # Forward
            # ----------------------------------------------------------
            m1, m2, m3, final_repr = smurf_model(textf, audiof, visualf)

            # ----------------------------------------------------------
            # Correlation / disentanglement loss
            # ----------------------------------------------------------
            corr_loss, L_uncor, L_cor = compute_corr_loss(
                m1, m2, m3, data["length"]
            )

            # ----------------------------------------------------------
            # Synergy loss
            # ----------------------------------------------------------
            # Assumes labels are already flattened exactly like the NLL targets
            L_syn, L_joint, L_guard = smurf_model.compute_synergy_loss(
                m1, m2, m3,
                labels=labels,
                lengths=data["length"],
            )

            # ----------------------------------------------------------
            # Visualisation
            # ----------------------------------------------------------
            if epoch % 10 == 0 and idx == 0:
                visualize_embeddings(m1, m2, m3, epoch, method="pca")
                visualize_embeddings(m1, m2, m3, epoch, method="tsne")

            # ----------------------------------------------------------
            # Main NLL loss from SMURF fusion head
            # final_repr is [seq, batch, n_classes]
            # Need to remove padding positions before NLL
            # ----------------------------------------------------------
            logit_smurf = final_repr.permute(1, 0, 2)   # [batch, seq, n_classes]
            masked_logits = []
            for i, L in enumerate(data["length"]):
                masked_logits.append(logit_smurf[i, :L])
            logit_smurf = torch.cat(masked_logits, dim=0)   # [sum(lengths), n_classes]

            prob_smurf = F.log_softmax(logit_smurf, dim=-1)
            nll = criterion(prob_smurf, labels)

            # ----------------------------------------------------------
            # Total loss
            # ----------------------------------------------------------
            loss = nll + lambda_corr * corr_loss + lambda_syn * L_syn

            if not torch.isfinite(loss):
                raise RuntimeError(
                    "Non-finite SMURF pretrain loss. "
                    f"nll={nll.item():.6g}, "
                    f"corr={corr_loss.item():.6g}, "
                    f"L_uncor={L_uncor.item():.6g}, "
                    f"L_cor={L_cor.item():.6g}, "
                    f"L_syn={L_syn.item():.6g}, "
                    f"L_joint={L_joint.item():.6g}, "
                    f"L_guard={L_guard.item():.6g}"
                )

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                smurf_model.parameters(),
                max_norm=args.grad_norm_max,
                norm_type=args.grad_norm
            )

            optim_main.step()
            optim_syn.step()

            if epoch % 50 == 0:
                pbar.set_description(
                    f"Epoch {epoch+1} | "
                    f"loss={loss.item():.4f} "
                    f"nll={nll.item():.4f} "
                    f"corr={corr_loss.item():.4f} "
                    f"syn={L_syn.item():.4f} "
                    f"joint={L_joint.item():.4f} "
                    f"guard={L_guard.item():.4f} "
                    f"lr_main={main_lr:.1e} "
                    f"lr_syn={syn_lr:.1e}"
                )

    return m1, m2, m3, final_repr, smurf_model

def generate_all_data_versions(self, data, smurf_model):
    data_versions = []
    x1 = data["tensor"]['t']
    x2 = data["tensor"]['a']
    x3 = data["tensor"]['v']
    textf   = (x1.permute(1, 2, 0)).transpose(1, 2)
    audiof  = (x2.permute(1, 2, 0)).transpose(1, 2)
    visualf = (x3.permute(1, 2, 0)).transpose(1, 2)
    smurf_model.eval()
    with torch.no_grad():
        m1, m2, m3, final_repr = smurf_model(textf, audiof, visualf)

    # ── Compute cross-modal relationship features ─────────────────
    # These capture agreement/contradiction between modalities
    # using the already-decomposed u, r, s vectors.
    #
    # For each pair (i, j):
    #   agreement:    u_i * u_j     (element-wise, high when similar)
    #   discrepancy:  u_i - u_j     (high when different)
    #   shared_diff:  r_i - r_j     (should be ~0 if alignment worked)
    #
    # We concatenate these alongside the standard [u, r, s]
    # so the backbone can explicitly see cross-modal relationships.
    #
    # Why this works for sarcasm:
    #   u_t encodes "text says positive"
    #   u_a encodes "audio is flat/negative"
    #   u_t - u_a is LARGE → explicit contradiction signal
    #   The backbone no longer needs to re-discover this.

    u1, r1, s1 = m1
    u2, r2, s2 = m2
    u3, r3, s3 = m3

    # Cross-modal features for each modality
    # Modality t gets its relationship with a and v
    cross_t = torch.cat([u1 - u2, u1 - u3, u1 * u2, u1 * u3], dim=-1)
    cross_a = torch.cat([u2 - u1, u2 - u3, u2 * u1, u2 * u3], dim=-1)
    cross_v = torch.cat([u3 - u1, u3 - u2, u3 * u1, u3 * u2], dim=-1)

    # ── Original data ─────────────────────────────────────────────
    ori_data = copy.deepcopy(data)
    ori_data["tensor"] = {
        "t": torch.cat([u1, r1, s1, cross_t], dim=-1),
        "a": torch.cat([u2, r2, s2, cross_a], dim=-1),
        "v": torch.cat([u3, r3, s3, cross_v], dim=-1),
    }
    data_versions.append(ori_data)

    # ── Masked data ───────────────────────────────────────────────
    for i, mod_key in enumerate(self.modalities):
        masked_data = {}
        for j, mod_key2 in enumerate(self.modalities):
            if i == j:
                masked_data[mod_key2] = ori_data["tensor"][mod_key2]
            else:
                masked_data[mod_key2] = torch.zeros_like(
                    ori_data["tensor"][mod_key2]
                )
        data_versions.append({
            "tensor": masked_data,
            "length": data["length"],
            "label_tensor": data["label_tensor"],
            "speaker_tensor": data["speaker_tensor"],
        })

    # ── Augmented data (noise on r only) ──────────────────────────
    for noise_std in [0.2, 0.1]:
        aug_r1 = r1 + torch.randn_like(r1) * noise_std
        aug_r2 = r2 + torch.randn_like(r2) * noise_std
        aug_r3 = r3 + torch.randn_like(r3) * noise_std

        # Cross-modal features stay the same (computed from u, not r)
        aug_data = copy.deepcopy(data)
        aug_data["tensor"] = {
            "t": torch.cat([u1, aug_r1, s1, cross_t], dim=-1),
            "a": torch.cat([u2, aug_r2, s2, cross_a], dim=-1),
            "v": torch.cat([u3, aug_r3, s3, cross_v], dim=-1),
        }
        data_versions.append(aug_data)

    # ── Transpose back ────────────────────────────────────────────
    for version in data_versions:
        version["tensor"] = {
            m: (feat.transpose(0, 1)
                if isinstance(feat, torch.Tensor)
                else (feat[0].transpose(0, 1),
                      feat[1].transpose(0, 1),
                      feat[2].transpose(0, 1)))
            for m, feat in version["tensor"].items()
        }

    return data_versions
def train(model: nn.Module,
          train_set: Dataloader,
          dev_set: Dataloader,
          test_set: Dataloader,
          optimizer,
          logger: Experiment,
          args):

    modalities = args.modalities
    device = args.device
    dev_f1, loss = [], []
    best_dev_f1 = None
    best_test_f1 = None
    best_state = None
    best_epoch = None

    optimizer.set_parameters(model.parameters(), args.optimizer)

    early_stopping_count = 0
    if args.use_divide and args.use_refine:
        input_dim = args.input_embedding_dim[args.dataset]
        smurf_model = ThreeModalityModel(
            t_dim=input_dim["t"],
            a_dim=input_dim["a"],
            v_dim=input_dim["v"],
            out_dim=args.divide_dim,
            final_dim=len(args.dataset_label_dict[args.dataset]),
        ).to(device)
    else:
        smurf_model = None
    ## representation pretraining (input: representations of 3 modalities, output: new representations of 3 modalities with 3 components decomposed: unique, shared1, shared2)
    if args.use_divide and args.use_refine:
        _, _, _, _, smurf_model = smurf_pretrain(smurf_model, train_set, args)
        print("SMURF module pretrained.")
        # Save pretrained SMURF
        torch.save(smurf_model.state_dict(), "smurf_pretrained.pt")
        print("✅ SMURF pretrained model saved to smurf_pretrained.pt")
        smurf_model.eval()
        for param in smurf_model.parameters():
            param.requires_grad_(False)
    
    ## legacy training module/backbone
    for epoch in range(args.epochs):
        start_time = time.time()
        total_take_sample = 0
        total_sample = 0
        loss = "NaN"
        _loss = 0
        loss_m = {m: 0 for m in modalities}
        
        model.train()
        train_set.shuffle()

        for idx in (pbar := tqdm(range(len(train_set)), desc=f"Epoch {epoch+1}, Train loss {loss}")):
            model.zero_grad()

            data = train_set[idx]
            for k, v in data.items():
                if k == "utterance_texts":
                    continue
                if k == "tensor":
                    for m, feat in data[k].items():
                        data[k][m] = feat.to(device)
                else:
                    data[k] = v.to(device)
            labels = data["label_tensor"]
            sample_idx = data["uid"]
            
            # Generate all data versions (include original, 3 masked, 2 augmented)
            data_versions = generate_all_data_versions(model, data, smurf_model) if args.use_refine else [data]
            ori_data = data_versions[0]
            masked_data_versions = data_versions[1:1+len(modalities)]
            augmented_data_versions = data_versions[1+len(modalities):]
            
            ###################### DEV: stack all versions for speed up
            # -------- STACKING STEP --------
            if args.use_refine:
                if args.use_hightway:
                    rep_masked, rep_augmented = forward_masked_augmented(model, data_versions)
                else:
                    # masked outputs
                    rep_masked = []
                    for masked_data in masked_data_versions:
                        _, _, rep_m = model.net(masked_data)
                        # print("Masked representation inspect",rep_m)
                        # print(rep_m.shape)
                        rep_masked.append(rep_m) 
                    # augmented outputs
                    rep_augmented = []
                    for augmented_data in augmented_data_versions:
                        _, _, rep_a = model.net(augmented_data)
                        # print("augmented representation inspect",rep_a)
                        # print(rep_a.shape)
                        rep_augmented.append(rep_a)
            
            # Compute comm loss
                comm_loss = 0
                prototype = -1
                for rep_m in rep_masked:
                    for rep_a in rep_augmented:
                        comm_loss_value = info_nce_loss(rep_m, rep_a, temperature=0.7)
                        comm_loss += comm_loss_value
                comm_loss = comm_loss / 2
                comm_loss_aug = info_nce_loss(rep_augmented[0], rep_augmented[1], temperature=0.7)
                comm_loss += comm_loss_aug
            nll, ratio, take_samp, uni_nll = model.get_loss(ori_data)
            total_take_sample += take_samp
            total_sample += len(labels)
            loss = nll + (0.1 * comm_loss if args.use_refine else 0)
            # print(f"negative log likelihood: {nll.item()},comm loss: {comm_loss.item() if args.use_refine else 0}, total loss: {loss.item()}")
            _loss += loss.item()
            for m in modalities:
                loss_m[m] += uni_nll[m].item()

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(), max_norm=args.grad_norm_max, norm_type=args.grad_norm)

            optimizer.step()
            
            # pbar.set_description(f"Epoch {epoch+1}, Train loss {loss.item():,.4f}")

            del data
        if epoch % 10 == 0:
            end_time = time.time()
            print(
                f"[Epoch {epoch}] [Time: {end_time - start_time}]")
            for m in modalities:
                print(f'Ratio {m}: {ratio[m].item()}', end=" ")
        if args.use_cl:
            rate = total_take_sample / total_sample
            print(f"[Rate: {rate}, Threshold: {model.threshold}]")

        dev_f1, dev_acc, dev_loss = evaluate(model, smurf_model, dev_set, args, logger, test=False)
        if epoch % 10 == 0:    
            print(f"[Dev Loss: {dev_loss}]\n[Dev F1: {dev_f1}]\n[Dev Acc: {dev_acc}]")

        if args.use_cl:
            model.increase_threshold()

        if args.comet:
            logger.log_metric("train_loss", loss, epoch=epoch)
            logger.log_metric("dev_loss", dev_loss, epoch=epoch)
            logger.log_metric("dev_f1", dev_f1, epoch=epoch)
            logger.log_metric("dev_acc", dev_acc, epoch=epoch)
            logger.log_metric("train/loss", _loss / len(train_set), epoch=epoch)
            if args.use_cl:
                logger.log_metric("self-paced rate", rate)
                logger.log_metric("threshold", model.threshold)

            for m in modalities:
                logger.log_metric(f"ratio {m}", ratio[m], epoch=epoch)

        if best_dev_f1 is None or dev_f1 > best_dev_f1:
            best_dev_f1 = dev_f1
            best_test_f1, _, _ = evaluate(
                model, smurf_model, test_set, args, logger, test=False)
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            early_stopping_count = 0
        else:
            early_stopping_count += 1

        if early_stopping_count == args.early_stopping:
            print(f"Early stopping at epoch: {epoch+1}")
            break    

    # best model
    print(f"Best model at epoch: {best_epoch}")
    print(f"Best dev F1: {best_dev_f1}")
    model.load_state_dict(best_state)
    f1, acc, _ = evaluate(model, smurf_model, test_set, args, logger, test=True)
    print(f"Best test F1: {f1}")
    print(f"Best test Acc: {acc}")

    # Optional post-hoc interpretability pass for DnR.
    # It is intentionally run after best-model evaluation so the KLD reports
    # explain the same checkpoint used for the final test metrics.
    if args.run_kld_analysis:
        if not (args.use_divide and args.use_refine):
            raise ValueError("--run_kld_analysis requires --use_divide --use_refine.")
        summary = run_dnr_kld_analysis(
            model, smurf_model, test_set, args, split_name="test"
        )
        print(f"KLD analysis saved to: {summary['files']}")

    if args.comet:
        logger.log_metric("best_test_f1", f1, epoch=epoch)
        logger.log_metric("best_test_acc", acc, epoch=epoch)
        logger.log_metric("best_dev_f1", best_dev_f1, epoch=epoch)

    return best_dev_f1, best_test_f1, best_state


def evaluate(model, smurf_model, dataset, args, logger, test=True):
    criterion = nn.NLLLoss()
    device = args.device
    model.eval()
    label_dict = args.dataset_label_dict[args.dataset]
    labels_name = list(label_dict.keys())

    with torch.no_grad():
        golds, preds = [], []
        loss = 0
        for idx in range(len(dataset)):
            data = dataset[idx]
            for k, v in data.items():
                if k == "utterance_texts":
                    continue
                if k == "tensor":
                    for m, feat in data[k].items():
                        data[k][m] = feat.to(device)
                else:
                    data[k] = v.to(device)

            if args.use_divide and args.use_refine:
                x1 = data["tensor"]['t']
                x2 = data["tensor"]['a']
                x3 = data["tensor"]['v']
                textf   = (x1.permute(1, 2, 0)).transpose(1, 2)
                audiof  = (x2.permute(1, 2, 0)).transpose(1, 2)
                visualf = (x3.permute(1, 2, 0)).transpose(1, 2)
                m1, m2, m3, final_repr = smurf_model(textf, audiof, visualf)

                u1, r1, s1 = m1
                u2, r2, s2 = m2
                u3, r3, s3 = m3

                # Same cross-modal features as training
                cross_t = torch.cat([u1-u2, u1-u3, u1*u2, u1*u3], dim=-1)
                cross_a = torch.cat([u2-u1, u2-u3, u2*u1, u2*u3], dim=-1)
                cross_v = torch.cat([u3-u1, u3-u2, u3*u1, u3*u2], dim=-1)

                data["tensor"]['t'] = torch.cat(
                    [u1, r1, s1, cross_t], dim=-1
                ).transpose(0, 1)
                data["tensor"]['a'] = torch.cat(
                    [u2, r2, s2, cross_a], dim=-1
                ).transpose(0, 1)
                data["tensor"]['v'] = torch.cat(
                    [u3, r3, s3, cross_v], dim=-1
                ).transpose(0, 1)

            labels = data["label_tensor"]
            golds.append(labels.to("cpu"))
            prob, _, _ = model(data)
            nll = criterion(prob, labels)
            y_hat = torch.argmax(prob, dim=-1)
            preds.append(y_hat.detach().to("cpu"))
            loss += nll.item()

        golds = torch.cat(golds, dim=-1).numpy()
        preds = torch.cat(preds, dim=-1).numpy()
        loss /= len(dataset)
        f1  = metrics.f1_score(golds, preds, average="weighted")
        acc = metrics.accuracy_score(golds, preds)

        if test:
            print(metrics.classification_report(
                golds, preds, target_names=labels_name, digits=4))
            if args.comet:
                logger.log_confusion_matrix(
                    golds.tolist(), preds,
                    labels=list(labels_name), overwrite=True)

        return f1, acc, loss


def get_argurment():
    parser = argparse.ArgumentParser()
    # ________________________________ Logging Setting ______________________________________
    parser.add_argument(
        "--comet", action="store_true", default=False
    )
    parser.add_argument(
        "--comet_api", type=str, default="",
    )
    parser.add_argument(
        "--comet_workspace", type=str, default="",
    )
    parser.add_argument(
        "--project_name", type=str, default="",
    )
    
    # ________________________________ Trainning Setting ____________________________________
    parser.add_argument(
        "--name", type=str, default="default"
    )

    parser.add_argument(
        "--dataset",
        type=str,
        choices=[
            "iemocap", "meld", "mosi", "mosei", "humor", "sarcasm",
            "iemocap_coid", "meld_coid", "mosi_coid", "mosei_coid",
            "humor_coid", "sarcasm_coid",
        ],
        default="iemocap",
    )

    parser.add_argument(
        "--emotion",
        type=str,
        default=None,
    )

    parser.add_argument(
        "--devset_ratio", type=float, default=0.1
    )

    parser.add_argument(
        "--backbone", type=str, default="late_fusion",
        choices=["late_fusion", "mmgcn", "dialogue_gcn", "mm_dfn", "simple"],
    )

    parser.add_argument(
        "--modalities",
        type=str,
        choices=["atv", "at", "av", "tv", "a", "t", "v"],
        default="atv",
    )

    parser.add_argument(
        "--data_dir_path", type=str, default="data",
    )

    parser.add_argument(
        "--seed", default=12,
    )

    parser.add_argument(
        "--optimizer",
        type=str,
        choices=["sgd", "adam", "adamw", "rmsprop"],
        default="adam",
    )

    parser.add_argument(
        "--scheduler", type=str, choices="reduceLR", default="reduceLR",
    )

    parser.add_argument(
        "--learning_rate", type=float, default=0.0002,
    )

    parser.add_argument(
        "--weight_decay", type=float, default=1e-8,
    )

    parser.add_argument(
        "--early_stopping", type=int, default=-1,
    )

    parser.add_argument(
        "--batch_size", type=int, default=16,
    )

    parser.add_argument(
        "--epochs", type=int, default=50,
    )

    parser.add_argument(
        "--device", type=str, default="cuda", choices=["cuda", "cpu"]
    )

    parser.add_argument(
        "--modulation", action="store_true", default=False
    )

    parser.add_argument(
        "--alpha", type=float, default=0.5
    )


    parser.add_argument(
        "--normalize", action="store_true", default=False
    )


    parser.add_argument(
        "--grad_clipping", action="store_true", default=False,
    )

    parser.add_argument(
        "--grad_norm", type=float, default=2.0,
    )

    parser.add_argument(
        "--grad_norm_max", type=float, default=2.0,
    )

    # ________________________________ CL Setting ____________________________________

    parser.add_argument(
        "--use_cl", action="store_true", default=False,
    )
    parser.add_argument(
        "--regularizer", type=str, default="hard", choices=["hard", "soft"],
    )
    parser.add_argument(
        "--cl_threshold", type=float, default=0.4,
    )
    parser.add_argument(
        "--cl_growth", type=float, default=1.25,
    )

    # ________________________________ Model Setting ____________________________________

    parser.add_argument(
        "--encoder_modules", type=str, default="transformer", choices=["transformer"]
    )

    parser.add_argument(
        "--encoder_nlayers", type=int, default=2,
    )

    parser.add_argument(
        "--beta", type=float, default=0.7,
    )

    parser.add_argument(
        "--hidden_dim", type=int, default=200,
    )

    parser.add_argument(
        "--hidden2_dim", type=int, default=150, help="party's state in BiDDIN/DialogueRNN"
    )

    parser.add_argument(
        "--hidden3_dim", type=int, default=100, help="emotion's represent in BiDDIN/DialogueRNN"
    )

    parser.add_argument(
        "--hidden4_dim", type=int, default=100, help="linear's emotion's represent in BiDDIN/DialogueRNN"
    )
    
    parser.add_argument(
        "--D_att", type=int, default=100, help="concat attention in BiDDIN/DialogueRNN"
    )

    parser.add_argument(
        "--listener_state", action="store_true", default=False, help="for BiDDIN/DialogueRNN"
    )
    
    parser.add_argument(
        "--context_attention", type=str, default="simple", help="for BiDDIN/DialogueRNN"
    )

    parser.add_argument(
        "--drop_rate", type=float, default=0.5,
    )
    
    parser.add_argument(
        "--trans_head", type=int, default=1, help="number of head of transformer encoder"
    )

    parser.add_argument(
        "--d_state", type=int, default=128,
    )
    
    parser.add_argument(
        "--wp", type=int, default=2,
    )

    parser.add_argument(
        "--wf", type=int, default=2,
    )

    parser.add_argument(
        "--use_speaker", action="store_true", default=False,
    )
    parser.add_argument(
        "--use_refine", action="store_true", default=False,
    )
    parser.add_argument(
        "--use_divide", action="store_true", default=False,
    )
    parser.add_argument(
        "--plot_smurf_decomp", action="store_true", default=False,
    )
    parser.add_argument(
        "--use_hightway", action="store_true", default=False,
    )
    parser.add_argument(
        "--divide_dim", type=int, default=256,
        help="Per-component output dimension for Divide before concatenating U/R/S representations.",
    )
    parser.add_argument(
        "--pretrain_epochs", type=int, default=200,
        help="Number of Divide/SMURF pretraining epochs before backbone training.",
    )
    parser.add_argument(
        "--run_kld_analysis", action="store_true", default=False,
        help="Run post-hoc KLD analysis for DnR U/R/S components after testing.",
    )
    parser.add_argument(
        "--analysis_dir", type=str, default="analysis_outputs",
        help="Directory for KLD analysis CSV/JSON outputs.",
    )
    parser.add_argument(
        "--analysis_tau", type=float, default=1.0,
        help="Temperature used to convert U/R/S vectors into distributions.",
    )
    parser.add_argument(
    "--lambda_syn", type=float, default=0.5,
    help="Weight for the synergy loss (masked reconstruction + guard).",
    )
    parser.add_argument(
    "--lambda_masked", type=float, default=0.5,
    help="Weight for the margin-based masked reconstruction synergy loss.",
)
    parser.add_argument(
        "--lambda_guard", type=float, default=0.5,
        help="Weight for the synergy orthogonality guard loss.",
    )
    parser.add_argument(
    "--smurf_syn_lr_mult", type=float, default=3.0,
    help="LR multiplier for SMURF synergy-related parameters during pretraining."
)
    args, unknown = parser.parse_known_args()

    raw_embedding_dim = {
        "iemocap": {
            "a": 512,
            "t": 768,
            "v": 1024,
        },
        "mosei": {
            "a": 74,
            "t": 300,
            "v": 35,
        },
        "mosi": {
            "a": 5,
            "t": 300,
            "v": 20,
        },
        "humor": {
            "a": 81,
            "t": 300,
            "v": 371,
        },
        "sarcasm": {
            "a": 81,
            "t": 300,
            "v": 371,
        },
        "meld": {
            "a": 300,
            "t": 768,
            "v": 342,
        },
    }
    refined_embedding_dim = {m: 7 * args.divide_dim for m in ["a", "t", "v"]}
    args.input_embedding_dim = {}
    args.embedding_dim = {}
    for dataset in [
        "iemocap", "meld", "mosi", "mosei", "humor", "sarcasm",
        "iemocap_coid", "meld_coid", "mosi_coid", "mosei_coid",
        "humor_coid", "sarcasm_coid",
    ]:
        base = base_dataset_name(dataset)
        args.input_embedding_dim[dataset] = raw_embedding_dim[base]
        args.embedding_dim[dataset] = (
            refined_embedding_dim
            if args.use_refine
            else raw_embedding_dim[base]
        )

    args.dataset_label_dict = {
        "iemocap": {"hap": 0, "sad": 1, "neu": 2, "ang": 3, "exc": 4, "fru": 5},
        "iemocap_coid": {"hap": 0, "sad": 1, "neu": 2, "ang": 3, "exc": 4, "fru": 5},
        "iemocap_4": {"hap": 0, "sad": 1, "neu": 2, "ang": 3},
        "iemocap_4_coid": {"hap": 0, "sad": 1, "neu": 2, "ang": 3},
        "meld": {"neu": 0, "sup": 1, "fea": 2, "sad": 3, "joy": 4, "dis": 5, "ang": 6},
        "meld_coid": {"neu": 0, "sup": 1, "fea": 2, "sad": 3, "joy": 4, "dis": 5, "ang": 6},
        "mosi": {"neg": 0, "pos": 1},
        "mosi_coid": {"neg": 0, "pos": 1},
        "mosei": {"neg": 0, "pos": 1},
        "mosei_coid": {"neg": 0, "pos": 1},
        "humor": {"not_humor": 0, "humor": 1},
        "humor_coid": {"not_humor": 0, "humor": 1},
        "sarcasm": {"not_sarcasm": 0, "sarcasm": 1},
        "sarcasm_coid": {"not_sarcasm": 0, "sarcasm": 1},
        "mosei7": {
            "Strong Negative": 0,
            "Weak Negative": 1,
            "Negative": 2,
            "Neutral": 3,
            "Positive": 4,
            "Weak Positive": 5,
            "Strong Positive": 6, },
        "mosei2": {
            "Negative": 0,
            "Positive": 1, },
    }

    args.dataset_num_speakers = {
        "iemocap": 2,
        "iemocap_coid": 2,
        "iemocap_4": 2,
        "iemocap_4_coid": 2,
        "mosei7": 1,
        "mosei2": 1,
        "mosi": 1,
        "mosi_coid": 1,
        "mosei": 1,
        "mosei_coid": 1,
        "humor": 1,
        "humor_coid": 1,
        "sarcasm": 1,
        "sarcasm_coid": 1,
        "meld": 8,
        "meld_coid": 8,
    }

    if args.seed == "time":
        args.seed = int(datetime.now().timestamp())
    else:
        args.seed = int(args.seed)

    if not torch.cuda.is_available():
        args.device = "cpu"

    return args


def main(args):
    set_seed(args.seed)

    base_dataset = base_dataset_name(args.dataset)
    if base_dataset == "iemocap":
        data = load_iemocap()
    elif base_dataset == "meld":
        data = load_meld()
    elif base_dataset in ["mosi", "mosei", "humor", "sarcasm"]:
        data = load_affect_dataset(base_dataset)
    else:
        raise ValueError(f"Unsupported dataset: {args.dataset}")

    inferred_dims = infer_feature_dims(data)
    args.input_embedding_dim[args.dataset] = inferred_dims
    if not args.use_refine:
        args.embedding_dim[args.dataset] = inferred_dims

    train_set = Dataloader(data["train"], args)
    dev_set = Dataloader(data["dev"], args)
    test_set = Dataloader(data["test"], args)

    optim = Optimizer(args.learning_rate, args.weight_decay)


    # model = Model(args).to(args.device)

    CONVERSATIONAL_DATASETS = {
        "iemocap", "iemocap_coid", "iemocap_4", "iemocap_4_coid",
        "meld", "meld_coid",
    }

    base = base_dataset_name(args.dataset)
    if base in CONVERSATIONAL_DATASETS:
        model = Model(args).to(args.device)
    else:
        model = SimpleMultimodalModel(args).to(args.device) # Other multimodal's tasks haven't been prepared -> refer to this simple backbones
        print(f"Using SimpleMultimodalModel for non-conversational dataset: {args.dataset}")

    if args.comet:
        if Experiment is None:
            raise ImportError("Install comet_ml or run without --comet.")
        logger = Experiment(project_name=args.project_name,
                            api_key=args.comet_api,
                            workspace=args.comet_workspace,
                            auto_param_logging=False,
                            auto_metric_logging=False)
        logger.log_parameters(args)
    else:
        logger = None
    dev_f1, test_f1, state = train(
        model, train_set, dev_set, test_set, optim, logger, args)

    checkpoint_path = os.path.join("checkpoint", f"{args.dataset}_{args.use_refine}_best_f1.pt")
    # if not os.path.exists(checkpoint_path):
    #     print(checkpoint_path)
    #     os.makedirs(os.path.dirname(checkpoint_path))
    torch.save({"args": args, "state_dict": state}, checkpoint_path)


if __name__ == "__main__":
    args = get_argurment()
    print(args)
    main(args)
