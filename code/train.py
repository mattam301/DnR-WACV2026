from comet_ml import Experiment
from comet_ml.integration.pytorch import log_model
from tqdm import tqdm
import copy
import argparse
import os
import time
from datetime import datetime
from comm_loss import CoMMLoss 

# Create a folder for saving plots
os.makedirs("viz_embeddings", exist_ok=True)


import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn import metrics

from model import Model
from dataloader import load_iemocap, load_meld, Dataloader
from optimizer import Optimizer
from utils import set_seed, weight_visualize, info_nce_loss, visualize_embeddings, forward_masked_augmented
import json
from divide_decomp import ThreeModalityModel, compute_corr_loss

def smurf_pretrain(smurf_model: ThreeModalityModel, train_set: Dataloader, args):
    m1, m2, m3, final_repr = None, None, None, None
    device = args.device
    if args.use_divide and args.use_refine:
        print("Pretraining SMURF module...")
        optim = Optimizer(args.learning_rate, args.weight_decay)
        optim.set_parameters(smurf_model.parameters(), args.optimizer)
        smurf_model.to(device)
        for epoch in range(200):
            for idx in (pbar := tqdm(range(len(train_set)), desc=f"Epoch {epoch+1}")):
                smurf_model.zero_grad()
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
                textf = data["tensor"]['t']
                audiof = data["tensor"]['a']
                visualf = data["tensor"]['v']

                textf = (textf.permute(1, 2, 0)).transpose(1, 2)
                audiof = (audiof.permute(1, 2, 0)).transpose(1, 2)
                visualf = (visualf.permute(1, 2, 0)).transpose(1, 2)
                #
                m1, m2, m3, final_repr = smurf_model(textf, audiof, visualf)
                corr_loss, L_uncor, L_cor = compute_corr_loss(m1, m2, m3)
                # Compare tensors m1[0] and m1[2]
                if epoch % 10 == 0 and idx == 0:  # visualize once per 10 epochs, first batch
                    visualize_embeddings(m1, m2, m3, epoch, method="pca")
                    visualize_embeddings(m1, m2, m3, epoch, method="tsne")
                final_logits = final_repr
    
                # mask out padding and flatten (hot fix)
                logit_smurf = final_logits.permute(1, 0, 2)  # -> [batch, seq, n_classes]
                masked_logits = []
                for i, L in enumerate(data["length"]):  # lengths per dialogue
                    masked_logits.append(logit_smurf[i, :L])  # keep only valid utterances
                logit_smurf = torch.cat(masked_logits, dim=0)  # -> [sum(lengths), n_classes]
                # now prob_smurf matches joint/logit shape
                prob_smurf = F.log_softmax(logit_smurf, dim=-1)
                # predict loss
                criterion = nn.NLLLoss()
                nll = criterion(prob_smurf, labels)
                loss = nll + 5*corr_loss
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    smurf_model.parameters(), max_norm=args.grad_norm_max, norm_type=args.grad_norm)
                optim.step()
                pbar.set_description(f"Pretrained Epoch {epoch+1}, Pretrain loss {loss:,.4f}, Corr loss {L_cor:,.4f}, Uncorr loss {L_uncor:,.4f}")
                
    return m1, m2, m3, final_repr, smurf_model

def generate_all_data_versions(self, data, smurf_model):
        data_versions = []
        # data refinement
        x1 = data["tensor"]['t']
        x2 = data["tensor"]['a']
        x3 = data["tensor"]['v']
        textf = (x1.permute(1, 2, 0)).transpose(1, 2)
        audiof = (x2.permute(1, 2, 0)).transpose(1, 2)
        visualf = (x3.permute(1, 2, 0)).transpose(1, 2)
        m1, m2, m3, final_repr = smurf_model(textf, audiof, visualf)
        
        # Original data
        ori_data = copy.deepcopy(data)
        # concat instead of sum -> tensor dim x 3
        ori_data["tensor"] = {
            "t": torch.cat([m1[0], m1[1], m1[2]], dim=-1),
            "a": torch.cat([m2[0], m2[1], m2[2]], dim=-1),
            "v": torch.cat([m3[0], m3[1], m3[2]], dim=-1),
        }
        data_versions.append(ori_data)
        # Masked data (one modality left unmasked at a time)
        for i, m in enumerate(self.modalities):
            masked_data = {}
            for j, m in enumerate(self.modalities):
                if i == j:
                    masked_data[m] = ori_data["tensor"][m]
                else:
                    masked_data[m] = torch.zeros_like(ori_data["tensor"][m])
            data_versions.append({
                "tensor": masked_data,
                "length": data["length"],
                "label_tensor": data["label_tensor"],
                "speaker_tensor": data["speaker_tensor"]
            })
        augment1_1 = m1[1] + torch.randn_like(m1[1]) * 0.2
        augment1_2 = m2[1] + torch.randn_like(m2[1]) * 0.2
        augment1_3 = m3[1] + torch.randn_like(m3[1]) * 0.2
        augmented_data = copy.deepcopy(data)
        augmented_data["tensor"] = {
            "t": torch.cat([m1[0], augment1_1, m1[2]], dim=-1),
            "a": torch.cat([m2[0], augment1_2, m2[2]], dim=-1),
            "v": torch.cat([m3[0], augment1_3, m3[2]], dim=-1),
        }
        data_versions.append(augmented_data)

        augmented_data = copy.deepcopy(data)
        augment2_1 = m1[1] + torch.randn_like(m1[1]) * 0.1
        augment2_2 = m2[1] + torch.randn_like(m2[1]) * 0.1
        augment2_3 = m3[1] + torch.randn_like(m3[1]) * 0.1
        augmented_data["tensor"] = {
            "t": torch.cat([m1[0], augment2_1, m1[2]], dim=-1),
            "a": torch.cat([m2[0], augment2_2, m2[2]], dim=-1),
            "v": torch.cat([m3[0], augment2_3, m3[2]], dim=-1),
        }
        data_versions.append(augmented_data)
        
        # transpose all versions back to original shape
        for version in data_versions:
            version["tensor"] = {
                m: feat.transpose(0, 1) if isinstance(feat, torch.Tensor) else (feat[0].transpose(0,1), feat[1].transpose(0,1), feat[2].transpose(0,1))
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
    if args.dataset == "iemocap_coid":
        smurf_model = ThreeModalityModel(t_dim=768, a_dim=512, v_dim=1024, out_dim=256, final_dim=256).to(device)
    elif args.dataset == "meld_coid":
        smurf_model = ThreeModalityModel(t_dim=768, a_dim=300, v_dim=342, out_dim=256, final_dim=256).to(device)
    else:
        smurf_model = None
    ## representation pretraining (input: representations of 3 modalities, output: new representations of 3 modalities with 3 components decomposed: unique, shared1, shared2)
    if args.use_divide and args.use_refine:
        _, _, _, _, smurf_model = smurf_pretrain(smurf_model, train_set, args)
        print("SMURF module pretrained.")
        # Save pretrained SMURF
        torch.save(smurf_model.state_dict(), "smurf_pretrained.pt")
        print("✅ SMURF pretrained model saved to smurf_pretrained.pt")
    
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
            
            pbar.set_description(f"Epoch {epoch+1}, Train loss {loss.item():,.4f}")

            del data

        end_time = time.time()
        print(
            f"[Epoch {epoch}] [Time: {end_time - start_time}]")
        for m in modalities:
            print(f'Ratio {m}: {ratio[m].item()}', end=" ")
        print()
        if args.use_cl:
            rate = total_take_sample / total_sample
            print(f"[Rate: {rate}, Threshold: {model.threshold}]")

        dev_f1, dev_acc, dev_loss = evaluate(model, smurf_model, dev_set, args, logger, test=False)
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
                textf = (x1.permute(1, 2, 0)).transpose(1, 2)
                audiof = (x2.permute(1, 2, 0)).transpose(1, 2)
                visualf = (x3.permute(1, 2, 0)).transpose(1, 2)
                m1, m2, m3, final_repr = smurf_model(textf, audiof, visualf)
                textf = torch.cat([m1[0], m1[1], m1[2]], dim=-1)
                audiof = torch.cat([m2[0], m2[1], m2[2]], dim=-1)
                visualf = torch.cat([m3[0], m3[1], m3[2]], dim=-1)
                # update data tensor
                # check
                data["tensor"]['t'] = textf.transpose(0,1)
                data["tensor"]['a'] = audiof.transpose(0,1)
                data["tensor"]['v'] = visualf.transpose(0,1)
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
        f1 = metrics.f1_score(golds, preds, average="weighted")
        acc = metrics.accuracy_score(golds, preds)

        if test:
            print(metrics.classification_report(
                golds, preds, target_names=labels_name, digits=4))
            if args.comet:
                logger.log_confusion_matrix(
                    golds.tolist(), preds, labels=list(labels_name), overwrite=True)

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
        choices=["iemocap", "meld", "iemocap_coid", "meld_coid"],
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
        choices=["late_fusion", "mmgcn", "dialogue_gcn", "mm_dfn", "biddin"],
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

    args, unknown = parser.parse_known_args()

    args.embedding_dim = {
        "iemocap": {
            "a": 512,
            "t": 768,
            "v": 1024,
        },
        "mosei": {
            "a": 512,
            "t": 768,
            "v": 1024,
        },
        "meld": {
            "a": 300,
            "t": 768,
            "v": 342,
        },
        "iemocap_coid": { # all dim 768
            "a": 768,
            "t": 768,
            "v": 768,
        },
        "meld_coid": { # all dim 768
            "a": 768,
            "t": 768,
            "v": 768,
        },
    }

    args.dataset_label_dict = {
        "iemocap": {"hap": 0, "sad": 1, "neu": 2, "ang": 3, "exc": 4, "fru": 5},
        "iemocap_coid": {"hap": 0, "sad": 1, "neu": 2, "ang": 3, "exc": 4, "fru": 5},
        "iemocap_4": {"hap": 0, "sad": 1, "neu": 2, "ang": 3},
        "iemocap_4_coid": {"hap": 0, "sad": 1, "neu": 2, "ang": 3},
        "meld": {"neu": 0, "sup": 1, "fea": 2, "sad": 3, "joy": 4, "dis": 5, "ang": 6},
        "meld_coid": {"neu": 0, "sup": 1, "fea": 2, "sad": 3, "joy": 4, "dis": 5, "ang": 6},
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

    if "iemocap" in args.dataset:
        data = load_iemocap()
    if "meld" in args.dataset:
        data = load_meld()

    train_set = Dataloader(data["train"], args)
    dev_set = Dataloader(data["dev"], args)
    test_set = Dataloader(data["test"], args)

    optim = Optimizer(args.learning_rate, args.weight_decay)
    model = Model(args).to(args.device)

    if args.comet:
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

    # checkpoint_path = os.path.join("checkpoint", f"{args.dataset}_best_f1.pt")
    # if not os.path.exists(checkpoint_path):
    #     os.makedirs(os.path.dirname(checkpoint_path))
    # torch.save({"args": args, "state_dict": state}, checkpoint_path)


if __name__ == "__main__":
    args = get_argurment()
    print(args)
    main(args)
