"""
analysis/run_all_analysis.py
"""

import argparse
import torch
import sys
import os
import copy

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from new_divide_decomp import ThreeModalityModel
from model import Model
from dataloader import (
    Dataloader, load_iemocap, load_meld,
    load_affect_dataset, base_dataset_name, infer_feature_dims
)
from utils import set_seed

from component_analysis   import run_component_analysis
from emotion_class_analysis import compute_per_class_improvement
from case_study_miner     import mine_cases


# ══════════════════════════════════════════════════════════════════════
#  Architecture inference helpers
# ══════════════════════════════════════════════════════════════════════

def infer_smurf_dims(state_dict):
    """Read SMURF architecture from checkpoint weight shapes."""
    t_dim     = state_dict["branch_t.norm.weight"].shape[0]
    a_dim     = state_dict["branch_a.norm.weight"].shape[0]
    v_dim     = state_dict["branch_v.norm.weight"].shape[0]
    out_dim   = state_dict["branch_t.fc_unique.weight"].shape[0]
    n_classes = state_dict["fusion.weight"].shape[0]
    return t_dim, a_dim, v_dim, out_dim, n_classes
def infer_backbone_dims(state_dict):
    """
    Read backbone architecture from checkpoint weight shapes.
    net.modality_encoder.{m}.lin_out.weight : [hidden_dim, input_dim_m]
    """
    hidden_dim = state_dict["net.modality_encoder.a.lin_out.weight"].shape[0]
    a_in = state_dict["net.modality_encoder.a.lin_out.weight"].shape[1]
    t_in = state_dict["net.modality_encoder.t.lin_out.weight"].shape[1]
    v_in = state_dict["net.modality_encoder.v.lin_out.weight"].shape[1]

    # Detect whether cross-modal features were used
    # by checking what multiple of divide_dim each input dim is
    # divide_dim = out_dim from SMURF = e.g. 256
    # 3 * 256 = 768  → no cross features
    # 7 * 256 = 1792 → with cross features
    print(f"  [Backbone] hidden_dim={hidden_dim} "
          f"a_in={a_in} t_in={t_in} v_in={v_in}")
    for factor, label in [(7, "with cross-modal features"),
                          (3, "without cross-modal features")]:
        if t_in % factor == 0:
            divide_dim_inferred = t_in // factor
            print(f"  → Detected: {label} "
                  f"(divide_dim={divide_dim_inferred})")
            break

    return hidden_dim, {"a": a_in, "t": t_in, "v": v_in}
# ══════════════════════════════════════════════════════════════════════
#  Loaders
# ══════════════════════════════════════════════════════════════════════

def load_smurf(checkpoint_path, device):
    """Load SMURF, inferring architecture from the checkpoint."""
    state_dict = torch.load(
        checkpoint_path, map_location=device, weights_only=False
    )
    t_dim, a_dim, v_dim, out_dim, n_classes = infer_smurf_dims(state_dict)

    print(f"  [SMURF] t={t_dim} a={a_dim} v={v_dim} "
          f"out={out_dim} classes={n_classes}")

    model = ThreeModalityModel(
        t_dim=t_dim, a_dim=a_dim, v_dim=v_dim,
        out_dim=out_dim, final_dim=n_classes,
    ).to(device)
    model.load_state_dict(state_dict)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    return model, {
        "t_dim": t_dim, "a_dim": a_dim, "v_dim": v_dim,
        "out_dim": out_dim, "n_classes": n_classes,
    }


def load_backbone(base_args, checkpoint_path, device):
    """
    Load backbone, inferring hidden_dim and input dims from checkpoint.
    This avoids mismatch when the saved model used different hyperparams
    than the current defaults.
    """
    state = torch.load(
        checkpoint_path, map_location=device, weights_only=False
    )
    # Support {"args": ..., "state_dict": ...} or plain state_dict
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]

    # Infer architecture
    hidden_dim, emb_dims = infer_backbone_dims(state)
    print(f"  [Backbone] hidden_dim={hidden_dim} "
          f"a={emb_dims['a']} t={emb_dims['t']} v={emb_dims['v']}")

    # Build args with correct dims
    args = copy.deepcopy(base_args)
    args.hidden_dim = hidden_dim
    args.embedding_dim[args.dataset] = emb_dims

    model = Model(args).to(device)
    model.load_state_dict(state)
    model.eval()
    return model, args   # return args too so caller knows actual dims


# ══════════════════════════════════════════════════════════════════════
#  Minimal args builder
# ══════════════════════════════════════════════════════════════════════

def build_base_args(dataset, backbone, seed, device):
    """
    Build a minimal args Namespace with all fields Model() needs.
    Architecture-specific fields (hidden_dim, embedding_dim) will be
    overridden by load_backbone() after reading the checkpoint.
    """
    raw_dims = {
        "iemocap":  {"a": 512,  "t": 768,  "v": 1024},
        "meld":     {"a": 300,  "t": 768,  "v": 342},
        "mosi":     {"a": 5,    "t": 300,  "v": 20},
        "mosei":    {"a": 74,   "t": 300,  "v": 35},
        "humor":    {"a": 81,   "t": 300,  "v": 371},
        "sarcasm":  {"a": 81,   "t": 300,  "v": 371},
    }
    label_dicts = {
        "iemocap":      {"hap":0,"sad":1,"neu":2,"ang":3,"exc":4,"fru":5},
        "iemocap_coid": {"hap":0,"sad":1,"neu":2,"ang":3,"exc":4,"fru":5},
        "meld":         {"neu":0,"sup":1,"fea":2,"sad":3,"joy":4,"dis":5,"ang":6},
        "meld_coid":    {"neu":0,"sup":1,"fea":2,"sad":3,"joy":4,"dis":5,"ang":6},
        "mosi":         {"neg":0,"pos":1},
        "mosi_coid":    {"neg":0,"pos":1},
        "mosei":        {"neg":0,"pos":1},
        "mosei_coid":   {"neg":0,"pos":1},
        "humor":        {"not_humor":0,"humor":1},
        "humor_coid":   {"not_humor":0,"humor":1},
        "sarcasm":      {"not_sarcasm":0,"sarcasm":1},
        "sarcasm_coid": {"not_sarcasm":0,"sarcasm":1},
    }
    num_speakers = {
        "iemocap":2,"iemocap_coid":2,
        "meld":8,  "meld_coid":8,
        "mosi":1,  "mosi_coid":1,
        "mosei":1, "mosei_coid":1,
        "humor":1, "humor_coid":1,
        "sarcasm":1,"sarcasm_coid":1,
    }

    base = base_dataset_name(dataset)

    return argparse.Namespace(
        dataset          = dataset,
        backbone         = backbone,
        seed             = seed,
        device           = device,
        modalities       = "atv",
        use_divide       = False,
        use_refine       = False,
        use_cl           = False,
        use_speaker      = False,
        use_hightway     = False,
        # These will be overridden by load_backbone():
        hidden_dim       = 200,
        hidden2_dim      = 150,
        hidden3_dim      = 100,
        hidden4_dim      = 100,
        D_att            = 100,
        drop_rate        = 0.3,
        encoder_modules  = "transformer",
        encoder_nlayers  = 2,
        trans_head       = 1,
        d_state          = 128,
        wp               = 2,
        wf               = 2,
        beta             = 0.7,
        alpha            = 0.5,
        listener_state   = False,
        context_attention= "simple",
        data_dir_path    = "data",
        batch_size       = 64,
        divide_dim       = 256,
        pretrain_epochs  = 200,
        lambda_syn       = 0.5,
        lambda_masked    = 0.5,
        lambda_guard     = 0.5,
        smurf_syn_lr_mult= 3.0,
        run_kld_analysis = False,
        # Placeholder dims — will be overridden:
        embedding_dim    = {dataset: raw_dims.get(base, raw_dims["iemocap"]),
                            base:    raw_dims.get(base, raw_dims["iemocap"])},
        input_embedding_dim = {dataset: raw_dims.get(base, raw_dims["iemocap"]),
                               base:    raw_dims.get(base, raw_dims["iemocap"])},
        dataset_label_dict   = label_dicts,
        dataset_num_speakers = num_speakers,
    )


# ══════════════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="DnR post-hoc analysis suite")
    parser.add_argument("--dataset",          type=str, required=True)
    parser.add_argument("--backbone",         type=str, default="mmgcn")
    parser.add_argument("--smurf_checkpoint", type=str, required=True)
    parser.add_argument("--raw_checkpoint",   type=str, required=True)
    parser.add_argument("--dnr_checkpoint",   type=str, required=True)
    parser.add_argument("--output_dir",       type=str, default="analysis_outputs")
    parser.add_argument("--seed",             type=int, default=42)
    parser.add_argument("--device",           type=str, default="cuda")
    parser.add_argument("--n_cases",          type=int, default=15)
    cli = parser.parse_args()

    device = torch.device(cli.device if torch.cuda.is_available() else "cpu")
    set_seed(cli.seed)
    os.makedirs(cli.output_dir, exist_ok=True)

    # ── Load data ─────────────────────────────────────────────────────────
    base = base_dataset_name(cli.dataset)
    print(f"Loading {cli.dataset}...")
    if base == "iemocap":
        data = load_iemocap()
    elif base == "meld":
        data = load_meld()
    else:
        data = load_affect_dataset(base)

    inferred_dims = infer_feature_dims(data)
    print(f"  Inferred feature dims: {inferred_dims}")

    # ── Build base args ───────────────────────────────────────────────────
    base_args = build_base_args(cli.dataset, cli.backbone, cli.seed, str(device))
    base_args.input_embedding_dim[cli.dataset] = inferred_dims
    base_args.input_embedding_dim[base]        = inferred_dims

    # ── Load SMURF ────────────────────────────────────────────────────────
    print("\nLoading SMURF checkpoint...")
    smurf_model, smurf_dims = load_smurf(cli.smurf_checkpoint, device)

    # Sanity check: SMURF must match the dataset's raw feature dims
    for m_key, dim_key in [("a","a_dim"), ("t","t_dim"), ("v","v_dim")]:
        expected = inferred_dims[m_key]
        got      = smurf_dims[dim_key]
        if expected != got:
            raise ValueError(
                f"SMURF checkpoint {dim_key}={got} but dataset "
                f"{m_key}_dim={expected}. "
                f"Did you use the wrong smurf_checkpoint?"
            )
    print("  SMURF dims match dataset ✓")

    # ── Load raw backbone ─────────────────────────────────────────────────
    print("\nLoading raw backbone checkpoint...")
    model_raw, args_raw = load_backbone(
        base_args, cli.raw_checkpoint, device
    )
    args_raw.use_divide = False
    args_raw.use_refine = False

    # ── Load DnR backbone ─────────────────────────────────────────────────
    print("\nLoading DnR backbone checkpoint...")
    model_dnr, args_dnr = load_backbone(
        base_args, cli.dnr_checkpoint, device
    )
    args_dnr.use_divide = True
    args_dnr.use_refine = True

    # ── Build test sets ───────────────────────────────────────────────────
    # Use args_raw for the test set (raw feature dims match dataloader)
    test_set = Dataloader(data["test"], args_raw)

    # ── Analysis 1: Component properties ─────────────────────────────────
    print("\n" + "="*60)
    print("ANALYSIS 1: Component Properties")
    print("="*60)
    run_component_analysis(
        smurf_model = smurf_model,
        test_set     = test_set,
        args        = args_raw,
        output_dir  = os.path.join(cli.output_dir, "components"),
        device      = device,
    )

    # ── Analysis 2: Per-class improvement ────────────────────────────────
    print("\n" + "="*60)
    print("ANALYSIS 2: Per-Class Improvement")
    print("="*60)
    compute_per_class_improvement(
        model_raw   = model_raw,
        model_dnr   = model_dnr,
        smurf_model = smurf_model,
        test_set    = test_set,
        args_raw    = args_raw,
        args_dnr    = args_dnr,
        device      = device,
        output_dir  = os.path.join(cli.output_dir, "per_class"),
    )

    # ── Analysis 3: Case mining ───────────────────────────────────────────
    print("\n" + "="*60)
    print("ANALYSIS 3: Case Mining")
    print("="*60)
    mine_cases(
        model_raw        = model_raw,
        model_dnr        = model_dnr,
        smurf_model      = smurf_model,
        test_set         = test_set,
        args             = args_dnr,
        device           = device,
        output_dir       = os.path.join(cli.output_dir, "cases"),
        n_cases_per_type = cli.n_cases,
    )

    print(f"\n{'='*60}")
    print(f"All analyses complete. Results in: {cli.output_dir}")
    print(f"{'='*60}")

    # ── Analysis 4: Explainability ────────────────────────────────────────
    from explainability_analysis import run_explainability_analysis

    print("\n" + "="*60)
    print("ANALYSIS 4: Explainability")
    print("="*60)
    run_explainability_analysis(
        smurf_model = smurf_model,
        model_raw   = model_raw,
        model_dnr   = model_dnr,
        test_set    = test_set,
        args        = args_dnr,
        args_raw    = args_raw,
        device      = device,
        output_dir  = os.path.join(cli.output_dir, "explainability"),
    )


if __name__ == "__main__":
    main()