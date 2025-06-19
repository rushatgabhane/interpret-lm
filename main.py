from pathlib import Path
import collections, warnings, numpy as np, torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel

from models.models import Example
from helpers.build_input import build_inputs
from helpers.metrics import dot_product, probes_needed, reciprocal_rank
from helpers.get_device import get_device
from helpers.gold_mask import evidence_mask  # NEW
from lm_saliency import saliency, input_x_gradient, l1_grad_norm, erasure_scores


def explanation_vectors(ex, ctx_ids, mask_ids, tgt, foil, model):
    if ex.is_one_prefix:
        g, e = saliency(model, ctx_ids, mask_ids, correct=tgt, foil=foil)
        return {
            "GN": np.abs(l1_grad_norm(g)),
            "GI": np.abs(input_x_gradient(g, e)),
            "E": np.abs(
                erasure_scores(model, ctx_ids, mask_ids, correct=tgt, foil=foil)
            ),
        }
    good_ids, bad_ids = ctx_ids
    g_grad, g_emb = saliency(model, good_ids, mask_ids, correct=tgt)
    b_grad, b_emb = saliency(model, bad_ids, mask_ids, correct=tgt)
    grad, emb = g_grad - b_grad, g_emb - b_emb
    return {
        "GN": np.abs(l1_grad_norm(grad)),
        "GI": np.abs(input_x_gradient(grad, emb)),
        "E": np.abs(
            erasure_scores(model, good_ids, mask_ids, correct=tgt)
            - erasure_scores(model, bad_ids, mask_ids, correct=tgt)
        ),
    }


def main():
    tok = GPT2Tokenizer.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("gpt2").to(get_device())

    metrics = collections.defaultdict(list)

    for line in Path("data/a_dataset_for_testing.jsonl").open():
        ex = Example.model_validate_json(line)
        ctx_ids, mask_ids, tgt_id, foil_id = build_inputs(ex, tok)

        # -------- gold evidence over *prefix* --------
        flat_ctx = ctx_ids if isinstance(ctx_ids, list) else ctx_ids[0]
        prefix_ids = flat_ctx[:-1]  # saliency’s prefix
        gold = evidence_mask(ex, prefix_ids, tok)
        if gold.sum() == 0:  # drop, like paper
            warnings.warn(f"Skipped (no gold mask): {ex.UID}")
            continue

        # -------- saliency explanations --------
        vecs = explanation_vectors(ex, ctx_ids, mask_ids, tgt_id, foil_id, model)

        # -------- aggregate metrics --------
        for name, v in vecs.items():
            L = min(len(v), len(gold))  # align lengths
            v, g = v[:L], gold[:L]
            if g.sum() == 0:  # shouldn’t happen, but safe
                continue
            metrics[(name, "DP")].append(dot_product(v, g))
            metrics[(name, "PN")].append(probes_needed(v, g))
            metrics[(name, "MRR")].append(reciprocal_rank(v, g))

    # -------- print results --------
    if not metrics:
        print("No examples scored.")
        return
    n = len(next(iter(metrics.values())))
    for m in ["DP", "PN", "MRR"]:
        print(f"\n=== {m} (n={n}) ===")
        for expl in ["GN", "GI", "E"]:
            print(f"{expl}: {np.mean(metrics[(expl,m)]):.3f}")


if __name__ == "__main__":
    main()
