from pathlib import Path
import collections, warnings, numpy as np, torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel

from models.models import Example
from helpers.build_input import build_inputs
from helpers.metrics import dot_product, probes_needed, reciprocal_rank
from helpers.get_device import get_device
from helpers.gold_mask import evidence_mask  # NEW
from lm_saliency import (
    saliency,
    input_x_gradient,
    l1_grad_norm,
    erasure_scores,
    visualize,
)
import numpy as np

tok = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2").to(get_device())

skip_count = 0
processed_count = 0


def explanation_vectors(ex, ctx_ids, mask_ids, tgt, foil, model):
    global skip_count
    global processed_count
    processed_count += 1
    if processed_count % 100 == 0:
        print(f"processed count: {processed_count}")
    if ex.is_one_prefix:
        g, e = saliency(model, ctx_ids, mask_ids, correct=tgt, foil=foil)
        if g.ndim != 2:
            skip_count += 1
            print(
                f"UID: {ex.UID}, skipping — unexpected grad shape: {g.shape} skip count: {skip_count}, done count: {processed_count - skip_count}"
            )
            return None

        return {
            "GN": np.abs(l1_grad_norm(g, normalize=True)),
            "GI": np.abs(input_x_gradient(g, e, normalize=True)),
            "E": np.abs(
                erasure_scores(
                    model, ctx_ids, mask_ids, correct=tgt, foil=foil, normalize=True
                )
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
        if vecs is None:
            continue

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


def visualize_explaination():
    input = "Can you stop the dog from "
    input_tokens = tok(input)['input_ids']
    attention_ids = tok(input)['attention_mask']

    target = "barking"
    foil = "crying"
    CORRECT_ID = tok(" "+ target)['input_ids'][0]
    FOIL_ID = tok(" "+ foil)['input_ids'][0]

    base_saliency_matrix, base_embd_matrix = saliency(model, input_tokens, attention_ids)
    saliency_matrix, embd_matrix = saliency(model, input_tokens, attention_ids, foil=FOIL_ID)

    # Input x gradient
    base_explanation = input_x_gradient(base_saliency_matrix, base_embd_matrix, normalize=True)
    contra_explanation = input_x_gradient(saliency_matrix, embd_matrix, normalize=True)

    # Gradient norm
    base_explanation = l1_grad_norm(base_saliency_matrix, normalize=True)
    contra_explanation = l1_grad_norm(saliency_matrix, normalize=True)

    # Erasure
    base_explanation = erasure_scores(model, input_tokens, attention_ids, normalize=True)
    contra_explanation = erasure_scores(model, input_tokens, attention_ids, correct=CORRECT_ID, foil=FOIL_ID, normalize=True)

    visualize(np.array(base_explanation), tok, [input_tokens], print_text=True, title=f"Why did the model predict {target}?")
    visualize(np.array(contra_explanation), tok, [input_tokens], print_text=True, title=f"Why did the model predict {target} instead of {foil}?")

if __name__ == "__main__":
    # main()
    visualize_explaination()
    print(f"\nSkipped {skip_count} examples with unexpected grad shapes.")
