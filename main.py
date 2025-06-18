from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch
from typing import cast
from lm_saliency import *
from pathlib import Path
from models.models import Example
from typing import List, Tuple, Union, Dict
from helpers.get_device import get_device
from helpers.build_input import build_inputs

# input = "Can you stop the dog from "
# input_tokens = tokenizer(input)["input_ids"]
# attention_ids = tokenizer(input)["attention_mask"]

# # Visualize the saliency of the input tokens


# target = "barking"
# foil = "crying"
# CORRECT_ID = tokenizer(" " + target)["input_ids"][0]
# FOIL_ID = tokenizer(" " + foil)["input_ids"][0]

# base_saliency_matrix, base_embd_matrix = saliency(model, input_tokens, attention_ids)
# saliency_matrix, embd_matrix = saliency(
#     model, input_tokens, attention_ids, foil=FOIL_ID
# )

# print(f"Base saliency matrix: {saliency_matrix}")

# # Input x gradient
# base_explanation = input_x_gradient(
#     base_saliency_matrix, base_embd_matrix, normalize=True
# )
# contra_explanation = input_x_gradient(saliency_matrix, embd_matrix, normalize=True)

# # Gradient norm
# base_explanation = l1_grad_norm(base_saliency_matrix, normalize=True)
# contra_explanation = l1_grad_norm(saliency_matrix, normalize=True)

# # # Erasure
# base_explanation = erasure_scores(model, input_tokens, attention_ids, normalize=True)
# contra_explanation = erasure_scores(
#     model, input_tokens, attention_ids, correct=CORRECT_ID, foil=FOIL_ID, normalize=True
# )

# visualize(
#     np.array(base_explanation),
#     tokenizer,
#     [input_tokens],
#     print_text=True,
#     title=f"Why did the model predict {target}?",
# )


# visualize(
#     np.array(contra_explanation),
#     tokenizer,
#     [input_tokens],
#     print_text=True,
#     title=f"Why did the model predict {target} instead of {foil}?",
# )

jsonl_path = Path("data/a_dataset_for_testing.jsonl")


def explanation_vectors(
    ex: Example,
    ctx_ids: Union[List[int], Tuple[List[int], List[int]]],
    mask_ids: List[int],
    tgt: int,
    foil: int,
    model: GPT2LMHeadModel,
) -> Dict[str, np.ndarray]:
    """
    Compute GN, GI, and Erasure vectors for the example.
    For two-prefix items we subtract bad-context saliency from good-context saliency.
    """
    if ex.is_one_prefix:
        grads, embds = saliency(model, ctx_ids, mask_ids, correct=tgt, foil=foil)
        GN = l1_grad_norm(grads)
        GI = input_x_gradient(grads, embds)
        ER = erasure_scores(model, ctx_ids, mask_ids, correct=tgt, foil=foil)

    else:  # two-prefix
        good_ids, bad_ids = ctx_ids
        g_gr, g_em = saliency(model, good_ids, mask_ids, correct=tgt)
        b_gr, b_em = saliency(model, bad_ids, mask_ids, correct=tgt)
        grads = g_gr - b_gr
        embds = g_em - b_em
        GN = l1_grad_norm(grads)
        GI = input_x_gradient(grads, embds)
        ER = erasure_scores(model, good_ids, mask_ids, correct=tgt) - erasure_scores(
            model, bad_ids, mask_ids, correct=tgt
        )

    return {"GN": np.abs(GN), "GI": np.abs(GI), "E": np.abs(ER)}


def main():
    jsonl_path = "data/a_dataset_for_testing.jsonl"

    model = GPT2LMHeadModel.from_pretrained("gpt2").to(get_device())
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

    with Path(jsonl_path).open() as fh:
        for line in fh:
            ex = Example.model_validate_json(line)
            ctx, mask, tgt, foil = build_inputs(ex, tokenizer)
            S = explanation_vectors(ex, ctx, mask, tgt, foil, model)

            # quick demo: print top-5 tokens for each method
            decoded = tokenizer.convert_ids_to_tokens(
                ctx[0] if ex.is_two_prefix else ctx
            )
            print(f"\n{ex.UID}: {ex.sentence_good}")
            for name, vec in S.items():
                top = sorted(enumerate(vec), key=lambda x: -abs(x[1]))[:5]
                toks = " | ".join(f"{decoded[i]}:{vec[i]:.2f}" for i, _ in top)
                print(f"  {name} → {toks}")


if __name__ == "__main__":
    main()
