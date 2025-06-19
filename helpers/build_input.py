# helpers/build_input.py
from models.models import Example
from typing import List, Tuple, Union
from transformers import GPT2Tokenizer


def build_inputs(ex: Example, tok: GPT2Tokenizer) -> Tuple[
    Union[List[int], Tuple[List[int], List[int]]],  # ctx_ids
    List[int],  # attention mask
    int,  # target id
    int,  # foil id
]:
    """
    Return (ctx_ids, mask_ids, target_id, foil_id).

    * One-prefix → single ctx_ids list.
    * Two-prefix → tuple (good_ids, bad_ids) **truncated to common length**
                   so that saliency(good) and saliency(bad) are broadcast-safe.
    """
    # ----------  1) ONE-PREFIX  ----------
    if ex.is_one_prefix:
        ctx_txt = f"{ex.one_prefix_prefix} {ex.one_prefix_word_good}"
        ctx_ids = tok(ctx_txt, add_special_tokens=False)["input_ids"]

        tgt_id = tok(f" {ex.one_prefix_word_good}", add_special_tokens=False)[
            "input_ids"
        ][0]
        foil_id = tok(f" {ex.one_prefix_word_bad}", add_special_tokens=False)[
            "input_ids"
        ][0]

        mask = [1] * len(ctx_ids)
        return ctx_ids, mask, tgt_id, foil_id

    # ----------  2) TWO-PREFIX  ----------
    if ex.is_two_prefix:
        g_ctx = f"{ex.two_prefix_prefix_good} {ex.two_prefix_word}"
        b_ctx = f"{ex.two_prefix_prefix_bad} {ex.two_prefix_word}"

        g_ids = tok(g_ctx, add_special_tokens=False)["input_ids"]
        b_ids = tok(b_ctx, add_special_tokens=False)["input_ids"]

        # Align lengths so grad / emb shapes match
        L = min(len(g_ids), len(b_ids))
        g_ids, b_ids = g_ids[:L], b_ids[:L]

        tgt_id = tok(f" {ex.two_prefix_word}", add_special_tokens=False)["input_ids"][0]
        mask = [1] * L  # shared attention mask

        return (g_ids, b_ids), mask, tgt_id, tgt_id

    # ----------  3) Fallback  ----------
    raise ValueError(f"{ex.UID}: neither one-prefix nor two-prefix template")
