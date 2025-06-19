"""
Gold-evidence extraction for the 4 remaining BLiMP phenomena
(after removing every example whose UID starts with "anaphor_").

Dependencies:
    pip install "spacy==3.5.4" numpy==1.26.4
    python -m spacy download en_core_web_sm
"""

from __future__ import annotations
import numpy as np, spacy

# ------------------------------------------------------------
# spaCy     (only for dependency parsing)
# ------------------------------------------------------------
nlp = spacy.load("en_core_web_sm")

# ------------------------------------------------------------
# Lexical inventories needed by the rules
# ------------------------------------------------------------
NPI = {"ever", "even", "either", "any", "yet"}


# ------------------------------------------------------------
# Helper – map a word to the first GPT-2 sub-token index
# ------------------------------------------------------------
def first_subtok_idx(word: str, prefix_ids: list[int], tok) -> int | None:
    for form in (word, " " + word):  # sentence-initial vs middle
        ids = tok.encode(form, add_special_tokens=False)
        for i in range(len(prefix_ids) - len(ids) + 1):
            if prefix_ids[i : i + len(ids)] == ids:
                return i
    return None


# ------------------------------------------------------------
# Main function used in main.py
# ------------------------------------------------------------
def evidence_mask(ex, prefix_ids: list[int], tokenizer):
    """
    Returns a binary vector (len = |prefix_ids|) with 1-marks on tokens that
    should count as evidence for the model’s decision, **excluding anaphor
    paradigms** (they are filtered out upstream).
    """
    gold = np.zeros(len(prefix_ids), dtype=int)
    doc = nlp(ex.sentence_good)

    # 1 ─ Subject-verb agreement → nominal subject
    if ex.UID.startswith("subject_verb_agreement"):
        root = next((t for t in doc if t.dep_ == "ROOT"), None)
        subj = next(
            (c for c in root.children if c.dep_ in {"nsubj", "nsubjpass"}), None
        )
        if subj:
            j = first_subtok_idx(subj.text, prefix_ids, tokenizer)
            if j is not None:
                gold[j] = 1

    # 2 ─ Determiner-noun agreement → determiner token
    elif ex.UID.startswith("determiner_noun"):
        det = next((t for t in doc if t.dep_ == "det"), None)
        if det:
            j = first_subtok_idx(det.text, prefix_ids, tokenizer)
            if j is not None:
                gold[j] = 1

    # 3 ─ Argument structure / animacy → main verb (ROOT)
    elif any(
        ex.UID.startswith(p)
        for p in ("animate_subject", "animate_object", "animate_", "s-selection")
    ):
        root = next((t for t in doc if t.dep_ == "ROOT"), None)
        token_for_evidence = root

        # If the verb isn't present in the truncated prefix (e.g. two-prefix items
        # like "Tina" vs "The horse"), use the nominal subject instead.
        if first_subtok_idx(root.text, prefix_ids, tokenizer) is None:
            token_for_evidence = next(
                (t for t in doc if t.dep_ in {"nsubj", "nsubjpass"}), root
            )

        j = first_subtok_idx(token_for_evidence.text, prefix_ids, tokenizer)
        if j is not None:
            gold[j] = 1

    # 4 ─ NPI licensing → the NPI word itself
    elif ex.UID.startswith("npi_"):
        npi_tok = next((t for t in doc if t.text.lower() in NPI), None)
        if npi_tok:
            j = first_subtok_idx(npi_tok.text, prefix_ids, tokenizer)
            if j is not None:
                gold[j] = 1

    # any UID not matching these four templates leaves gold all-zeros
    return gold
