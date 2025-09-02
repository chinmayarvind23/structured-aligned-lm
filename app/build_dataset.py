from __future__ import annotations
from pathlib import Path
from collections import defaultdict
import json

from datasets import load_dataset

OUT_TRAIN = Path("data/struct_train.jsonl")
OUT_VAL = Path("data/struct_val.jsonl")
OUT_TRAIN.parent.mkdir(parents=True, exist_ok=True)

CANON_ORDER = ["BACKGROUND", "OBJECTIVE", "METHODS", "RESULTS", "CONCLUSIONS"]
LABEL_MAP = {
    "BACKGROUND": "Background",
    "OBJECTIVE": "Objective",
    "METHODS": "Methods",
    "RESULTS": "Results",
    "CONCLUSIONS": "Conclusions",
}

ID_CANDIDATES = ["abstract_id", "pmid", "article_id", "doc_id", "id"]
TEXT_CANDIDATES = ["abstract_text", "sentence", "text", "content"]
LABEL_CANDIDATES = ["labels", "label", "section_label", "section"]
ORDER_CANDIDATES = ["order", "line_number", "sentence_index", "position", "idx"]


def _first_present_key(example: dict, keys: list[str]) -> str | None:
    for k in keys:
        if k in example:
            return k
    return None


def detect_schema(ds_split) -> tuple[str, str, str, str | None]:
    """
    Peek at one example to figure out which keys are used for id/text/label/(optional) order.
    """
    ex = ds_split[0]
    id_key = _first_present_key(ex, ID_CANDIDATES)
    text_key = _first_present_key(ex, TEXT_CANDIDATES)
    label_key = _first_present_key(ex, LABEL_CANDIDATES)
    order_key = _first_present_key(ex, ORDER_CANDIDATES)
    if not id_key or not text_key or not label_key:
        raise KeyError(
            f"Could not detect schema. Found keys: {list(ex.keys())}\n"
            f"Need one of id={ID_CANDIDATES}, text={TEXT_CANDIDATES}, label={LABEL_CANDIDATES}"
        )
    return id_key, text_key, label_key, order_key


def normalize_label(lbl) -> str | None:
    if isinstance(lbl, str):
        u = lbl.strip().upper()
        for k in CANON_ORDER:
            if k in u:
                return k
        if u in LABEL_MAP:
            return u
    return None


def build_pair(abstract_id: str, ordered_sents: list[tuple[int, str, str]]) -> dict:
    buckets: dict[str, list[str]] = defaultdict(list)
    seen_order: list[str] = []
    for _, sent, lab in ordered_sents:
        if lab not in LABEL_MAP:
            continue
        buckets[lab].append(sent.strip())
        if lab not in seen_order:
            seen_order.append(lab)
    order = seen_order if seen_order else CANON_ORDER
    outline = [LABEL_MAP[lab] for lab in order]
    sections = []
    for i, lab in enumerate(order, 1):
        heading = f"{i}. {LABEL_MAP[lab]}"
        body = " ".join(buckets.get(lab, [])) or "(no content available)"
        sections.append(f"{heading}\n\n{body}")
    body_text = "\n\n".join(sections)

    prompt = (
        "You will write a structured abstract with numbered sections.\n"
        "Task: First produce an Outline with 3–6 bullets using standard scholarly headings, "
        "then write one numbered section per bullet. Follow the order and keep sections cohesive.\n"
        f"Abstract ID: {abstract_id}"
    )
    response = "Outline:\n" + "\n".join([f"{i+1}. {name}" for i, name in enumerate(outline)]) + "\n\n" + body_text
    return {"prompt": prompt, "response": response}


def main():
    ds = load_dataset("armanc/pubmed-rct20k")
    id_key, text_key, label_key, order_key = detect_schema(ds["train"])
    print(f"[schema] id={id_key} text={text_key} label={label_key} order={order_key}")
    grouped: dict[str, list[tuple[int, str, str]]] = defaultdict(list)
    counts = 0
    for split in ["train", "validation", "test"]:
        for ex in ds[split]:
            aid = str(ex[id_key])
            sent = ex[text_key]
            lab_raw = ex[label_key]
            lab = normalize_label(lab_raw)
            if not isinstance(sent, str) or not lab:
                continue
            if order_key and order_key in ex and isinstance(ex[order_key], int):
                ord_idx = int(ex[order_key])
            else:
                ord_idx = len(grouped[aid])
            grouped[aid].append((ord_idx, sent, lab))
            counts += 1
    print(f"[grouping] collected {counts} sentences across {len(grouped)} abstracts")
    pairs: list[dict] = []
    for aid, triples in grouped.items():
        triples.sort(key=lambda t: t[0])
        pairs.append(build_pair(aid, triples))
    train = pairs[:2000]
    val = pairs[2000:2300]

    with OUT_TRAIN.open("w", encoding="utf-8") as f:
        for r in train:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with OUT_VAL.open("w", encoding="utf-8") as f:
        for r in val:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"[done] wrote {len(train)} train, {len(val)} val to data/")
    print("Example outline:", json.loads(Path(OUT_TRAIN).read_text(encoding='utf-8').splitlines()[0])["response"].splitlines()[1:5])


if __name__ == "__main__":
    main()