"""Extract text and image features from Polyvore data.

Creates:
 - <output_dir>/text_features.npz  (keys, embeddings)
 - <output_dir>/image_features.npz (keys, embeddings)  [if images processed]
 - <output_dir>/items.json         (list of item metadata and index)
 - <output_dir>/vocab.json         (token -> id)

This is a lightweight, self-contained extractor that uses the project's
`SimpleTextEncoder` and `ImageEncoder` classes. Image extraction requires
either local image files or the `--download-images` flag to fetch images
from URLs in the dataset. Downloading many images may be slow and large.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import hashlib
import re
from typing import List, Dict, Tuple, Optional

import numpy as np
import torch

from src.features.text_features import SimpleTextEncoder, extract_text_feature
from src.features.image_features import ImageEncoder, extract_image_feature


def simple_tokenize(text: str) -> List[str]:
    # lowercase, remove non-alphanumeric (keep spaces), split on whitespace
    if not text:
        return []
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    toks = [t for t in text.split() if t]
    return toks


def build_vocab(texts: List[str], max_vocab: int = 10000) -> Dict[str,int]:
    freq = {}
    for t in texts:
        for tok in simple_tokenize(t):
            freq[tok] = freq.get(tok, 0) + 1
    # sort by freq
    items = sorted(freq.items(), key=lambda x: x[1], reverse=True)
    # reserve 0 for PAD, 1 for UNK
    vocab = {}
    idx = 2
    for tok, _ in items[: max(0, max_vocab - 2)]:
        vocab[tok] = idx
        idx += 1
    return vocab


def text_to_ids(text: str, vocab: Dict[str,int], seq_len: int = 16) -> List[int]:
    toks = simple_tokenize(text)
    ids = []
    for t in toks[:seq_len]:
        ids.append(vocab.get(t, 1))
    # pad
    if len(ids) < seq_len:
        ids = ids + [0] * (seq_len - len(ids))
    return ids


def md5_hash(s: str) -> str:
    return hashlib.md5(s.encode('utf-8')).hexdigest()


def collect_items_from_file(path: Path) -> List[Dict]:
    # Each file is a JSON array of outfit dicts; we extract items from each outfit
    with path.open('r', encoding='utf-8') as f:
        data = json.load(f)
    items = []
    for outfit in data:
        for it in outfit.get('items', []):
            # create a stable key for this physical item -- prefer image URL if present
            image_url = it.get('image', '')
            name = it.get('name', '')
            cat = it.get('categoryid', None)
            # some items may repeat across outfits; we'll dedupe by image_url if present, else name+category
            if image_url:
                key = f"img::{image_url}"
            else:
                key = f"name::{name}::cat::{cat}"
            items.append({
                'key': key,
                'name': name,
                'image_url': image_url,
                'categoryid': cat
            })
    return items


def download_image(url: str, cache_dir: Path) -> Optional[Path]:
    import requests
    if not url:
        return None
    fn = md5_hash(url) + '.jpg'
    dst = cache_dir / fn
    if dst.exists():
        return dst
    try:
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        dst.write_bytes(resp.content)
        return dst
    except Exception:
        return None


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--input', '-i', type=str, required=True,
                   help='Input file or directory containing polyvore JSON files')
    p.add_argument('--output', '-o', type=str, required=True,
                   help='Output directory for feature files')
    p.add_argument('--mode', choices=['text', 'image', 'both'], default='both')
    p.add_argument('--vocab-size', type=int, default=10000)
    p.add_argument('--seq-len', type=int, default=16)
    p.add_argument('--batch-size', type=int, default=256)
    p.add_argument('--download-images', action='store_true',
                   help='Download images referenced by URLs before extracting')
    p.add_argument('--image-cache', type=str, default='image_cache')
    args = p.parse_args()

    input_path = Path(args.input)
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    # gather files
    files = []
    if input_path.is_dir():
        for pth in input_path.glob('*.json'):
            files.append(pth)
    else:
        files = [input_path]
    if not files:
        raise SystemExit('No input JSON files found')

    # collect items (may contain duplicates across files)
    all_items = []
    for fpath in files:
        all_items.extend(collect_items_from_file(fpath))

    # dedupe by key and keep first occurrence
    seen = {}
    unique_items = []
    for it in all_items:
        k = it['key']
        if k in seen:
            continue
        idx = len(unique_items)
        seen[k] = idx
        unique_items.append(it)

    print(f"Collected {len(all_items)} items, {len(unique_items)} unique items")

    # Save items metadata
    items_meta = []
    for idx, it in enumerate(unique_items):
        items_meta.append({
            'index': idx,
            'key': it['key'],
            'name': it['name'],
            'image_url': it['image_url'],
            'categoryid': it['categoryid']
        })
    (out_dir / 'items.json').write_text(json.dumps(items_meta, indent=2), encoding='utf-8')

    # TEXT features
    if args.mode in ('text', 'both'):
        texts = [it['name'] or '' for it in unique_items]
        vocab = build_vocab(texts, max_vocab=args.vocab_size)
        # ensure unk and pad are implicitly present (0 pad, 1 unk)
        vocab_path = out_dir / 'vocab.json'
        vocab_path.write_text(json.dumps(vocab, indent=2), encoding='utf-8')

        seq_len = args.seq_len
        ids = np.array([text_to_ids(t, vocab, seq_len=seq_len) for t in texts], dtype=np.int64)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # create model sized according to vocab
        vocab_size = max(args.vocab_size, max((v for v in vocab.values()), default=2) + 1)
        text_model = SimpleTextEncoder(vocab_size=vocab_size, embed_dim=128, out_dim=None)
        # if SimpleTextEncoder requires out_dim, adapt: we can examine its attribute
        # but to keep compatible, attempt to create with TEXT_EMBED_DIM default by ignoring out_dim
        try:
            text_model = SimpleTextEncoder(vocab_size=vocab_size, embed_dim=128)
        except TypeError:
            text_model = SimpleTextEncoder()
        text_model = text_model.to(device)

        # batch inference
        bs = args.batch_size
        embeddings = []
        for i in range(0, len(ids), bs):
            batch = torch.tensor(ids[i:i+bs], dtype=torch.long)
            emb = extract_text_feature(batch, text_model, device=device)
            # extract_text_feature may return vector per item or single; ensure shape
            emb = np.asarray(emb)
            if emb.ndim == 1:
                emb = emb.reshape(1, -1)
            embeddings.append(emb)
        text_emb = np.vstack(embeddings).astype(np.float32)
        np.savez_compressed(out_dir / 'text_features.npz', keys=[it['key'] for it in unique_items], embeddings=text_emb)
        print(f"Wrote text features: {text_emb.shape} to {out_dir / 'text_features.npz'}")

    # IMAGE features
    if args.mode in ('image', 'both'):
        cache_dir = Path(args.image_cache)
        if args.download_images:
            cache_dir.mkdir(parents=True, exist_ok=True)
        else:
            cache_dir = None

        image_keys = []
        image_embs = []
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        image_model = ImageEncoder().to(device)

        for it in unique_items:
            url = it['image_url']
            if not url:
                continue
            local_path = None
            if cache_dir is not None:
                local_path = download_image(url, cache_dir)
            # if local_path None, try to interpret url as local file
            if local_path is None:
                maybe_local = Path(url)
                if maybe_local.exists():
                    local_path = maybe_local
            if local_path is None:
                # skip if no local image
                continue
            try:
                emb = extract_image_feature(local_path, image_model, device=device)
            except Exception:
                continue
            image_keys.append(it['key'])
            image_embs.append(np.asarray(emb).astype(np.float32))

        if image_embs:
            image_emb_arr = np.vstack(image_embs)
            np.savez_compressed(out_dir / 'image_features.npz', keys=image_keys, embeddings=image_emb_arr)
            print(f"Wrote image features: {image_emb_arr.shape} to {out_dir / 'image_features.npz'}")
        else:
            print("No image embeddings were extracted (no images available or download disabled)")


if __name__ == '__main__':
    main()
