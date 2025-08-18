import argparse
import os
import re
import csv
from functools import lru_cache
from typing import Dict, Optional, List, Tuple

import torch
from difflib import get_close_matches
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

from Player2Vec.models.model import SLMWrapper


def _load_embeddings(emb_path: str) -> Dict[str, torch.Tensor]:
    """
    Load player embeddings p (dim=256 expected) from:
    - .pt/.pth: a dict mapping player_id (str or int) -> 1D tensor
    - .csv: header with id plus 256 columns p0..p255 (comma-separated)
    Returns a dict[str, Tensor(256)].
    """
    if not os.path.exists(emb_path):
        raise FileNotFoundError(f"Embeddings not found: {emb_path}")

    ext = os.path.splitext(emb_path)[1].lower()
    if ext in [".pt", ".pth"]:
        obj = torch.load(emb_path, map_location="cpu")
        if isinstance(obj, dict):
            out: Dict[str, torch.Tensor] = {}
            for k, v in obj.items():
                key = str(k)
                t = torch.as_tensor(v).float().view(-1)
                out[key] = t
            return out
        else:
            raise ValueError("Expected a dict in the .pt/.pth file mapping id->tensor")

    if ext == ".csv":
        import csv
        out: Dict[str, torch.Tensor] = {}
        with open(emb_path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                pid = str(row.get("id"))
                if pid is None:
                    raise ValueError("CSV must contain an 'id' column")
                # Collect all numeric fields except 'id'
                vec = []
                for k, v in row.items():
                    if k == "id":
                        continue
                    if v is None or v == "":
                        continue
                    vec.append(float(v))
                t = torch.tensor(vec, dtype=torch.float32)
                out[pid] = t
        return out

    raise ValueError(f"Unsupported embedding format: {ext}")


def _select_p(embeds: Dict[str, torch.Tensor], player_id: str) -> torch.Tensor:
    if player_id in embeds:
        return embeds[player_id]
    # try integer key fallback
    if player_id.isdigit() and str(int(player_id)) in embeds:
        return embeds[str(int(player_id))]
    # soft match: try zero-padded and raw
    if player_id.lstrip("0") in embeds:
        return embeds[player_id.lstrip("0")]
    raise KeyError(f"player_id '{player_id}' not found in embeddings. Available keys sample: {list(embeds.keys())[:5]}")


def _load_players_csv(path: str) -> List[Dict[str, str]]:
    """Load player roster CSV with columns: id,name,team,alt_names (alt separated by '|').
    Returns a list of dicts with lowercase helper fields for matching.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"players_csv not found: {path}")
    rows: List[Dict[str, str]] = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        # normalize header names
        headers = {h.lower(): h for h in reader.fieldnames or []}
        id_key = headers.get("id") or headers.get("player_id") or headers.get("pid")
        name_key = headers.get("name") or headers.get("player_name")
        team_key = headers.get("team") or headers.get("team_name")
        alt_key = headers.get("alt_names") or headers.get("aliases") or headers.get("aka")
        if not id_key or not name_key:
            raise ValueError("players_csv must contain at least 'id' and 'name' columns")
        for row in reader:
            pid = str(row[id_key]).strip()
            name = (row.get(name_key) or "").strip()
            team = (row.get(team_key) or "").strip()
            alt = (row.get(alt_key) or "").strip()
            alts = [a.strip() for a in alt.split("|") if a.strip()] if alt else []
            rows.append({
                "id": pid,
                "name": name,
                "team": team,
                "alts": alts,
                "_name_lc": name.casefold(),
                "_team_lc": team.casefold(),
                "_alts_lc": [a.casefold() for a in alts],
            })
    return rows


def _parse_question_for_player(question: str) -> Tuple[Optional[str], Optional[str]]:
    """Try to extract (team, name) from Japanese-like phrasing.
    Patterns handled:
      - "(TEAM)の(NAME)選手"
      - "(NAME)選手"
    Returns (team, name) possibly None.
    """
    q = question.strip()
    # Most specific: TEAM の NAME 選手
    m = re.search(r"(?P<team>.+?)の(?P<name>.+?)選手", q)
    if m:
        return m.group("team").strip(), m.group("name").strip()
    # Fallback: NAME 選手
    m2 = re.search(r"(?P<name>[^\s　]+?)選手", q)
    if m2:
        return None, m2.group("name").strip()
    return None, None


def _resolve_player_id(question: str, roster: List[Dict[str, str]]) -> Optional[str]:
    """Resolve player id from question using roster information.
    Matching order:
      1) Filter by team if present (contains match)
      2) Exact name/alias match (case-insensitive)
      3) Substring contains match
      4) Fuzzy match via difflib
    Returns player id or None if unresolved.
    """
    team, name = _parse_question_for_player(question)
    if not name:
        return None
    name_lc = name.casefold()
    cand = roster
    if team:
        team_lc = team.casefold()
        cand = [r for r in cand if team_lc in r.get("_team_lc", "")] or cand

    # 1) exact name/alias match
    for r in cand:
        if name_lc == r.get("_name_lc") or name_lc in r.get("_alts_lc", []):
            return r["id"]

    # 2) substring contains
    contains = [r for r in cand if (name_lc in r.get("_name_lc", "") or any(name_lc in a for a in r.get("_alts_lc", [])))]
    if len(contains) == 1:
        return contains[0]["id"]

    # 3) fuzzy among candidates (by name and alts)
    names_pool = [(r["id"], r["name"]) for r in cand]
    alts_pool = [(r["id"], alt) for r in cand for alt in r.get("alts", [])]
    label_map: Dict[str, str] = {}
    labels: List[str] = []
    for pid, label in names_pool + alts_pool:
        key = f"{pid}::{label}"
        label_map[key] = pid
        labels.append(label)
    if labels:
        match = get_close_matches(name, labels, n=1, cutoff=0.75)
        if match:
            # find corresponding pid
            mlabel = match[0]
            for pid, label in names_pool + alts_pool:
                if label == mlabel:
                    return pid
    return None


@lru_cache(maxsize=2)
def _load_mt(model_name: str):
    tok = AutoTokenizer.from_pretrained(model_name)
    mdl = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    return tok, mdl


def _translate(text: str, model_name: str, device: Optional[str] = None) -> str:
    tok, mdl = _load_mt(model_name)
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    mdl = mdl.to(device)
    batch = tok(text, return_tensors="pt", truncation=True).to(device)
    with torch.no_grad():
        out = mdl.generate(**batch, max_new_tokens=256)
    return tok.decode(out[0], skip_special_tokens=True)


def generate_style(player_id: Optional[str], question: str, emb_path: str, model_name: str = "distilgpt2", device: Optional[str] = None, max_new_tokens: int = 64, players_csv: Optional[str] = None, bridge_ja_en: bool = False, mt_ja_en: str = "Helsinki-NLP/opus-mt-ja-en", mt_en_ja: str = "Helsinki-NLP/opus-mt-en-ja") -> str:
    """
    Load p from emb_path and generate natural language with SLMWrapper.
    emb_path: .pt/.pth (dict id->tensor) or .csv
    player_id: key in the dict (string or int convertible)
    """
    embeds = _load_embeddings(emb_path)
    resolved_id = None
    if player_id is not None:
        resolved_id = str(player_id)
    elif players_csv:
        roster = _load_players_csv(players_csv)
        resolved_id = _resolve_player_id(question, roster)
        if resolved_id is None:
            raise ValueError("Could not resolve player from question. Provide --player_id or improve players_csv/phrasing.")
    else:
        raise ValueError("player_id not provided and players_csv not set. Provide one of them.")

    p = _select_p(embeds, str(resolved_id))

    # Ensure 256-dim as per design; if longer, take first 256; if shorter, pad zeros
    dim = p.numel()
    if dim < 256:
        p = torch.cat([p, torch.zeros(256 - dim)], dim=0)
    elif dim > 256:
        p = p[:256]

    # l2 normalize p as specified
    p = torch.nn.functional.normalize(p, dim=0)

    slm = SLMWrapper(model_name=model_name)

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    slm = slm.to(device)

    with torch.no_grad():
        if bridge_ja_en:
            # JA -> EN
            en_q = _translate(question, mt_ja_en, device=device)
            en_out = slm.generate(p.to(device), en_q, max_new_tokens=max_new_tokens)
            # EN -> JA
            text = _translate(en_out, mt_en_ja, device=device)
        else:
            text = slm.generate(p.to(device), question, max_new_tokens=max_new_tokens)
    return text


def main():
    parser = argparse.ArgumentParser(description="Player style QA inference")
    parser.add_argument("--player_id", type=str, required=False, help="If omitted, resolve from question using --players_csv")
    parser.add_argument("--question", type=str, required=True)
    parser.add_argument("--emb", type=str, default="checkpoints/player_embeddings.pt", help="Path to p embeddings (.pt/.pth/.csv)")
    parser.add_argument("--model", type=str, default="distilgpt2", help="HF Causal LM model name")
    parser.add_argument("--max_new_tokens", type=int, default=64)
    parser.add_argument("--players_csv", type=str, default=None, help="Roster CSV with columns: id,name,team,alt_names (alt separated by '|')")
    parser.add_argument("--bridge_ja_en", action="store_true", help="Translate JA→EN for input and EN→JA for output")
    parser.add_argument("--mt_ja_en", type=str, default="Helsinki-NLP/opus-mt-ja-en", help="MT model for JA→EN")
    parser.add_argument("--mt_en_ja", type=str, default="Helsinki-NLP/opus-mt-en-ja", help="MT model for EN→JA")
    args = parser.parse_args()

    answer = generate_style(
        player_id=args.player_id,
        question=args.question,
        emb_path=args.emb,
        model_name=args.model,
        max_new_tokens=args.max_new_tokens,
        players_csv=args.players_csv,
        bridge_ja_en=args.bridge_ja_en,
        mt_ja_en=args.mt_ja_en,
        mt_en_ja=args.mt_en_ja,
    )
    print(answer)


if __name__ == "__main__":
    main()
