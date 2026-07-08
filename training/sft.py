"""Stage-1 SFT: multi-head routing fine-tune.

Pools the backbone's last-token hidden state and trains three heads:
task classification, route classification over the model catalog, and a
feature projection used online by the per-user bandit.

Two modes:
  * full fine-tune (default) -- backbone + heads train together.
  * --freeze-backbone -- backbone is frozen (requires_grad=False); we precompute
    pooled embeddings ONCE (batched, no grad, cached), then fit only the three
    linear heads on the cache. This is the efficient path for large corpora and
    the PRIVACY-SAFE path for private data: the published weights are only the
    linear heads (pooled-embedding projections carry no generative capacity and
    cannot reconstruct corpus text); the backbone stays the public base model.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader, Dataset, TensorDataset
from transformers import AutoModel, AutoTokenizer


class RoutingDataset(Dataset):
    def __init__(self, path: Path, tasks: list[str], catalog: list[str]):
        self.rows = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
        self.tasks = {t: i for i, t in enumerate(tasks)}
        self.catalog = {m: i for i, m in enumerate(catalog)}
        self.rows = [r for r in self.rows if r["route"] in self.catalog]

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        return r["prompt"], self.tasks.get(r["task"], self.tasks["general"]), self.catalog[r["route"]]


class ZenRouter(nn.Module):
    def __init__(self, base: str, n_tasks: int, n_routes: int, feat_dim: int, dtype=torch.float32):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(base, dtype=dtype)
        hidden = self.backbone.config.hidden_size
        self.task_head = nn.Linear(hidden, n_tasks)
        self.route_head = nn.Linear(hidden, n_routes)
        self.feature_head = nn.Linear(hidden, feat_dim)

    def embed(self, input_ids, attention_mask):
        out = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        idx = attention_mask.sum(dim=1) - 1
        return out.last_hidden_state[torch.arange(input_ids.size(0)), idx].float()

    def heads(self, pooled):
        return self.task_head(pooled), self.route_head(pooled), self.feature_head(pooled)

    def forward(self, input_ids, attention_mask):
        return self.heads(self.embed(input_ids, attention_mask))


def collate_fn(tok, max_len):
    def collate(batch):
        prompts, tasks, routes = zip(*batch)
        enc = tok(list(prompts), padding=True, truncation=True, max_length=max_len, return_tensors="pt")
        return enc, torch.tensor(tasks), torch.tensor(routes)
    return collate


def precompute_embeddings(model, dl, device):
    """One no-grad pass over the data caching pooled embeddings + labels."""
    model.backbone.eval()
    embs, ts, rs = [], [], []
    with torch.no_grad():
        for i, (enc, tasks, routes) in enumerate(dl):
            enc = {k: v.to(device) for k, v in enc.items()}
            embs.append(model.embed(**enc).cpu())
            ts.append(tasks)
            rs.append(routes)
            if i % 50 == 0:
                print(f"  embed batch {i} (cached {sum(e.size(0) for e in embs)} rows)")
    return torch.cat(embs), torch.cat(ts), torch.cat(rs)


def train_heads(model, cache, device, epochs, lr, batch=512):
    E, ts, rs = cache
    heads = list(model.task_head.parameters()) + list(model.route_head.parameters()) + list(model.feature_head.parameters())
    opt = torch.optim.AdamW(heads, lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    dl = DataLoader(TensorDataset(E, ts, rs), batch_size=batch, shuffle=True)
    for epoch in range(epochs):
        tot = 0.0
        for e, t, r in dl:
            e, t, r = e.to(device), t.to(device), r.to(device)
            tl, rl, _ = model.heads(e)
            loss = loss_fn(tl, t) + loss_fn(rl, r)
            loss.backward()
            opt.step()
            opt.zero_grad()
            tot += loss.item()
        print(f"epoch {epoch} mean_loss {tot / len(dl):.4f}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--data", type=Path, default=None, help="override cfg data path")
    ap.add_argument("--freeze-backbone", action="store_true",
                    help="freeze backbone, precompute embeddings, train heads only (private-safe)")
    args = ap.parse_args()
    cfg = yaml.safe_load(args.config.read_text())
    sft = cfg["sft"]
    data_path = args.data or Path(cfg.get("data", "data/routing-sft.jsonl"))

    catalog = [m["id"] for m in yaml.safe_load(Path(cfg["catalog"]).read_text())["models"]]
    tok = AutoTokenizer.from_pretrained(cfg["base_model"])
    ds = RoutingDataset(Path(data_path), cfg["tasks"], catalog)
    model = ZenRouter(cfg["base_model"], len(cfg["tasks"]), len(catalog), cfg["heads"]["features"]["dim"])
    print(f"route head: {len(catalog)} classes (classes_from: catalog); task head: {len(cfg['tasks'])} classes")
    print(f"data: {data_path}  rows: {len(ds)}  freeze_backbone: {args.freeze_backbone}")
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    model.to(device)

    dl = DataLoader(ds, batch_size=sft["batch_size"], shuffle=not args.freeze_backbone,
                    collate_fn=collate_fn(tok, sft["max_seq_len"]))

    if args.freeze_backbone:
        for p in model.backbone.parameters():
            p.requires_grad_(False)
        print("precomputing pooled embeddings (backbone frozen)...")
        cache = precompute_embeddings(model, dl, device)
        print(f"cached {cache[0].size(0)} embeddings, dim {cache[0].size(1)}; training heads...")
        train_heads(model, cache, device, sft["epochs"], float(sft["lr"]))
    else:
        model.train()
        opt = torch.optim.AdamW(model.parameters(), lr=float(sft["lr"]))
        loss_fn = nn.CrossEntropyLoss()
        for epoch in range(sft["epochs"]):
            for step, (enc, tasks, routes) in enumerate(dl):
                enc = {k: v.to(device) for k, v in enc.items()}
                t_logits, r_logits, _ = model(**enc)
                loss = loss_fn(t_logits, tasks.to(device)) + loss_fn(r_logits, routes.to(device))
                loss.backward()
                opt.step()
                opt.zero_grad()
                if step % 50 == 0:
                    print(f"epoch {epoch} step {step} loss {loss.item():.4f}")

    out = Path(cfg["output_dir"])
    out.mkdir(parents=True, exist_ok=True)
    # Frozen backbone == public base model, so publish only the trained heads.
    state = model.state_dict()
    if args.freeze_backbone:
        state = {k: v for k, v in state.items() if not k.startswith("backbone.")}
    torch.save(state, out / "zen-router.pt")
    tok.save_pretrained(out)
    (out / "router_config.json").write_text(
        json.dumps({"base": cfg["base_model"], "tasks": cfg["tasks"], "catalog": catalog,
                    "frozen_backbone": args.freeze_backbone})
    )
    print(f"saved -> {out}  (heads_only={args.freeze_backbone})")


if __name__ == "__main__":
    main()
