"""Stage-1 SFT: multi-head routing fine-tune.

Pools the backbone's last-token hidden state and trains three heads:
task classification, route classification over the model catalog, and a
feature projection used online by the per-user bandit.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader, Dataset
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
    def __init__(self, base: str, n_tasks: int, n_routes: int, feat_dim: int):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(base, torch_dtype=torch.bfloat16)
        hidden = self.backbone.config.hidden_size
        self.task_head = nn.Linear(hidden, n_tasks)
        self.route_head = nn.Linear(hidden, n_routes)
        self.feature_head = nn.Linear(hidden, feat_dim)

    def forward(self, input_ids, attention_mask):
        out = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        last = out.last_hidden_state
        idx = attention_mask.sum(dim=1) - 1
        pooled = last[torch.arange(last.size(0)), idx].float()
        return self.task_head(pooled), self.route_head(pooled), self.feature_head(pooled)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    args = ap.parse_args()
    cfg = yaml.safe_load(args.config.read_text())

    catalog = [m["id"] for m in yaml.safe_load(Path(cfg["catalog"]).read_text())["models"]]
    tok = AutoTokenizer.from_pretrained(cfg["base_model"])
    ds = RoutingDataset(Path("data/routing-sft.jsonl"), cfg["tasks"], catalog)
    model = ZenRouter(cfg["base_model"], len(cfg["tasks"]), len(catalog), cfg["heads"]["features"]["dim"])
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    model.to(device).train()

    sft = cfg["sft"]
    opt = torch.optim.AdamW(model.parameters(), lr=float(sft["lr"]))
    loss_fn = nn.CrossEntropyLoss()

    def collate(batch):
        prompts, tasks, routes = zip(*batch)
        enc = tok(list(prompts), padding=True, truncation=True, max_length=sft["max_seq_len"], return_tensors="pt")
        return enc, torch.tensor(tasks), torch.tensor(routes)

    dl = DataLoader(ds, batch_size=sft["batch_size"], shuffle=True, collate_fn=collate)
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
    torch.save(model.state_dict(), out / "zen-router.pt")
    tok.save_pretrained(out)
    (out / "router_config.json").write_text(json.dumps({"tasks": cfg["tasks"], "catalog": catalog}))
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
