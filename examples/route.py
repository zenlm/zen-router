"""Route a prompt with a trained zen-router checkpoint."""
import json
import sys
from pathlib import Path

import torch
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from training.sft import ZenRouter

model_dir = Path("out/zen-router")
cfg = json.loads((model_dir / "router_config.json").read_text())
tok = AutoTokenizer.from_pretrained(model_dir)
model = ZenRouter(str(model_dir), len(cfg["tasks"]), len(cfg["catalog"]), 256)
model.load_state_dict(torch.load(model_dir / "zen-router.pt", map_location="cpu"))
model.eval()

prompt = sys.argv[1] if len(sys.argv) > 1 else "refactor this rust function to be async"
enc = tok(prompt, return_tensors="pt")
with torch.no_grad():
    t_logits, r_logits, feats = model(**enc)
print("task :", cfg["tasks"][t_logits.argmax().item()])
print("route:", cfg["catalog"][r_logits.argmax().item()])
