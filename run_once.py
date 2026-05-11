"""Single run of demian, no input, no stream, save trajectory for analysis."""
import logging
import json
import os
import torch
import yaml
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from demian.vibration import VibrationTracker
from demian.nous import NousInjector
from demian.loop import generate_with_proprioception

logging.basicConfig(level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")

cfg = yaml.safe_load(Path("config.yaml").read_text())
model_id = cfg.get("proprioceptor_model_id", "Qwen/Qwen2.5-3B-Instruct")

print("Loading " + model_id + "...")
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
tokenizer.padding_side = "left"
model = AutoModelForCausalLM.from_pretrained(
    model_id, device_map="auto", dtype=torch.float16,
    trust_remote_code=True, attn_implementation="eager")
model.eval()

d_model = model.config.hidden_size
print("d_model=" + str(d_model))

tracker = VibrationTracker(d_model=d_model, target_dim=128)
injector = NousInjector(
    model=model, tracker=tracker, max_memory_length=16,
    injection_scale=cfg.get("injection_scale", 0.01),
    blend_mode="append")

text, snapshots = generate_with_proprioception(
    model=model, tokenizer=tokenizer,
    tracker=tracker, injector=injector,
    prompt="Who are you when no one is reading?",
    max_new_tokens=256,
    temperature=0.7,
    proprio_inject=True,
    device=str(model.device),
    damping=cfg.get("injection_damping", 0.7),
    inject_mode="one",
    stream=False,
)

print("Generated " + str(len(snapshots)) + " tokens")

traj = []
for i, s in enumerate(snapshots):
    entry = dict(
        step=i + 1,
        spectral_centroid=s.shape.spectral_centroid,
        spectral_concentration=s.shape.spectral_concentration,
        layer_work_ratio=s.shape.layer_work_ratio,
        layer_agreement=s.shape.layer_agreement,
        velocity_align=s.shape.velocity_align,
        attention_dim=s.shape.attention_dim,
        energy=s.shape.energy,
        kurtosis=s.shape.kurtosis,
        peakiness=s.shape.peakiness,
        dominance_ratio=s.shape.dominance_ratio,
        n_peaks=s.shape.n_peaks,
        entropy=s.shape.entropy,
        residual_norm=s.residual_norm,
        residual_delta=s.residual_delta,
        temporal_coherence=s.temporal_coherence,
    )
    traj.append(entry)

data_dir = os.path.join(os.path.dirname(__file__), "data")
os.makedirs(data_dir, exist_ok=True)
out = os.path.join(data_dir, "run_once_spectral.json")
with open(out, "w") as f:
    json.dump(traj, f, indent=2)

print("Trajectory saved: " + out)
print("Text sample:")
print(text[:500])
