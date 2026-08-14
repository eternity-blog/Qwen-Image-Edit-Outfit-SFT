---
license: other
license_name: cc-by-nc-sa-derived
pretty_name: Qwen Outfit IDM Synth v2
task_categories:
  - image-to-image
tags:
  - virtual-try-on
  - image-editing
  - sft
  - qwen-image-edit
---

# Qwen Outfit IDM Synth (v2 prompts)

Personal research dataset for Qwen-Image-Edit keyframe outfit-swap SFT.

## Contents

- `idm_unpaired_train/manifest.jsonl` + `images/` — IDM-VTON synthetic try-on targets  
- `converted_idm_synth_train_v2/metadata_*.jsonl` — DiffSynth-style rows with **full** garment-edit prompts  

Person/cloth paths in metadata point at **VITON-HD** layout; download VITON yourself and symlink.

## License

**Non-commercial.** Derived from:

- VITON-HD ([CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/))  
- IDM-VTON ([CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/))  

Cite upstream papers/repos. Not for commercial product training without rights.

## Code

Training / conversion scripts: see the companion GitHub repo (this card is data-only).
