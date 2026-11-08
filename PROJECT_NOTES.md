# Implementation decisions and research notes

## Why this is an original starter, not a copied repository

The execution environment could not resolve `github.com`, so a live `git clone` could not complete. Public repository README and source-file views were reviewed through web search. This archive therefore contains a fresh implementation rather than claiming to be a clone or a verified modification of another author's code.

## Biggest research risk

A multimodal model needs genuinely aligned samples. Each training record must link the image, symptom observations and diagnosis from the same patient/episode. Pairing independent symptom-only and image-only datasets by class name does not produce valid multimodal training examples. Creating symptom strings from the diagnosis label also causes label leakage.

## Baseline limitations

- ResNet-18 late fusion is a practical baseline, not a novel architecture.
- A softmax score is not automatically a calibrated probability. Temperature scaling is fit on the validation data, then must be tested on a held-out set.
- The uncertainty thresholds are heuristics that need dataset-specific validation.
- Grad-CAM highlights spatial influence on a selected model score. It does not prove that the highlighted region is medically relevant.
- Symptom sensitivity is based on feature ablation and is not a causal explanation.
- Out-of-distribution detection is not implemented. A skin-only model may still produce scores for unsupported images, so input-domain checks and an abstention strategy remain future work.
