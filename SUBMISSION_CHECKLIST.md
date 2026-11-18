# Assignment 4 deliverables

| # | Requirement | Evidence |
|---:|---|---|
| 1 | ResNet-50 teacher and training code | `models/kd/teacher.pt`; `network`, `train`, and `kd` in `experiment.py`; teacher history and test metrics |
| 2 | Three student architectures | `network` and `Lite0` in `experiment.py` |
| 3 | Three independently trained baselines | `models/kd/*_baseline.pt`; separate hard-label-only branch; 22-epoch histories |
| 4 | KD training and three distilled checkpoints | `models/kd/*_distilled.pt`; frozen teacher, temperature 4, hard/soft loss and intermediate feature loss |
| 5 | NAS space and generator | `sample` and `network("nas", config)` in `experiment.py` |
| 6 | Executable evolutionary algorithm | `python experiment.py nas`; elitist mutation and random exploration |
| 7 | Search and winner evidence | `results/nas/search_history.csv`, `best_architecture.json`, candidate histories |
| 8 | Optimized trained architecture | `models/nas/nas_winner.pt`; 34-epoch history and final constraint check |
| 9 | Unified inference | `python experiment.py infer`, for all student variants and NAS |
| 10 | ONNX CPU deployment | Seven `.onnx` files; `CPUExecutionProvider`; local graph and parity verification |
| 11 | Automated CPU benchmarks and CSVs | `benchmark_models.py`, original `results/{kd,nas}/benchmark.csv`, separate verification CSV |
| 12 | Technical documentation and screenshots | `docs/CIFAR_Edge_Distillation_NAS_22I-0450_A.pdf`; `evidence/` |
| 13 | Installation and execution | `README.md`, `requirements.txt` |

Artifact integrity: original Kaggle manifests are in each task's results directory.
Resolve `.pt` and `.onnx` manifest entries in the matching models directory and
other entries in the results directory. `results/artifact_audit.json` and
`results/verification/` record the submission checks.
