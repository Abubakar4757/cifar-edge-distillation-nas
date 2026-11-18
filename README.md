# CIFAR Edge Distillation and NAS

Assignment 4 · Edge Computing & Intelligent Systems  
Abubakar Bilal · 22I-0450 · Section A

Two reproducible experiments for lightweight CPU image classification: teacher-to-student knowledge distillation on CIFAR-100, and resource-constrained evolutionary architecture search on CIFAR-10.

## Results from the saved Kaggle runs

The ResNet-50 teacher reached **68.77%** CIFAR-100 test accuracy.

| CIFAR-100 student | Baseline | Distilled | Change | Distilled ONNX CPU p50 |
|---|---:|---:|---:|---:|
| MobileNetV2 | 60.96% | 65.28% | +4.32 pp | 1.181 ms |
| SqueezeNet 1.1 | 1.00% | 29.94% | +28.94 pp | 0.539 ms |
| EfficientNet-Lite0 | 60.17% | 65.27% | +5.10 pp | 1.497 ms |

SqueezeNet's baseline stayed at 1.00% validation accuracy for all 22 epochs, with training loss near log(100). Its unusually large gain is relative to a collapsed baseline under this shared training recipe. One run per condition does not establish statistical significance or an architecture's best possible result.

NAS evaluated 18 configurations, including four rejected before training for breaking the latency or size limits. Trial 14 had the best eligible proxy validation accuracy (69.42%). After 34 epochs from scratch, it reached **84.28%** test accuracy. Its 251,189-byte checkpoint and 1.989 ms isolated CPU p50 passed the 1 MiB and 4 ms limits; the 232,106-byte ONNX export measured **0.458 ms** p50. These measurements are specific to the Kaggle CPU environment.

## Install

Python 3.12 or 3.13. For a CPU environment, install the PyTorch CPU wheels first. On Kaggle use its GPU PyTorch build.

```text
python -m pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
```

Training used Python 3.13.16, PyTorch 2.11.0+cu130 and one Tesla T4 in a T4 x2 allocation. The original environment JSON files identify the software versions. The full experiments download official torchvision CIFAR archives and keep the official test set separate from the 45,000/5,000 stratified fitting and validation split (seed 450).

## Inference and benchmarking

Run from the repository root. Replace `example.png` with an RGB image. Use the corresponding checkpoint or ONNX file for any of the six student variants. ONNX defaults to CIFAR-100; specify `--arch nas` for the CIFAR-10 export. Native checkpoints infer the family from their metadata. Output is the zero-based CIFAR class index and confidence.

```text
python experiment.py infer --checkpoint models/kd/mobilenet_v2_distilled.pt --image example.png
python experiment.py infer --checkpoint models/kd/efficientnet_lite0_baseline.onnx --image example.png
python experiment.py infer --arch nas --checkpoint models/nas/nas_winner.onnx --image example.png
python benchmark_models.py --out verification
python benchmark_models.py --full-test --data cifar-data --out verification_full
```

The first benchmark command checks hashes, checkpoint reconstruction, shapes and close logits for three seeded synthetic examples per deployment pair. Optional `--full-test` repeats accuracy on all 10,000 official images in each dataset and adds the teacher's native test accuracy. It measures the current host separately; the untouched Kaggle CSVs under `results/kd/` and `results/nas/` are the reported experimental measurements. Timing is single-threaded, batch one, with 20 warmups and 100 forwards. It excludes model loading, preprocessing and file I/O. ONNX uses CPUExecutionProvider.

## Reproduce the experiments

```text
python experiment.py kd --data cifar-data --out reproduced/kd --require-gpu
python experiment.py nas --data cifar-data --out reproduced/nas --require-gpu
```

Use separate output directories. Add `--smoke` and use a disposable output directory for a one-epoch, reduced-data preflight. The submitted models come from the full experiments, not the preflights.

- **Distillation:** ResNet-50 for 32 epochs, then 22 epochs per student condition. Baselines use cross-entropy only, with no teacher or projection; the KD branch adds temperature-scaled teacher KL divergence and normalized intermediate feature matching. The student pairs use the same split, architecture, initial seed and schedule. Both branches retain the first checkpoint with the highest validation accuracy.
- **NAS:** variable 3-7 block depth, channel widths, kernels, standard or depthwise separable convolutions, activations and resolutions. Candidate CPU probes run in separate processes before four-epoch proxy training on eligible candidates. The best validation candidate is saved and trained independently for 34 epochs. Linux peak RSS includes process and runtime overhead; it is recorded separately from the enforced checkpoint and p50 limits.
- **Deployment:** PyTorch state dictionaries and metadata, opset-17 ONNX exports with a dynamic batch axis and fixed spatial inputs. CPU latency depends on the host and configuration; these results do not characterize a physical edge device.

## Files to review

| Location | Contents |
|---|---|
| [experiment.py](experiment.py) | Teacher, student architectures, baseline and KD training, search, export and inference |
| [benchmark_models.py](benchmark_models.py) | Standalone artifact checks, CPU timing and optional official test evaluation |
| [models/kd/](models/kd/) | Teacher, six student checkpoints and six ONNX exports |
| [models/nas/](models/nas/) | Winner checkpoint and ONNX export |
| [results/](results/) | Benchmark CSVs, histories, constraints, split indices, search and verification |
| [evidence/](evidence/) | Saved Kaggle completion screenshots |
| [PDF report](docs/CIFAR_Edge_Distillation_NAS_22I-0450_A.pdf) | Standalone technical documentation |
| [SUBMISSION_CHECKLIST.md](SUBMISSION_CHECKLIST.md) | Evidence for every required deliverable |

Submit this repository link and the PDF separately. No archives or datasets are needed.
