"""Check the saved PyTorch/ONNX pairs and reproduce CPU latency measurements."""
import argparse
import csv
import hashlib
import json
import platform
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
from torchvision import datasets

from experiment import cpu_metrics, open_model, transform, write_csv


@torch.inference_mode()
def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=Path(__file__).resolve().parent)
    p.add_argument("--out", type=Path, default=Path("verification"))
    p.add_argument("--data", type=Path, default=Path("cifar-data"))
    p.add_argument("--full-test", action="store_true",
                   help="Evaluate all official test images in both runtimes; downloads CIFAR if needed.")
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    torch.manual_seed(450)
    rows, pairs, checked, heldout = [], [], [], {}
    for task in ("kd", "nas"):
        artifacts = args.root/"models"/task
        saved = args.root/"results"/task
        manifest = json.loads((saved/"SHA256.json").read_text())
        for name, expected in manifest.items():
            file = (artifacts if name.endswith((".pt", ".onnx")) else saved)/name
            assert hashlib.sha256(file.read_bytes()).hexdigest() == expected, name
            checked.append(f"{task}/{name}")
        for path in sorted(artifacts.glob("*.pt")):
            model, cp = open_model(path)
            cfg = cp["config"]
            side = cfg["resolution"] if cfg else 32
            classes = 10 if task=="nas" else 100
            rng = torch.Generator().manual_seed(450)
            example = torch.randn(3,3,side,side,generator=rng)
            native = model(example).numpy()
            assert native.shape==(3,classes) and np.isfinite(native).all()
            history = list(csv.DictReader((saved/f"{path.stem}_history.csv").open(encoding="utf-8")))
            selected = max(history,key=lambda row:float(row["validation_accuracy"]))
            assert cp["epoch"]==int(selected["epoch"])
            assert cp["validation_accuracy"]==float(selected["validation_accuracy"])
            deployed = path.with_suffix(".onnx")
            metadata = dict(name=path.stem, classes=classes, input_resolution=side,
                            parameters=sum(x.numel() for x in model.parameters()),
                            checkpoint_bytes=path.stat().st_size,
                            selected_epoch=cp["epoch"], validation_accuracy=cp["validation_accuracy"])
            p50 = cpu_metrics(model,example[:1])
            native_row = dict(task=task,model=path.stem,runtime="pytorch",bytes=path.stat().st_size,
                              test_images=0,test_accuracy=None,**p50)
            rows.append(native_row)
            if not deployed.exists():
                assert path.stem=="teacher"
                if args.full_test:
                    key=(classes,side)
                    if key not in heldout:
                        heldout[key]=datasets.CIFAR100(args.data,train=False,download=True,
                                                       transform=transform(classes,side))
                    iterator=torch.utils.data.DataLoader(heldout[key],batch_size=64)
                    correct=sum(int((model(images).argmax(1)==target).sum()) for images,target in iterator)
                    native_row.update(test_images=len(heldout[key]),test_accuracy=100*correct/len(heldout[key]))
                pairs.append(metadata)
                continue
            onnx.checker.check_model(str(deployed))
            options=ort.SessionOptions()
            options.intra_op_num_threads=options.inter_op_num_threads=1
            session=ort.InferenceSession(str(deployed),options,providers=["CPUExecutionProvider"])
            def run(x):
                return session.run(None,{"image":x})[0]
            exported=run(example.numpy())
            assert np.isfinite(exported).all() and exported.shape==native.shape
            np.testing.assert_allclose(native,exported,rtol=1e-3,atol=2e-4)
            np.testing.assert_array_equal(native.argmax(1),exported.argmax(1))
            ort_row=dict(task=task,model=path.stem,runtime="onnxruntime",bytes=deployed.stat().st_size,
                         test_images=0,test_accuracy=None,**cpu_metrics(run,example[:1].numpy()))
            rows.append(ort_row)
            if args.full_test:
                key=(classes,side)
                if key not in heldout:
                    cls=datasets.CIFAR10 if classes==10 else datasets.CIFAR100
                    heldout[key]=cls(args.data,train=False,download=True,transform=transform(classes,side))
                iterator=torch.utils.data.DataLoader(heldout[key],batch_size=64)
                count=0; native_correct=0; onnx_correct=0; agree=0
                for x,y in iterator:
                    pred=model(x).argmax(1).numpy(); other=run(x.numpy()).argmax(1)
                    native_correct+=int((pred==y.numpy()).sum())
                    onnx_correct+=int((other==y.numpy()).sum())
                    agree+=int((pred==other).sum()); count+=len(y)
                native_row.update(test_images=count,test_accuracy=100*native_correct/count)
                ort_row.update(test_images=count,test_accuracy=100*onnx_correct/count)
                metadata["test_top1_agreement"]=100*agree/count
            metadata.update(onnx_bytes=deployed.stat().st_size,
                            cpu_smoke_max_logit_error=float(np.abs(native-exported).max()),
                            cpu_smoke_predictions=exported.argmax(1).tolist())
            pairs.append(metadata)
            print(f"{path.stem}: PyTorch/ONNX CPU checks passed",flush=True)
    assert len(pairs)==8 and sum("onnx_bytes" in x for x in pairs)==7
    write_csv(args.out/"local_cpu_benchmark.csv",rows)
    (args.out/"local_model_verification.json").write_text(json.dumps(
        {"host":platform.platform(),"torch":torch.__version__,"onnxruntime":ort.__version__,
         "full_test_repeated":args.full_test,"verified_original_file_hashes":len(checked),
         "seeded_smoke_batch":3,"cpu_threads":1,"measurements":pairs},indent=2),encoding="utf-8")

if __name__=="__main__":
    main()
