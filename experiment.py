"""CIFAR knowledge distillation and evolutionary search for CPU deployment."""
import argparse
import copy
import csv
import json
import os
import platform
import random
import statistics
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torchvision import datasets, models, transforms

SEED = 450
STUDENTS = ("mobilenet_v2", "squeezenet", "efficientnet_lite0")
STATS = {10: ((.4914,.4822,.4465),(.247,.2435,.2616)),
         100: ((.5071,.4867,.4408),(.2675,.2565,.2761))}

def seed(n=SEED):
    random.seed(n); np.random.seed(n); torch.manual_seed(n)

def write_json(path, data):
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")

def write_csv(path, rows):
    if rows:
        with path.open("w", newline="", encoding="utf-8") as stream:
            out = csv.DictWriter(stream, fieldnames=list(rows[0]))
            out.writeheader(); out.writerows(rows)

def bnconv(a,b,k=3,s=1,g=1,act=nn.ReLU6):
    return nn.Sequential(nn.Conv2d(a,b,k,s,k//2,groups=g,bias=False),nn.BatchNorm2d(b),act())

class LiteBlock(nn.Module):
    def __init__(self,a,b,k,s,e):
        super().__init__()
        c=a*e
        self.body=nn.Sequential(bnconv(a,c,1) if e!=1 else nn.Identity(),
            bnconv(c,c,k,s,c),bnconv(c,b,1,act=nn.Identity))
        self.skip=a==b and s==1

    def forward(self,x):
        y=self.body(x)
        return x+y if self.skip else y

class Lite0(nn.Module):
    """EfficientNet-Lite0 stage widths, expansions and repeat counts; CIFAR stem."""
    def __init__(self):
        super().__init__()
        a=32; layers=[bnconv(3,a,s=1)]
        for b,k,s,e,n in [(16,3,1,1,1),(24,3,2,6,2),(40,5,2,6,2),
                          (80,3,2,6,3),(112,5,1,6,3),(192,5,2,6,4),(320,3,1,6,1)]:
            for i in range(n):
                layers.append(LiteBlock(a,b,k,s if not i else 1,e)); a=b
        self.features=nn.Sequential(*layers,bnconv(a,1280,1))
        self.classifier=nn.Sequential(nn.Dropout(.2),nn.Linear(1280,100))

    def forward(self,x):
        return self.classifier(self.features(x).mean((2,3)))

def network(name,config=None):
    if name=="teacher":
        m=models.resnet50(weights=None,num_classes=100)
        m.conv1=nn.Conv2d(3,64,3,1,1,bias=False); m.maxpool=nn.Identity()
        return m, m.layer4, 2048
    if name=="mobilenet_v2":
        m=models.mobilenet_v2(weights=None,num_classes=100)
        m.features[0][0].stride=(1,1)
        return m,m.features[-1],1280
    if name=="squeezenet":
        m=models.squeezenet1_1(weights=None,num_classes=100)
        m.features[0].stride=(1,1)
        return m,m.features[-1],512
    if name=="efficientnet_lite0":
        m=Lite0()
        return m,m.features[-1],1280
    if name!="nas":
        raise ValueError(name)
    act={"relu":nn.ReLU,"relu6":nn.ReLU6,"silu":nn.SiLU}[config["activation"]]
    a=3; seq=[]
    for i,b in enumerate(config["channels"]):
        k=config["kernels"][i]; s=2 if i in (1,3,5) else 1
        if config["ops"][i]=="dw":
            seq.extend((bnconv(a,a,k,s,a,act),bnconv(a,b,1,act=act)))
        else:
            seq.append(bnconv(a,b,k,s,act=act))
        a=b
    return nn.Sequential(*seq,nn.AdaptiveAvgPool2d(1),nn.Flatten(),nn.Linear(a,10)),None,a

def transform(classes,side,training=False):
    ops=[transforms.RandomCrop(32,padding=4),transforms.RandomHorizontalFlip()] if training else []
    if side!=32: ops.append(transforms.Resize((side,side),antialias=True))
    return transforms.Compose(ops+[transforms.ToTensor(),transforms.Normalize(*STATS[classes])])

def loaders(root,classes,side,batch,smoke):
    ds=datasets.CIFAR100 if classes==100 else datasets.CIFAR10
    fit=ds(root,train=True,download=True,transform=transform(classes,side,True))
    val=ds(root,train=True,download=False,transform=transform(classes,side))
    test=ds(root,train=False,download=True,transform=transform(classes,side))
    rng=np.random.default_rng(SEED)
    labels=np.array(fit.targets)
    groups=[rng.permutation(np.flatnonzero(labels==i)) for i in range(classes)]
    n=5000//classes
    vi=rng.permutation(np.concatenate([group[:n] for group in groups])).tolist()
    fi=rng.permutation(np.concatenate([group[n:] for group in groups])).tolist()
    assert not (set(vi)&set(fi)) and len(fi)+len(vi)==len(fit)
    def loader(data, indices,shuffle=False):
        if smoke: indices=indices[:max(batch*2,32)]
        return torch.utils.data.DataLoader(torch.utils.data.Subset(data,indices),batch_size=batch,
                shuffle=shuffle,num_workers=0 if smoke else 2,pin_memory=torch.cuda.is_available(),
                generator=torch.Generator().manual_seed(SEED))
    return loader(fit,fi,True),loader(val,vi),loader(test,list(range(len(test)))),{"fit":fi,"validation":vi}

@torch.inference_mode()
def evaluate(model,iterator,device):
    model.eval(); correct=0; total=0
    for x,y in iterator:
        p=model(x.to(device)).argmax(1).cpu()
        correct+=int((p==y).sum()); total+=len(y)
    return 100*correct/total

def feature_hook(layer,box):
    return layer.register_forward_hook(lambda _,args,output: box.append(output.mean((2,3))))

def train(name,config,load,epochs,device,out,tag,teacher=None,teacher_layer=None):
    seed()
    model,layer,channels=network(name,config)
    model.to(device,memory_format=torch.channels_last)
    sb=[]; tb=[]; hooks=[]
    projector=None
    if teacher is not None:
        teacher.eval(); teacher.requires_grad_(False)
        projector=nn.Linear(channels,2048,bias=False).to(device)
        hooks=[feature_hook(layer,sb),feature_hook(teacher_layer,tb)]
    params=list(model.parameters())+(list(projector.parameters()) if projector is not None else [])
    opt=torch.optim.SGD(params,lr=.1,momentum=.9,weight_decay=5e-4,nesterov=True)
    sch=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=epochs)
    cuda=device.type=="cuda"
    scale=torch.amp.GradScaler("cuda",enabled=cuda)
    best=-1.; history=[]
    for ep in range(1,epochs+1):
        model.train(); start=time.perf_counter(); totals=np.zeros(4); seen=0
        for x,y in load[0]:
            x=x.to(device,non_blocking=True,memory_format=torch.channels_last)
            y=y.to(device,non_blocking=True); opt.zero_grad(set_to_none=True)
            sb.clear(); tb.clear()
            with torch.autocast(device.type,dtype=torch.float16,enabled=cuda):
                pred=model(x); hard=F.cross_entropy(pred,y); loss=hard
                soft=feat=torch.zeros((),device=device)
                if teacher is not None:
                    with torch.no_grad(): target=teacher(x)
                    t=4.
                    soft=F.kl_div(F.log_softmax(pred.float()/t,dim=1),
                                  F.softmax(target.float()/t,dim=1),reduction="batchmean")*t*t
                    feat=F.mse_loss(F.normalize(projector(sb[0]).float()),
                                    F.normalize(tb[0].float()))*2048
                    loss=.5*hard+.5*soft+.25*feat
            scale.scale(loss).backward(); scale.step(opt); scale.update()
            totals+=np.array([loss.item(),hard.item(),soft.item(),feat.item()])*len(y); seen+=len(y)
        # Hooks are temporarily disabled during validation to avoid retaining activations.
        for h in hooks: h.remove()
        sb.clear(); tb.clear()
        val=evaluate(model,load[1],device)
        if teacher is not None: hooks=[feature_hook(layer,sb),feature_hook(teacher_layer,tb)]
        row=dict(epoch=ep,loss=totals[0]/seen,hard_loss=totals[1]/seen,
                 soft_loss=totals[2]/seen,feature_loss=totals[3]/seen,
                 validation_accuracy=val,seconds=time.perf_counter()-start)
        history.append(row); write_csv(out/f"{tag}_history.csv",history)
        if val>best:
            best=val
            path=out/f"{tag}.pt"
            torch.save({"network":name,"config":config,"state":model.state_dict(),
                        "validation_accuracy":best,"epoch":ep},path.with_suffix(".tmp"))
            path.with_suffix(".tmp").replace(path)
        print(tag,row,flush=True); sch.step()
    for h in hooks: h.remove()
    path=out/f"{tag}.pt"
    model.load_state_dict(torch.load(path,map_location=device,weights_only=True)["state"])
    return model,best

def cpu_metrics(call,x,warm=20,n=100):
    with torch.inference_mode():
        for _ in range(warm): call(x)
        values=[]
        for _ in range(n):
            a=time.perf_counter_ns(); call(x); values.append((time.perf_counter_ns()-a)/1e6)
    return dict(p50_ms=statistics.median(values),p95_ms=float(np.percentile(values,95)),samples=n)

def deploy(model,tag,out,side,loader,device):
    import onnx
    import onnxruntime as ort
    accuracy=evaluate(model,loader,device)
    cpu=copy.deepcopy(model).cpu().eval().to(memory_format=torch.contiguous_format)
    example=torch.randn(1,3,side,side)
    path=out/f"{tag}.onnx"
    torch.onnx.export(cpu,example,str(path),input_names=["image"],output_names=["logits"],
                      dynamic_axes={"image":{0:"batch"},"logits":{0:"batch"}},
                      opset_version=17,dynamo=False)
    onnx.checker.check_model(str(path))
    opt=ort.SessionOptions(); opt.intra_op_num_threads=1; opt.inter_op_num_threads=1
    session=ort.InferenceSession(str(path),opt,providers=["CPUExecutionProvider"])
    run=lambda x: session.run(None,{"image":x})[0]
    with torch.inference_mode(): ref=cpu(example).numpy()
    err=float(np.max(np.abs(ref-run(example.numpy()))))
    np.testing.assert_allclose(run(example.numpy()),ref,rtol=1e-3,atol=2e-4)
    matches=correct=count=0
    with torch.inference_mode():
        for x,y in loader:
            o=run(x.numpy()).argmax(1)
            matches+=int((o==cpu(x).argmax(1).numpy()).sum())
            correct+=int((o==y.numpy()).sum()); count+=len(y)
    return [{"model":tag,"runtime":runtime,"test_images":count,"test_accuracy":acc,
             "bytes":size,"onnx_native_agreement":100*matches/count,"example_max_error":err,**metrics}
            for runtime,acc,size,metrics in [
                ("pytorch",accuracy,(out/f"{tag}.pt").stat().st_size,cpu_metrics(cpu,example)),
                ("onnxruntime",100*correct/count,path.stat().st_size,cpu_metrics(run,example.numpy()))]]

def kd(args,device,out):
    load=loaders(args.data,100,32,args.batch,args.smoke)
    write_json(out/"split.json",load[3])
    teacher,_=train("teacher",None,load,args.teacher_epochs,device,out,"teacher")
    write_json(out/"teacher_metrics.json",{"test_accuracy":evaluate(teacher,load[2],device)})
    rows=[]
    for name in STUDENTS:
        for mode in ("baseline","distilled"):
            # Recreate each loader so equally seeded paired runs see the same order.
            load=loaders(args.data,100,32,args.batch,args.smoke)
            tag=f"{name}_{mode}"
            m,_=train(name,None,load,args.student_epochs,device,out,tag,
                      teacher if mode=="distilled" else None, teacher.layer4)
            rows.extend(deploy(m,tag,out,32,load[2],device))
            write_csv(out/"benchmark.csv",rows)

CHANNELS=(16,24,32,48,64,96,128)
KERNELS=(3,5,7)
OPS=("conv","dw")
ACTS=("relu","relu6","silu")
SIDES=(24,32,40,48)
LIMITS={"p50_ms":4.0,"bytes":1048576}

def sample(rng,parent=None):
    if parent is None:
        n=rng.randint(3,7)
        return {"channels":[rng.choice(CHANNELS) for _ in range(n)],
                "kernels":[rng.choice(KERNELS) for _ in range(n)],
                "ops":[rng.choice(OPS) for _ in range(n)],
                "activation":rng.choice(ACTS),"resolution":rng.choice(SIDES)}
    c=copy.deepcopy(parent)
    key=rng.choice(["channels","kernels","ops","activation","resolution","depth"])
    if key in ("channels","kernels","ops"):
        c[key][rng.randrange(len(c[key]))]=rng.choice({"channels":CHANNELS,"kernels":KERNELS,"ops":OPS}[key])
    elif key=="depth":
        n=len(c["channels"])
        if n>3 and (n==7 or rng.random()<.5):
            for k in ("channels","kernels","ops"): c[k].pop()
        else:
            for k,options in (("channels",CHANNELS),("kernels",KERNELS),("ops",OPS)): c[k].append(rng.choice(options))
    else: c[key]=rng.choice(ACTS if key=="activation" else SIDES)
    return c

def probe(cfg,out):
    import subprocess,sys
    path=out/"probe.pt"
    m,_,_=network("nas",cfg)
    torch.save({"network":"nas","config":cfg,"state":m.state_dict(),
                "validation_accuracy":0.,"epoch":0},path)
    data=json.loads(subprocess.check_output([sys.executable,__file__,"probe",
                  "--checkpoint",str(path)],text=True))
    path.unlink()
    return data

def nas(args,device,out):
    rng=random.Random(SEED); rows=[]; pool=[]; used=set()
    anchors=[dict(channels=[16,24,32],kernels=[3]*3,ops=["dw"]*3,activation="relu",resolution=24),
             dict(channels=[128]*7,kernels=[7]*7,ops=["conv"]*7,activation="silu",resolution=48)]
    write_json(out/"deployment_limits.json",LIMITS)
    for trial in range(args.trials):
        elites=sorted(pool,key=lambda r:r[0],reverse=True)[:4]
        parent=rng.choice(elites)[1] if trial>=6 and elites and rng.random()<.8 else None
        cfg=anchors[trial] if trial<len(anchors) else sample(rng,parent)
        while (key:=json.dumps(cfg,sort_keys=True)) in used: cfg=sample(rng,parent)
        used.add(key); size_latency=probe(cfg,out)
        reject=[k for k,v in LIMITS.items() if size_latency[k]>v]
        val=None
        if not reject:
            load=loaders(args.data,10,cfg["resolution"],args.batch,args.smoke)
            tag=f"candidate_{trial:02}"
            _,val=train("nas",cfg,load,args.candidate_epochs,device,out,tag)
            (out/f"{tag}.pt").unlink()
            pool.append((val,cfg,trial))
        row=dict(trial=trial,parent=json.dumps(parent,sort_keys=True),config=key,
                 eligible=not reject,rejection=";".join(reject),validation_accuracy=val,**size_latency)
        rows.append(row); write_csv(out/"search_history.csv",rows); print("NAS",row,flush=True)
    if not pool: raise RuntimeError("No architecture satisfied the predeclared limits")
    val,cfg,trial=max(pool,key=lambda r:r[0])
    write_json(out/"best_architecture.json",dict(trial=trial,proxy_validation_accuracy=val,config=cfg))
    load=loaders(args.data,10,cfg["resolution"],args.batch,args.smoke)
    write_json(out/"split.json",load[3])
    m,_=train("nas",cfg,load,args.nas_epochs,device,out,"nas_winner")
    write_csv(out/"benchmark.csv",deploy(m,"nas_winner",out,cfg["resolution"],load[2],device))
    metrics=probe_saved(out/"nas_winner.pt")
    write_json(out/"winner_constraint_check.json",{**metrics,"limits":LIMITS,
        "passes":all(metrics[k]<=v for k,v in LIMITS.items())})

def probe_saved(path):
    import subprocess,sys
    return json.loads(subprocess.check_output([sys.executable,__file__,"probe",
                      "--checkpoint",str(path)],text=True))

def open_model(path):
    cp=torch.load(path,map_location="cpu",weights_only=True)
    m,_,_=network(cp["network"],cp["config"]); m.load_state_dict(cp["state"])
    return m.eval(),cp

def infer(args):
    from PIL import Image
    classes=10 if args.arch=="nas" else 100
    if args.checkpoint.suffix==".onnx":
        import onnxruntime as ort
        options=ort.SessionOptions(); options.intra_op_num_threads=1; options.inter_op_num_threads=1
        sess=ort.InferenceSession(str(args.checkpoint),options,providers=["CPUExecutionProvider"])
        side=sess.get_inputs()[0].shape[-1]
        predict=lambda x:torch.from_numpy(sess.run(None,{"image":x.numpy()})[0])
    else:
        model,cp=open_model(args.checkpoint)
        cfg=cp["config"]; side=cfg["resolution"] if cfg else 32
        classes=10 if cp["network"]=="nas" else 100
        predict=model
    img=Image.open(args.image).convert("RGB").resize((32,32))
    with torch.inference_mode(): p=predict(transform(classes,side)(img).unsqueeze(0)).softmax(1)[0]
    print(json.dumps({"class_index":int(p.argmax()),"confidence":float(p.max())}))

def main():
    p=argparse.ArgumentParser()
    p.add_argument("task",choices=("kd","nas","infer","probe"))
    p.add_argument("--data",default="/tmp/assignment4-data")
    p.add_argument("--out",type=Path,default=Path("/kaggle/working/results"))
    p.add_argument("--batch",type=int,default=192)
    p.add_argument("--teacher-epochs",type=int,default=32)
    p.add_argument("--student-epochs",type=int,default=22)
    p.add_argument("--trials",type=int,default=18)
    p.add_argument("--candidate-epochs",type=int,default=4)
    p.add_argument("--nas-epochs",type=int,default=34)
    p.add_argument("--checkpoint",type=Path)
    p.add_argument("--image",type=Path)
    p.add_argument("--arch",choices=("student","nas"),default="student")
    p.add_argument("--smoke",action="store_true")
    p.add_argument("--require-gpu",action="store_true")
    args=p.parse_args()
    torch.set_num_threads(1); seed()
    if args.task=="infer": return infer(args)
    if args.task=="probe":
        model,cp=open_model(args.checkpoint)
        side=cp["config"]["resolution"]
        metrics=cpu_metrics(model,torch.randn(1,3,side,side))
        peak=None
        try:
            import resource
            peak=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024 if platform.system()=="Linux" else None
        except ImportError: pass
        return print(json.dumps({**metrics,"bytes":args.checkpoint.stat().st_size,"process_peak_rss_bytes":peak}))
    if args.smoke:
        args.teacher_epochs=args.student_epochs=args.candidate_epochs=args.nas_epochs=1
        args.trials=3; args.batch=8
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.require_gpu and device.type!="cuda": raise RuntimeError("Kaggle CUDA GPU is required")
    torch.backends.cudnn.benchmark=device.type=="cuda"
    args.out.mkdir(parents=True,exist_ok=True)
    import onnx,onnxruntime
    write_json(args.out/"environment.json",dict(python=platform.python_version(),torch=torch.__version__,
        platform=platform.platform(),cpu_threads=1,onnx=onnx.__version__,onnxruntime=onnxruntime.__version__,
        gpu=torch.cuda.get_device_name(0) if device.type=="cuda" else None,
        arguments={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()}))
    start=time.perf_counter()
    {"kd":kd,"nas":nas}[args.task](args,device,args.out)
    write_json(args.out/"RUN_STATUS.json",dict(complete=True,task=args.task,smoke=args.smoke,
                                             seconds=time.perf_counter()-start))
    print("RUN COMPLETE",args.task,flush=True)

if __name__=="__main__":
    main()
