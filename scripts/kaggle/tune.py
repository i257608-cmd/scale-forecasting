# Upload: zip of the repo (code + data/features) as the private Kaggle dataset named in kernel-metadata.json,
# then `kaggle kernels push -p scripts/kaggle`. The Kaggle API token lives in ~/.kaggle (never in the repo).
# Kaggle GPU runner for experiments/e1_tune.py (Optuna, equal budget). Results: /kaggle/working/results/tuning/*.json
import glob, os, shutil, subprocess, sys, zipfile
MODE = "FULL"   # SMOKE = 1 quick trial to test the setup; FULL = real run
W = "/kaggle/working/sf"
src = [p for p in glob.glob("/kaggle/input/**/scale_forecasting.zip", recursive=True)]
if src:                                   # dataset kept as zip
    zipfile.ZipFile(src[0]).extractall(W)
else:                                     # Kaggle auto-extracted the zip
    d = os.path.dirname(glob.glob("/kaggle/input/**/experiments/e1_tune.py", recursive=True)[0])
    shutil.copytree(os.path.dirname(d), W, dirs_exist_ok=True)
# keep the tabular cache valid (newer than windows.parquet) so it is not rebuilt
os.utime(f"{W}/data/features/tabular.parquet")
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "optuna"], check=False)
import torch; print("torch", torch.__version__, "cuda", torch.cuda.is_available(),
                   torch.cuda.get_device_name(0) if torch.cuda.is_available() else "")
if MODE == "SMOKE":
    cmd = ["--models", "B3", "PIPS", "--trials", "1", "--quick"]
    r = subprocess.run([sys.executable, "experiments/e1_tune.py", *cmd], cwd=W)
else:   # 2 x T4: one tuning process per GPU, disjoint model lists (one JSON per model -> no clash)
    # Runs used (2026-11-16): version 3 = {"0": ["B7", "PIPS"], "1": ["B8", "B8i", "B9"]} (B1-B4 were tuned
    # locally on the M4), version 4 = {"0": ["B5"], "1": ["B6"]} (moved off the M4: up to 3.5 h per fold there).
    jobs = {"0": ["B5"], "1": ["B6"]}
    ps = [subprocess.Popen([sys.executable, "experiments/e1_tune.py", "--models", *ms, "--trials", "20",
                            "--max-hours", "9.5"], cwd=W, env=os.environ | {"CUDA_VISIBLE_DEVICES": g},
                           stdout=open(f"/kaggle/working/tune_gpu{g}.log", "w"), stderr=subprocess.STDOUT)
          for g, ms in jobs.items()]
    codes = [p.wait() for p in ps]
    print("exit codes", codes, flush=True)
    for g in jobs:
        print(open(f"/kaggle/working/tune_gpu{g}.log").read()[-3000:], flush=True)
    r = subprocess.CompletedProcess([], max(codes))
os.makedirs("/kaggle/working/results", exist_ok=True)
shutil.copytree(f"{W}/results/tuning", "/kaggle/working/results/tuning", dirs_exist_ok=True)
shutil.rmtree(W)                          # outputs = only the tuning JSONs
sys.exit(r.returncode)
