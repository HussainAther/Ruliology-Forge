#!/usr/bin/env python3
"""Re-run the primary benchmark and the 88-class sensitivity analysis and compare with the shipped outputs.
Usage (from this folder):  python scripts/verify_reproduction.py
Takes about 15 seconds. Exits non-zero if any shipped value is not reproduced."""
import subprocess, sys, tempfile
from pathlib import Path
import numpy as np, pandas as pd

root = Path(__file__).resolve().parent.parent
tmp = Path(tempfile.mkdtemp())
subprocess.run([sys.executable, str(root / "scripts" / "run_restoration_benchmark_v2.py"), "--out", str(tmp)],
               check=True, stdout=subprocess.DEVNULL)
ok = True
for f in ["rule_metrics_aggregated.csv", "rule_metrics_by_perturbation_time.csv",
          "symmetry_validation.csv", "timing_rank_stability.csv"]:
    a, b = pd.read_csv(tmp / f), pd.read_csv(root / "data" / "primary_benchmark" / f)
    num = a.select_dtypes("number").columns
    same = a.shape == b.shape and list(a.columns) == list(b.columns) and \
        np.allclose(a[num].values, b[num].values, rtol=0, atol=1e-9, equal_nan=True) and \
        a.drop(columns=num).equals(b.drop(columns=num))
    print(("OK   " if same else "FAIL ") + f); ok &= same
r = subprocess.run([sys.executable, str(root / "scripts" / "eca_equivalence_class_analysis.py"),
                    str(root / "data/analysis_outputs/rule_metrics_with_wolfram_class.csv"),
                    "--metric-col", "relative_restoration_mean", "--out", str(tmp / "eq.csv")],
                   capture_output=True, text=True, check=True)
same = np.allclose(pd.read_csv(tmp / "eq.csv").select_dtypes("number").values,
                   pd.read_csv(root / "data/analysis_outputs/equivalence_class_sensitivity.csv").select_dtypes("number").values,
                   rtol=0, atol=1e-9)
print(("OK   " if same else "FAIL ") + "equivalence_class_sensitivity.csv"); ok &= same
sys.exit(0 if ok else 1)
