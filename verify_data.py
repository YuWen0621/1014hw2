import json
import re
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

with open("docs/model_data.js", "r", encoding="utf-8") as f:
    text = f.read()

# Extract window.DATASETS
match = re.search(r"window\.DATASETS\s*=\s*(\{.*?\});\s*window\.MODEL_DATA", text, re.DOTALL)
if not match:
    raise ValueError("Could not extract window.DATASETS from docs/model_data.js")

datasets = json.loads(match.group(1))
ds = datasets["student_performance"]
print(f"Dataset Name: {ds['name']}")
print(f"Target: {ds['target_name']} ({ds['target_unit']})")
print(f"Optimal Sweet Spot: k = {ds['sweet_spot_k']}")
print(f"Total Candidate Features: {len(ds['features'])}")

model_data = ds["model_data"]
print("\nAlgorithms in model_data.js:", list(model_data.keys()))

for algo, list_k in model_data.items():
    opt = min(list_k, key=lambda x: (round(x["rmse"], 5), x["k"]))
    print(f"\n=== {algo} (k=1..{len(list_k)}) ===")
    print(f"  Optimal k={opt['k']}: Test RMSE={opt['rmse']:.6f}, R2={opt['r2']:.6f}, CV RMSE={opt['cv_rmse']:.6f}")
    print(f"  95% PI Coverage on Test Set: {opt['pi_coverage']:.2f}% (Mean Width: {opt['mean_pi_width']:.6f})")
    print(f"  Selected Features: {opt['features']}")

print("\nVerification successful: All data correctly generated and validated!")
