import torch
import hashlib
import os

def file_hash(path):
    """Calculate MD5 hash of a file"""
    if not os.path.exists(path):
        return None
    with open(path, 'rb') as f:
        return hashlib.md5(f.read()).hexdigest()

# Compare all trigger files
paths = [
    "./ref_models/trigger_key_chain_28_100_10.pt",
    "../results3/key_chain/trigger_key_chain_28_100_10.pt",
    "./key_chain/trigger_key_chain_28_100_10.pt",
    "./key_chain/trigger_key_chain_28_10_10.pt",
    "./ref_models/trigger_key_chain_28_10_10.pt"
]

print("[CHECK] Trigger files comparison:")
print("="*70)

file_info = {}
for path in paths:
    if os.path.exists(path):
        try:
            pack = torch.load(path)
            size = pack['data'].shape[0]
            md5 = file_hash(path)
            file_info[path] = (size, md5)
            print(f"v {path}")
            print(f"  Samples: {size}")
            print(f"  MD5: {md5[:16]}...")
            print()
        except Exception as e:
            print(f"x {path}: {e}\n")
    else:
        print(f"x {path}: NOT FOUND\n")

print("="*70)
print("\n[ANALYSIS]")
hashes = [v[1] for v in file_info.values() if v[1]]
if len(set(hashes)) == 1 and len(hashes) > 0:
    print("v All existing files are IDENTICAL (same MD5)")
else:
    print("x Files are DIFFERENT or some missing:")
    for path, info in file_info.items():
        size, md5 = info
        print(f"  {path}: {size} samples, hash={md5[:16]}...")
