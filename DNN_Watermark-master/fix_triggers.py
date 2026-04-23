import shutil
import torch
import os

print("[FIX] Restoring Transfer trigger (1000 samples)...\n")

# Copy the trigger file from results3 to the current directory
src = "../results3/key_chain/trigger_key_chain_28_100_10.pt"
dst_keychain = "./key_chain/trigger_key_chain_28_100_10.pt"
dst_ref = "./ref_models/trigger_key_chain_28_100_10.pt"

if os.path.exists(src):
    # Copy to key_chain
    shutil.copy(src, dst_keychain)
    pack = torch.load(dst_keychain)
    print(f"v Restored {dst_keychain}")
    print(f"  Samples: {pack['data'].shape[0]}")
    print(f"  Unique labels: {torch.unique(pack['target']).tolist()}\n")
    
    # Copy to ref_models
    shutil.copy(dst_keychain, dst_ref)
    print(f"v Also copied to {dst_ref}\n")
else:
    print(f"x Source not found: {src}\n")

# List current trigger files
print("[INFO] Current trigger files:")
keychain_dir = "./key_chain/"
if os.path.exists(keychain_dir):
    for f in sorted(os.listdir(keychain_dir)):
        if f.startswith("trigger_") and f.endswith(".pt"):
            pack = torch.load(os.path.join(keychain_dir, f))
            print(f"  {f}: {pack['data'].shape[0]} samples")
