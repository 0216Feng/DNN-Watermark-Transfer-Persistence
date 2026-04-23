import os
import torch

def check_triggers():
    key_chain_dir = "./key_chain/"
    print("[CHECK] All trigger files in key_chain/:")
    if os.path.exists(key_chain_dir):
        for f in sorted(os.listdir(key_chain_dir)):
            if f.endswith(".pt"):
                try:
                    pack = torch.load(os.path.join(key_chain_dir, f))
                    print(f"  {f}: {pack['data'].shape[0]} samples, labels={torch.unique(pack['target']).tolist()}")
                except Exception as e:
                    print(f"  Error loading {f}: {e}")
    
    trigger_file = "ref_models/trigger_key_chain_28_100_10.pt"
    if os.path.exists(trigger_file):
        try:
            pack = torch.load(trigger_file)
            print(f"\n[CHECK] Specified trigger file {trigger_file}:")
            print(f"  Shape: {pack['data'].shape}")
            unique_targets, counts = torch.unique(pack['target'], return_counts=True)
            print(f"  Target distribution: {list(zip(unique_targets.tolist(), counts.tolist()))}")
            print(f"  Total samples: {pack['data'].shape[0]}")
        except Exception as e:
            print(f"\n[ERROR] Loading specified trigger file: {e}")
    else:
        print(f"\n[INFO] Specified trigger file {trigger_file} not found.")

if __name__ == '__main__':
    check_triggers()
