"""Calibration: measure val/train on a checkpoint with 30 batches.

Purpose: decide whether the 7.84 reading on ckpt.pt (iter 950) is REAL
(divergence) or a measurement bug. best.pt (iter 156, stored 4-batch val
6.3559) is the control: if this script reads ~6.4 on it, the script is sound.
"""
import sys, time, os
import torch
import numpy as np

os.chdir('/home/eneru/litmodel')
sys.path.insert(0, '.')
from config import Config
from model import GPT

which = sys.argv[1]
seed = int(sys.argv[2]) if len(sys.argv) > 2 else 999

ck = torch.load(f'out/{which}', map_location='cpu', weights_only=False)
c = Config(**ck['config'])
m = GPT(c)
m.load_state_dict(ck['model'])
m.eval()
torch.set_num_threads(2)
torch.manual_seed(seed)

tr = np.memmap('data/train.bin', dtype=np.uint16, mode='r')
va = np.memmap('data/val.bin', dtype=np.uint16, mode='r')

def batch(d):
    j = int(torch.randint(len(d) - c.block_size - 1, (1,)))
    x = torch.from_numpy(d[j:j + c.block_size].astype(np.int64)).unsqueeze(0)
    y = torch.from_numpy(d[j + 1:j + 1 + c.block_size].astype(np.int64)).unsqueeze(0)
    return x, y

@torch.no_grad()
def est(d, n):
    v = []
    for _ in range(n):
        x, y = batch(d)
        v.append(m(x, y).loss.item())
    return sum(v) / n, min(v), max(v)

t0 = time.time()
tv, vlo, vhi = est(va, 30)
tt, tlo, thi = est(tr, 30)
iter_s = ck.get('iter', '?')
best = ck.get('best_val', float('nan'))
from datetime import datetime
line = (f"{datetime.now().strftime('%F %T')} {which:20s} iter {iter_s:>5} seed {seed} "
        f"| VAL {tv:.4f} [{vlo:.2f},{vhi:.2f}] | TRAIN {tt:.4f} [{tlo:.2f},{thi:.2f}] "
        f"| stored_best {best:.4f} | {time.time()-t0:.0f}s")
print(line)
# every measurement is archived -- operator should never have to paste terminal
os.makedirs('logs', exist_ok=True)
with open('logs/eval.log', 'a') as f:
    f.write(line + '\n')
