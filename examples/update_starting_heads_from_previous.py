"""Update model starting heads from previous (initial steady-state) solution
"""
from pathlib import Path

import numpy as np
from flopy.utils import binaryfile as bf

model_ws = Path('.')
headfile = model_ws / 'model.hds'
starting_heads_file_fmt = str(model_ws / 'external/strt_{:03d}.dat')


hdsobj = bf.HeadFile(headfile)
print(f'reading {headfile}...')

initial_ss_heads = hdsobj.get_data(kstpkper=(0, 0))
for per, layer_heads in enumerate(initial_ss_heads):
    outfile = starting_heads_file_fmt.format(per)
    np.savetxt(starting_heads_file_fmt.format(per), layer_heads, fmt='%.2f')
    print(f"updated {outfile}")
