# import re

# MODEL_TO_LOAD = 'Agents/NFSP_Model/id=1_steps=2000'
# steps = int(re.findall(r'steps=(\d*)', MODEL_TO_LOAD)[-1])
# print(steps)

import numpy as np

for i in range(100):
    print(np.random.normal(100, 50))
