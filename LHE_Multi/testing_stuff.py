import re

MODEL_TO_LOAD = 'Agents/NFSP_Model/id=1_steps=2000'
steps = int(re.findall(r'steps=(\d*)', MODEL_TO_LOAD)[-1])
print(steps)

