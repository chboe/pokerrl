import torch


model_strings = [
    "Agents/NFSP_Model/id=2300_steps=3700000_target.model",
    "Agents/NFSP_Model/id=2300_steps=3700000_avg.model",
    "Agents/NFSP_Model/id=2301_steps=3550000_target.model",
    "Agents/NFSP_Model/id=2301_steps=3550000_avg.model"
]

for name in model_strings:
    print(f'Checking model={name}')
    model = torch.load(name)
    for param in model.parameters():
        if param.data.isnan().any():
            print(f'NaNs found in {name}')
            