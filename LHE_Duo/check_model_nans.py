import torch


model_strings = [
    "Agents/NFSP_Model/id=1900_steps=1250000_target.model",
    "Agents/NFSP_Model/id=1900_steps=1250000_avg.model",
    "Agents/NFSP_Model/id=1901_steps=1250000_target.model",
    "Agents/NFSP_Model/id=1901_steps=1250000_avg.model"
]

for name in model_strings:
    print(f'Checking model={name}')
    model = torch.load(name)
    for param in model.parameters():
        if param.data.isnan().any():
            print(f'NaNs found in {name}')