import torch


model_strings = [
    "Agents/NFSP_Model/id=2000_steps=6200000_target.model",
    "Agents/NFSP_Model/id=2000_steps=6200000_avg.model",
    "Agents/NFSP_Model/id=2001_steps=6150000_target.model",
    "Agents/NFSP_Model/id=2001_steps=6150000_avg.model"
]

for name in model_strings:
    print(f'Checking model={name}')
    model = torch.load(name)
    for param in model.parameters():
        if param.data.isnan().any():
            print(f'NaNs found in {name}')
            