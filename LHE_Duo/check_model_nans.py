import torch


model_strings = [
    "Agents/NFSP_Model/id=2100_steps=422500_target.model",
    "Agents/NFSP_Model/id=2100_steps=422500_avg.model",
    "Agents/NFSP_Model/id=2101_steps=417500_target.model",
    "Agents/NFSP_Model/id=2101_steps=417500_avg.model"
]

for name in model_strings:
    print(f'Checking model={name}')
    model = torch.load(name)
    for param in model.parameters():
        if param.data.isnan().any():
            print(f'NaNs found in {name}')
            