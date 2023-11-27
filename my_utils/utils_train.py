import torch
from torch import nn

def test_model(model, dataloader, config):
    total = 0
    correct = 0 
    error = nn.CrossEntropyLoss()
    with torch.no_grad():
        for idx, (data, target) in enumerate(dataloader):
            data, target = data.to(config['device']), target.to(config['device'])
            outputs = model(data)
            _, predicted = torch.max(outputs.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    # print('acc:', correct, total)
    return 100 * correct/total