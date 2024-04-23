import torch


def test(device, test_loader, model, loss_function):
    # 评估模型
    model.to(device)
    model.eval()
    epoch_valid_loss = 0.0
    total_correct = 0
    total_sample = 0
    # test禁止优化
    with torch.no_grad():
        for i, (vimages, vlabels) in enumerate(test_loader):
            voutput = model(vimages.to(device))
            # loss
            vbatch_loss = loss_function(voutput, vlabels.to(device))
            epoch_valid_loss += vbatch_loss

            # accrucy
            _, predicted = torch.max(voutput, dim=1)
            total_correct += (predicted == vlabels.to(device)).sum().item()
            total_sample += vlabels.size(0)
    vaild_avg_loss = epoch_valid_loss / len(test_loader)
    valid_accurcy = total_correct / total_sample
    # 保存模型

    return vaild_avg_loss, valid_accurcy
