def train(device, train_loader, input_size, model, optimizer, loss_function):
    # 训练模式
    model.train()
    epoch_train_loss = 0
    for i, (images, labels) in enumerate(train_loader):
        # 梯度清零
        optimizer.zero_grad()
        # 前向传播
        output = model(images.to(device))
        # 计算loss，优化目标
        batch_loss = loss_function(output, labels.to(device))

        # 反向传播与优化
        batch_loss.backward()
        optimizer.step()
        # 累加损失
        epoch_train_loss += batch_loss.item()
    # 计算平均训练损失，每个epoch的长度为所有数据
    train_avg_loss = epoch_train_loss / len(train_loader)
    return train_avg_loss
