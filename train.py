def train(device, epochs, train_loader, input_size, model, optimizer, loss_function):
    print("此模型在", device, "上训练")
    for epoch in range(epochs):
        # 训练模式
        model.train()
        epoch_train_loss = 0
        for i, (images, labels) in enumerate(train_loader):
            # 梯度清零
            optimizer.zero_grad()
            # 前向传播
            image = images.reshape(-1, input_size).to(device)
            label = labels.to(device)
            output = model(image)
            batch_loss = loss_function(output, label)

            # 反向传播与优化
            batch_loss.backward()
            optimizer.step()
            epoch_train_loss += batch_loss.item()
        # 计算平均训练损失
        train_avg_loss = epoch_train_loss / len(train_loader)
        # # 评估模型
        # model.eval()
        # validation_loss = 0.0
        # # with torch.no_grad():
        print("epoch", epoch, "\t", "loss", train_avg_loss)
