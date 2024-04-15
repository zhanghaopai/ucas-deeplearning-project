import torch


def train(device, epochs, train_loader, test_loader, input_size, model, optimizer, loss_function):
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
            # 累加损失
            epoch_train_loss += batch_loss.item()
        # 计算平均训练损失，每个epoch的长度为所有数据
        train_avg_loss = epoch_train_loss / len(train_loader)

        # 评估模型
        model.eval()
        epoch_valid_loss = 0.0
        # test禁止优化
        with torch.no_grad():
            for i, (vimages, vlabels) in enumerate(test_loader):
                vimage = vimages.reshape(-1, input_size)
                voutput = model(vimage)
                vbatch_loss = loss_function(voutput, vlabels)
                epoch_valid_loss += vbatch_loss
        vaild_avg_loss = epoch_valid_loss / len(test_loader)
        print("epoch: {}, train_loss: {}, test_loss: {}".format(epoch + 1, train_avg_loss, vaild_avg_loss))
