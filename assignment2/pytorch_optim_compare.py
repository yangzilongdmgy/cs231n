import torch

N, D_in, H, D_out = 64, 1000, 100, 10
x = torch.randn(N, D_in)
y = torch.randn(N, D_out)

model = torch.nn.Sequential(
        torch.nn.Linear(D_in, H),
        torch.nn.ReLU(),
        torch.nn.Linear(H, D_out))

learning_rate = 1e-2
for t in range(500):
    print("epoch: {}".format(t))
    y_pred = model(x)
    loss = torch.nn.functional.mse_loss(y_pred, y)
    print(loss)
    loss.backward()
    count = 0
    with torch.no_grad():
        for param in model.parameters():
            param -= learning_rate * param.grad
    model.zero_grad()
