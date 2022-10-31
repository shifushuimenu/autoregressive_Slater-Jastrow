import numpy as np
import torch 
from torchviz import make_dot

class Sdet(torch.nn.Module):
    def __init__(self, D=10):
        super(Sdet, self).__init__()
        self.T = torch.nn.Parameter(torch.triu(torch.zeros(D,D), diagonal=-1))
        # random initialization 
        T0 = torch.triu(torch.randn(D,D))
        self.U0 = torch.matrix_exp(T0 - T0.t())

        self.rotate_orbitals()

    def rotate_orbitals(self):
        self.R = torch.matrix_exp(self.T - self.T.t())
        self.U_rot = self.R @ self.U0 @ self.R.t()


torch.manual_seed(42)

num_steps=1000
eta = 0.01

model=Sdet(D=10)
optimizer = torch.optim.RMSprop(model.parameters(), lr=0.01)
#optimizer = torch.optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-8)

loss_array = np.zeros(num_steps)

for ii in range(num_steps):

    loss = torch.sum(model.U_rot.flatten())
    if ii == 1:
        viz_graph = make_dot(loss)
        viz_graph.view()

    loss_array[ii] = loss.item()

    # model.zero_grad()
    optimizer.zero_grad()
    loss.backward(retain_graph=False)
    optimizer.step()

    print("loss=", loss)
    #model.T.data -= eta * model.T.grad.data
    print(model.T.data[0,4])
    model.rotate_orbitals()

np.savetxt("loss_RMSprop.dat", loss_array)
