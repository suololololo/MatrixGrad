from node import Node
import numpy as np
class NetWork:
    def __init__(self, hidden_size, activate='relu') -> None:
        self.hidden_size = hidden_size
        self.activate = activate
        self.weight = []
        self.bias = []
        self.activate = activate
    def fit(self, x, y):
        """_summary_

        Args:
            x (np.ndarray): _description_
            y (np.ndarray): _description_
        """
        x = Node(x)
        y = Node(y)
        self.hidden_size.insert(0, x.tensor.shape[-1])

        for i in range(len(self.hidden_size)):
            if i == len(self.hidden_size)-1:
                w = np.random.uniform(-1, 1, (self.hidden_size[i], 1))
                self.weight.append(Node(w, requires_grad=True))
                b = np.random.uniform(-1, 1, (1, 1))
                self.bias.append(Node(b, requires_grad=True))
            else:
                w = np.random.uniform(-1, 1, (self.hidden_size[i], self.hidden_size[i+1]))
                self.weight.append(Node(w, requires_grad=True))
                b = np.random.uniform(-1, 1, (1, self.hidden_size[i+1]))
                self.bias.append(Node(b, requires_grad=True))
            
        for epoch in range(20):
            for i in range(len(self.hidden_size)):
                if i == 0:
                    H = x * self.weight[i] + Node(np.ones((x.tensor.shape[0], 1))) * self.bias[i]
                    H = H.relu()
                elif i == len(self.hidden_size)-1:
                    output = H * self.weight[i] + Node(np.ones((H.tensor.shape[0], 1))) * self.bias[i]
                else:
                    H = H * self.weight[i] + Node(np.ones((H.tensor.shape[0], 1))) * self.bias[i]
                    H = H.relu()

            loss = (y - output).T() *(y - output)
            print("Epoch: {}, loss: {:.4}".format(epoch, loss.tensor[0][0]))
            loss.backward()

            for w in self.weight:
                w.tensor = w.tensor - 0.001 * w.grad
                w.zeros_grad()

            for b in self.bias:
                b.tensor = b.tensor - 0.001 * b.grad
                b.zeros_grad()