from model import NetWork
from node import Node
import numpy as np

def test_model():
    mlp = NetWork([5, 3], activate="relu")
    x = np.random.uniform(-3.14, 3.14, (100, 2))
    y = np.sin(x[:, 0]) + np.cos(x[:, 1])
    y = y.reshape((y.shape[0], 1))
    
    mlp.fit(x, y)

if __name__ == "__main__":
    w1 = np.random.uniform(-1, 1, (2, 5))
    w2 = np.random.uniform(-1, 1, (5, 1))
    b1 = np.random.uniform(-1, 1, (1, 5))
    b2 = np.random.uniform(-1, 1, (1, 1))
    x = np.random.uniform(-3.14, 3.14, (100, 2))
    y = np.sin(x[:,0]) + np.cos(x[:,1])

    y = y.reshape((y.shape[0], 1))

    W1 = Node(w1, requires_grad=True)
    W2 = Node(w2, requires_grad=True)
    B1 = Node(b1, requires_grad=True)
    B2 = Node(b2, requires_grad=True)

    H1 = Node(x) * W1 + Node(np.ones((x.shape[0], 1))) *B1
    H1 = H1.relu()
    out = H1 * W2 + Node(np.ones((x.shape[0],1))) * B2

    loss = (Node(y)- out).T() * (Node(y) - out)

    loss.backward()
    
    print("Node W1: ")
    print("{}".format(" ".join(str(e) for e in W1.tensor[0])))
    print("{}".format(" ".join(str(e) for e in W1.tensor[1])))
    
    print("Node W2: ")
    print("{}".format(" ".join(str(e) for e in W2.tensor)))

    print("Node B1: ")
    print("{}".format(" ".join(str(e) for e in B1.tensor)))

    print("Node B2: ")
    print("{}".format(" ".join(str(e) for e in B2.tensor)))



