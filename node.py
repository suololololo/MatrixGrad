
"""
    define the node in computational graph
"""
import numpy as np
from grad import *

class Node:

    def __init__(self, tensor, requires_grad=False) -> None:
        self.tensor = np.array(tensor)
        self.requires_grad = requires_grad
        self.grad = 0.0
        self.grad_fn = None
        self.lchild = None
        self.rchild = None
        self.father = None
        self.is_leaf = True

    def __add__(self, value):
        """define the left add operation of node 

        Args:
            value (list,np.array, Node): the right value to add 

        Return:
            the new node add by this node and value
        """
        if isinstance(value, (list, np.ndarray)):
            value = Node(value)

        new_tensor = self.tensor + value.tensor
        new_node = Node(new_tensor)
        new_node.requires_grad = self.requires_grad or value.requires_grad
        new_node.grad_fn = grad_add

        new_node.lchild = self
        new_node.rchild = value
        new_node.is_leaf = False

        new_node.lchild.father = new_node
        new_node.rchild.father = new_node


        return new_node

    def __radd__(self, value):
        return self.__add__(value)

    def __sub__(self, value):
        """define the left sub operation of node 

        Args:
            value (list,np.array, Node): the right value to sub 

        Return:
            the new node sub by this node and value
        """
        if isinstance(value, (list, np.ndarray)):
            value = Node(value)

        new_tensor = self.tensor - value.tensor
        new_node = Node(new_tensor)
        new_node.requires_grad = self.requires_grad or value.requires_grad

        new_node.grad_fn = grad_sub
        new_node.lchild = self
        new_node.rchild = value
        new_node.is_leaf = False

        new_node.lchild.father = new_node
        new_node.rchild.father = new_node

        return new_node

    def __rsub__(self, value):
        """define the right sub operation of node 

        Args:
            value (list,np.array, Node): the left value to sub 

        Return:
            the new node sub by this node and value
        """

        if isinstance(value, (list, np.ndarray)):
            value = Node(value)
        elif isinstance(value, (int, float)):
            value = Node(np.ones(self.tensor.shape, dtype=np.float) * value)

        new_tensor = value.tensor - self.tensor
        new_node = Node(new_tensor)
        new_node.requires_grad = self.requires_grad or value.requires_grad

        new_node.grad_fn = grad_sub
        new_node.lchild = value
        new_node.rchild = self
        new_node.is_leaf = False

        new_node.lchild.father = new_node
        new_node.rchild.father = new_node

        return new_node


    def __mul__(self, value):
        """define the right mul operation of node 

        Args:
            value (list,np.array, Node): the right value to mul 

        Return:
            the new node mul by this node and value
        """

        if isinstance(value, (list, np.ndarray)):
            value = Node(value)

        if isinstance(value, (int, float)):
            value = Node(np.ones(self.tensor.shape, dtype=np.float) * value)

        new_tensor =  np.matmul(self.tensor, value.tensor)
        new_node = Node(new_tensor)
        new_node.requires_grad = self.requires_grad or value.requires_grad
        new_node.grad_fn = grad_mul
        new_node.lchild = self
        new_node.rchild = value
        new_node.is_leaf = False

        new_node.lchild.father = new_node
        new_node.rchild.father = new_node

        return new_node        

    def __rmul__(self, value):
        """define the right mul operation of node 

        Args:
            value (list,np.array, Node): the right value to mul 

        Return:
            the new node mul by this node and value
        """

        if isinstance(value, (list, np.ndarray)):
            value = Node(value)

        elif isinstance(value, (int, float)):
            value = Node(np.ones(self.tensor.shape, dtype=np.float) * value)

        new_tensor =  np.matmul(value.tensor, self.tensor)
        new_node = Node(new_tensor)
        new_node.requires_grad = self.requires_grad or value.requires_grad
        new_node.grad_fn = grad_mul
        new_node.lchild = value
        new_node.rchild = self
        new_node.is_leaf = False

        new_node.lchild.father = new_node
        new_node.rchild.father = new_node

        return new_node            

    def relu(self):
        """define relu operation of node 

        Returns:
            the new node pass by  relu
        """
        new_tensor = np.where(self.tensor > 0, self.tensor, 0)

        new_node = Node(new_tensor)

        new_node.requires_grad = self.requires_grad
        new_node.grad_fn = grad_relu 
        new_node.lchild = self
        new_node.rchild = None
        new_node.is_leaf = False

        new_node.lchild.father = new_node

        return new_node
    def T(self):
        new_tensor = self.tensor.T.copy()
        new_node = Node(new_tensor)

        new_node.requires_grad = self.requires_grad
        new_node.grad_fn = grad_T
        new_node.is_leaf = False
        new_node.lchild = self
        new_node.lchild.father = new_node
        new_node.rchild = None


        return new_node

    def __sum__(self, axis=0):
        """define sum operation of node 
            only two axis is supported
        Returns:
            the new node pass by sum
        """
        if axis == 0:
            value = np.ones((1, self.tensor.shape[0]))
            new_node = self.__rmul__(value)
        elif axis == 1:
            value = np.ones((self.tensor.shape[1], 1))
            new_node = self.__mul__(value)

        return new_node
    def __neg__(self):
        """
        define the negative operation of node
        """

        new_tensor = - self.tensor
        new_node = Node(new_tensor)

        new_node.requires_grad = self.requires_grad
        new_node.grad_fn = grad_neg
        new_node.lchild = self
        new_node.is_leaf = False

        new_node.lchild.father = new_node

        return new_node

    def backward(self, grad=np.ones((1,1), dtype=np.float64)):
        """
        define the backward process to calculate grad
        """
        if not self.requires_grad:
            return
        self.grad += grad
        if not self.is_leaf:
            left_grad, right_grad = self.grad_fn(grad, self.lchild, self.rchild)
            if self.lchild is not None:
                if self.lchild.requires_grad:
                    self.lchild.backward(left_grad)
            if self.rchild is not None:
                if self.rchild.requires_grad:
                    self.rchild.backward(right_grad)
            
    def zeros_grad(self):
        """
        define the operation to clear grad
        """
        self.grad = 0.0
        if self.lchild is not None:
            self.lchild.zeros_grad()
        if self.rchild is not None:
            self.rchild.zeros_grad()
