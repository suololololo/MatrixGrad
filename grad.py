import numpy as np
def grad_add(grad, lchlid, rchild):
    """
    the function to caculate grad of add operation

    Args:
        grad (_type_): gradient of predcessor node
        lchlid (_type_): left node 
        rchild (_type_): right node
    """
    return grad, grad

def grad_sub(grad, lchild, rchild):
    """
    the function to caculate grad of sub operation

    Args:
        grad (_type_): gradient of predcessor node
        lchlid (_type_): left node 
        rchild (_type_): right node
    """
    return grad, -grad

def grad_mul(grad, lchild, rchild):
    """
    the function to caculate grad of mul operation

    Args:
        grad (_type_): gradient of predcessor node
        lchlid (_type_): left node 
        rchild (_type_): right node
    """
    return np.matmul(grad, rchild.tensor.T), np.matmul(lchild.tensor.T, grad)

def grad_relu(grad, lchild, rchild):
    """
    the function to caculate grad of relu operation

    Args:
        grad (_type_): gradient of predcessor node
        lchlid (_type_): left node 
        rchild (_type_): right node
    """
    tmp_tensor = np.where(lchild.tensor > 0, 1.0, 0.0)
    return np.multiply(grad, tmp_tensor), None

def grad_neg(grad, lchild, rchild):
    return -grad, None

def grad_T(grad, lchild, rchild):
    return grad.T, None