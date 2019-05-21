import numpy as np

class Function(object):
    """
    Represents a node in computational graph that perfoms
    a computation.

    During forward mode computation, it takes in
    1 or more inputs/parents, and returns a result of the
    computation as the output.

    In reverse mode accumulation, it takes in the
    gradients w.r.t. the output of the node, accumulates them
    by calculating the gradients w.r.t it's inputs/parents
    and back-propogates the gradients to the parents.
    """
    # List of inputs to the node.
    parents = []

    def __init__(self):
        pass

    def forward(self, *args):
        """
        Forward mode computation of the 'operation'
        on the inputs/parents to be implemented here.
        """
        pass

    def backward(self, gradient):
        """
        Reverse mode computation of the node implemented here.
        """
        pass


class Add(Function):
    """
    Add parent inputs and return the result.
    """
    def forward(self, *args):
        """
        Forward computation of the binary Add operation.

        param:
        args (n=2 Tensors): 2 Tensors to be added.

        returns:
        value (ndarray): Result of "+" operation on
            input args.
        """
        # Extend this nodes' parents to include new inputs.
        self.parents = list(args)
        # Add the 2 input Tensor's values
        value = self.parents[0].value + self.parents[1].value
        return value

    def backward(self, gradient):
        """
        Accumulates the gradients for "Add" operation from its'
        children/outputs and passes them on to self.parents.

        param:
        gradient (ndarray or scalar): gradient w.r.t output of
        "Add"

        returns:
        None
        """
        # Accumulate gradient
        self.parents[0].grad += gradient
        self.parents[1].grad += gradient


class Sub(Function):
    """
    Subtract parent inputs and return the result.
    """
    def forward(self, *args):
        """
        Forward computation of the binary Subtract operation.

        param:
        args (n=2 Tensors): 2 Tensors to be subtracted.

        returns:
        value (ndarray): Result of "-" operation on
            input args.
        """
        # Extend this nodes' parents to include new inputs.
        self.parents = list(args)
        # Subtract the 2 input Tensor's values
        value = self.parents[0].value - self.parents[1].value
        return value

    def backward(self, gradient):
        """
        Accumulates the gradients for "Subtract" operation from its'
        children/outputs and passes them on to self.parents.

        param:
        gradient (ndarray or scalar): gradient w.r.t output of
        "Sub"

        returns:
        None
        """
        # Accumulate gradient
        self.parents[0].grad += gradient
        self.parents[1].grad -= gradient

class Mul(Function):
    """
    Multiply parent inputs and return the result.
    """
    def forward(self, *args):
        """
        Forward computation of the binary Multiply operation.

        param:
        args (n=2 Tensors): 2 Tensors to be multiplied.

        returns:
        value (ndarray): Result of "*" operation on
            input args.
        """
        # Extend this nodes' parents to include new inputs.
        self.parents = list(args)
        # Multiply the 2 input Tensor's values
        value = self.parents[0].value * self.parents[1].value
        return value

    def backward(self, gradient):
        """
        Accumulates the gradients for "Multiply" operation from its'
        children/outputs and passes them on to self.parents.

        param:
        gradient (ndarray or scalar): gradient w.r.t output of
        "Mul"

        returns:
        None
        """
        # Accumulate gradient
        self.parents[0].grad += np.multiply(gradient, self.parents[1].value)
        self.parents[1].grad += np.multiply(gradient, self.parents[0].value)

class Div(Function):
    """
    Divide parent inputs and return the result.
    """
    def forward(self, *args):
        """
        Forward computation of the binary Divide operation.

        param:
        args (n=2 Tensors): 2 Tensors to be multiplied.

        returns:
        value (ndarray): Result of "/" operation on
            input args.
        """
        # Extend this nodes' parents to include new inputs.
        self.parents = list(args)
        # Divide the 2 input Tensor's values
        value = self.parents[0].value / self.parents[1].value
        return value

    def backward(self, gradient):
        """
        Accumulates the gradients for "Divide" operation from its'
        children/outputs and passes them on to self.parents.

        param:
        gradient (ndarray or scalar): gradient w.r.t output of
        "Div"

        returns:
        None

        """
        # Accumulate gradient
        self.parents[0].grad += np.multiply(
                gradient, self.parents[1].value/self.parents[1].value**2
            )
        self.parents[1].grad -= np.multiply(
                gradient, self.parents[0].value/self.parents[1].value**2
            )