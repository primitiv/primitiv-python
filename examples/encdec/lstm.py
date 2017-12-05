# Hand-written LSTM with input/forget/output gates and no peepholes.
# Formulation:
#   i = sigmoid(W_xi . x[t] + W_hi . h[t-1] + b_i)
#   f = sigmoid(W_xf . x[t] + W_hf . h[t-1] + b_f)
#   o = sigmoid(W_xo . x[t] + W_ho . h[t-1] + b_o)
#   j = tanh   (W_xj . x[t] + W_hj . h[t-1] + b_j)
#   c[t] = i * j + f * c[t-1]
#   h[t] = o * tanh(c[t])

from primitiv import Model
from primitiv import Node
from primitiv import Parameter
from primitiv import operators as F
from primitiv import initializers as I


class LSTM(Model):
    """LSTM cell."""

    def __init__(self):
        self.pwxh = Parameter()
        self.pwhh = Parameter()
        self.pbh = Parameter()
        self.add_all_parameters()

    def init(self, in_size, out_size):
        """Creates a new LSTM."""
        self.pwxh.init([4 * out_size, in_size], I.XavierUniform())
        self.pwhh.init([4 * out_size, out_size], I.XavierUniform())
        self.pbh.init([4 * out_size], I.Constant(0))

    def restart(self, init_c = Node(), init_h = Node()):
        """Initializes internal states."""
        out_size = self.pwhh.shape()[1]
        self.wxh = F.parameter(self.pwxh)
        self.whh = F.parameter(self.pwhh)
        self.bh = F.parameter(self.pbh)
        self.c = init_c if init_c.valid() else F.zeros([out_size])
        self.h = init_h if init_h.valid() else F.zeros([out_size])

    def forward(self, x):
        """One step forwarding."""
        out_size = self.pwhh.shape()[1]
        u = self.wxh @ x + self.whh @ self.h + self.bh
        i = F.sigmoid(F.slice(u, 0, 0, out_size))
        f = F.sigmoid(F.slice(u, 0, out_size, 2 * out_size))
        o = F.sigmoid(F.slice(u, 0, 2 * out_size, 3 * out_size))
        j = F.tanh(F.slice(u, 0, 3 * out_size, 4 * out_size))
        self.c = i * j + f * self.c
        self.h = o * F.tanh(self.c)
        return self.h

    def get_c(self):
        """Retrieves current internal cell state."""
        return self.c

    def get_h(self):
        """Retrieves current hidden value."""
        return self.h
