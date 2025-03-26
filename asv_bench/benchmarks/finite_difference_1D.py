import numpy as np
from derivative import FiniteDifference, Kernel

class FiniteDifferenceBM:
    def setup(self):
        self.differentiator = FiniteDifference(k=1)
        self.t = np.arange(0, 2 * np.pi, .01)

    def time_derivative(self):
        self.differentiator.d(X=self.t, t=self.t, axis=0)

    def peakmem_derivative(self):
        self.differentiator.d(X=self.t, t=self.t, axis=0)

class KernelBM:
    def setup(self):
        self.differentiator = Kernel(lmbd=.1)
        self.t = np.arange(0, 2* np.pi, 0.01)
        self.x = np.sin(self.t)

    def time_derivative(self):
        self.differentiator.d(X=self.x, t=self.t, axis=0)

    def peakmem_derivative(self):
        self.differentiator.d(X=self.x, t=self.t, axis=0)
