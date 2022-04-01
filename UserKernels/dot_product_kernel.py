#!/usr/bin/env python3

import warnings

import torch

from gpytorch.constraints import Positive, Interval
from gpytorch.lazy import MatmulLazyTensor, RootLazyTensor
from gpytorch.kernels import Kernel


class DotProductKernel(Kernel):
    r"""
    Computes a covariance matrix based on the DotProductKernel
    between inputs :math:`\mathbf{x_1}` and :math:`\mathbf{x_2}`:

    .. math::
        \begin{equation*}
            k_\text{Linear}(\mathbf{x_1}, \mathbf{x_2}) = (\mathbf{x_1} - c)^\top 
            (\mathbf{x_2}-c).
        \end{equation*}

    where

    * :math:`c` is a :attr:`offset` parameter.


    .. note::

        To implement this efficiently, we use a :obj:`gpytorch.lazy.RootLazyTensor` during training and a
        :class:`gpytorch.lazy.MatmulLazyTensor` during test. These lazy tensors represent matrices of the form
        :math:`K = XX^{\top}` and :math:`K = XZ^{\top}`. This makes inference
        efficient because a matrix-vector product :math:`Kv` can be computed as
        :math:`Kv=X(X^{\top}v)`, where the base multiply :math:`Xv` takes only
        :math:`O(nd)` time and space.

    Args:
        :attr:`variance_prior` (:class:`gpytorch.priors.Prior`):
            Prior over the variance parameter (default `None`).
        :attr:`variance_constraint` (Constraint, optional):
            Constraint to place on variance parameter. Default: `Positive`.
        :attr:`active_dims` (list):
            List of data dimensions to operate on.
            `len(active_dims)` should equal `num_dimensions`.
    """

    def __init__(self,  offset_prior=None,  offset_constraint=None, **kwargs):
        super(DotProductKernel, self).__init__(**kwargs)
        if offset_constraint is None:
            offset_constraint = Interval(-4e5, 4e5)


        if offset_prior is not None:
            self.register_prior("offset_prior", offset_prior, lambda: self.offset, lambda v: self._set_offset(v))

        self.register_parameter(name="raw_offset", parameter=torch.nn.Parameter(torch.zeros(*self.batch_shape, 1, 1)))
        
        self.register_constraint("raw_offset", offset_constraint)

    @property
    def offset(self):
        return self.raw_offset_constraint.transform(self.raw_offset)

    @offset.setter
    def offset(self, value: torch.Tensor):
        self._set_offset(value)

    def _set_offset(self, value: torch.Tensor):
        if not torch.is_tensor(value):
            value = torch.as_tensor(value).to(self.raw_offset)
        self.initialize(raw_offset=self.raw_offset_constraint.inverse_transform(value))

    def forward(self, x1, x2, diag=False, last_dim_is_batch=False, **params):
        x1_ = x1 - self.offset
        if last_dim_is_batch:
            x1_ = x1_.transpose(-1, -2).unsqueeze(-1)

        if x1.size() == x2.size() and torch.equal(x1, x2):
            # Use RootLazyTensor when x1 == x2 for efficiency when composing
            # with other kernels
            prod = RootLazyTensor(x1_)

        else:
            x2_ = x2 - self.offset
            if last_dim_is_batch:
                x2_ = x2_.transpose(-1, -2).unsqueeze(-1)

            prod = MatmulLazyTensor(x1_, x2_.transpose(-2, -1))

        if diag:
            return prod.diag()
        else:
            return prod
