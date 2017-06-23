"""Basic or helper implementation."""
import torch
from torch import nn
from torch.autograd import Variable
from torch.nn import functional


def apply_nd(fn, input):
    """
    Apply fn whose output only depends on the last dimension values
    to an arbitrary n-dimensional input.
    It flattens dimensions except the last one, applies fn, and then
    restores the original size.
    """

    x_size = input.size()
    x_flat = input.view(-1, x_size[-1])
    output_flat = fn(x_flat)
    output_size = x_size[:-1] + (output_flat.size(-1),)
    return output_flat.view(*output_size)


def affine_nd(input, weight, bias):
    """
    An helper function to make applying the "wx + b" operation for
    n-dimensional x easier.

    Args:
        input (Variable): An arbitrary input data, whose size is
            (d0, d1, ..., dn, input_dim)
        weight (Variable): A matrix of size (output_dim, input_dim)
        bias (Variable): A bias vector of size (output_dim,)

    Returns:
        output: The result of size (d0, ..., dn, output_dim)
    """

    input_size = input.size()
    input_flat = input.view(-1, input_size[-1])
    bias_expand = bias.unsqueeze(0).expand(input_flat.size(0), bias.size(0))
    output_flat = torch.addmm(bias_expand, input_flat, weight)
    output_size = input_size[:-1] + (weight.size(1),)
    output = output_flat.view(*output_size)
    return output


def dot_nd(query, candidates):
    """
    Perform a dot product between a query and n-dimensional candidates.

    Args:
        query (Variable): A vector to query, whose size is
            (query_dim,)
        candidates (Variable): A n-dimensional tensor to be multiplied
            by query, whose size is (d0, d1, ..., dn, query_dim)

    Returns:
        output: The result of the dot product, whose size is
            (d0, d1, ..., dn)
    """

    cands_size = candidates.size()
    cands_flat = candidates.view(-1, cands_size[-1])
    output_flat = torch.mv(cands_flat, query)
    output = output_flat.view(*cands_size[:-1])
    return output


def convert_to_one_hot(indices, num_classes):
    """
    Args:
        indices (Variable): A vector containing indices,
            whose size is (batch_size,).
        num_classes (Variable): The number of classes, which would be
            the second dimension of the resulting one-hot matrix.

    Returns:
        result: The one-hot matrix of size (batch_size, num_classes).
    """

    batch_size = indices.size(0)
    indices = indices.unsqueeze(1)
    one_hot = Variable(indices.data.new(batch_size, num_classes).zero_()
                       .scatter_(1, indices.data, 1))
    return one_hot


def st_gumbel_softmax(logits, temperature=1.0):
    """
    Return the result of Straight-Through Gumbel-Softmax Estimation.
    It approximates the discrete sampling via Gumbel-Softmax trick
    and applies the biased ST estimator.
    In the forward propagation, it emits the discrete one-hot result,
    and in the backward propagation it approximates the categorical
    distribution via smooth Gumbel-Softmax distribution.

    Args:
        logits (Variable): A un-normalized probability values,
            which has the size (batch_size, num_classes)
        temperature (float): A temperature parameter. The higher
            the value is, the smoother the distribution is.

    Returns:
        y: The sampled output, which has the property explained above.

    """
    eps = 1e-20
    u = logits.data.new(*logits.size()).uniform_()
    gumbel_noise = Variable(-torch.log(-torch.log(u + eps) + eps))
    y = logits + gumbel_noise
    y = functional.softmax(y / temperature)
    y_argmax = y.max(1)[1].squeeze(1)
    y_hard = convert_to_one_hot(indices=y_argmax, num_classes=y.size(1)).float()
    y = (y_hard - y).detach() + y
    return y
