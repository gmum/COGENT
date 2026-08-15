import torch

def torch_interp(x, x_points, y_points):
    """
    diffrentiable version of 'np.interp' function in torch
    Notes:
        - x_points and y_points should be 1D vectors of corresponding points.
        - x_points should be sorted.
        - x can be of any shape

    From: https://discuss.pytorch.org/t/linear-interpolation-in-pytorch/66861/11
    """
    right_idx = torch.searchsorted(x_points, x)
    left_idx = (right_idx - 1).clamp(0)
    right_idx.clamp_(None, x_points.shape[0] - 1)

    left_point_x = x_points[left_idx]
    _lambda = (x - left_point_x) / (x_points[right_idx] - left_point_x)
    _lambda.clamp_(-1, 1)
    interpolated_y = y_points[left_idx] * (1 - _lambda) + y_points[right_idx] * _lambda
    return interpolated_y