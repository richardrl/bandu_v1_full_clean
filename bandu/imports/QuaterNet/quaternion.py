# Copyright (c) 2018-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import torch
import numpy as np

# PyTorch-backed implementations

def qmul(q, r):
    """
    Multiply quaternion(s) q with quaternion(s) r.
    Expects two equally-sized tensors of shape (*, 4), where * denotes any number of dimensions.
    Returns q*r as a tensor of shape (*, 4).
    """
    assert q.shape[-1] == 4
    assert r.shape[-1] == 4
    
    original_shape = q.shape
    
    # Compute outer product
    terms = torch.bmm(r.view(-1, 4, 1), q.view(-1, 1, 4))

    w = terms[:, 0, 0] - terms[:, 1, 1] - terms[:, 2, 2] - terms[:, 3, 3]
    x = terms[:, 0, 1] + terms[:, 1, 0] - terms[:, 2, 3] + terms[:, 3, 2]
    y = terms[:, 0, 2] + terms[:, 1, 3] + terms[:, 2, 0] - terms[:, 3, 1]
    z = terms[:, 0, 3] - terms[:, 1, 2] + terms[:, 2, 1] + terms[:, 3, 0]
    return torch.stack((w, x, y, z), dim=1).view(original_shape)


def qrot(q, v):
    """
    Rotate vector(s) v about the rotation described by quaternion(s) q.
    Expects a tensor of shape (*, 4) for q and a tensor of shape (*, 3) for v,
    where * denotes any number of dimensions.
    Returns a tensor of shape (*, 3).

    Assumes quaternions are in [x,y,z,w] format. (PyBullet format).
    Assumes besides the last dimension,q and v are the same size. If one quaternion
     is being applied to N vectors, reshaping will be required.

     q: must be shape [1,4] or [batch_size, 4]

    q should be a tensor or np ndarray
    """
    if isinstance(v, np.ndarray):
        v = torch.Tensor(v)
    if isinstance(q, np.ndarray):
        q = torch.Tensor(q)

    v = v.to(q.device)
    assert isinstance(q, torch.Tensor)
    assert q.shape[-1] == 4

    if len(q.shape) == 1:
        # q: [4] -> [num_points, 4]
        q = q.expand(v.shape[0], -1)
    elif len(q.shape) == 2:
        # v must be [batch_size, num_points, 3]
        assert v.shape[-1] == 3 and len(v.shape) == 3, v.shape

        # Broadcast the q to the shape of v
        q = q.unsqueeze(1).expand(-1, v.shape[1], -1)
    elif len(q.shape) == 3:
        # q: [nB, nO, 4]
        # v: [nB, nO, num_points, 3]
        nB, nO, _ = q.size()
        assert v.shape[1] == nO, (q.shape, v.shape)
        assert v.shape[3] == 3, (q.shape, v.shape)

        # q: [nB, nO, 4] -> [nB, nO, num_points_in_v, 3]
        q = q.unsqueeze(2).expand(-1, -1, v.shape[-2], -1)
    else:
        print("Shape not found")
        print(q.shape)
        raise NotImplementedError
    assert v.shape[-1] == 3

    assert q.shape[:-1] == v.shape[:-1], (q.shape[:-1], v.shape[:-1])

    original_shape = list(v.shape)

    # TODO: this contiguous call may be slowing things down
    q = q.contiguous().view(-1, 4)
    v = v.contiguous().view(-1, 3)

    # [batch_size, 3]
    quat_real = q[:, :-1].float()

    # [batch_size] -> [batch_size, 1]
    quat_imaginary = q[:, -1].unsqueeze(1)

    try:
        uv = torch.cross(quat_real, v.to(quat_real.device), dim=1)
    except Exception as e:
        print("shapes before cross products")
        print(quat_real.shape)
        print(v.shape)
        print(e)
        print(quat_real.dtype)
        print(v.dtype)
        raise Exception(e)
    uuv = torch.cross(quat_real, uv, dim=1)

    return (v + 2 * (quat_imaginary * uv + uuv)).view(original_shape)

def qeuler(q, order, epsilon=0):
    """
    Convert quaternion(s) q to Euler angles.
    Expects a tensor of shape (*, 4), where * denotes any number of dimensions.
    Returns a tensor of shape (*, 3).
    """
    assert q.shape[-1] == 4
    
    original_shape = list(q.shape)
    original_shape[-1] = 3
    q = q.view(-1, 4)
    
    q0 = q[:, 0]
    q1 = q[:, 1]
    q2 = q[:, 2]
    q3 = q[:, 3]
    
    if order == 'xyz':
        x = torch.atan2(2 * (q0 * q1 - q2 * q3), 1 - 2*(q1 * q1 + q2 * q2))
        y = torch.asin(torch.clamp(2 * (q1 * q3 + q0 * q2), -1+epsilon, 1-epsilon))
        z = torch.atan2(2 * (q0 * q3 - q1 * q2), 1 - 2*(q2 * q2 + q3 * q3))
    elif order == 'yzx':
        x = torch.atan2(2 * (q0 * q1 - q2 * q3), 1 - 2*(q1 * q1 + q3 * q3))
        y = torch.atan2(2 * (q0 * q2 - q1 * q3), 1 - 2*(q2 * q2 + q3 * q3))
        z = torch.asin(torch.clamp(2 * (q1 * q2 + q0 * q3), -1+epsilon, 1-epsilon))
    elif order == 'zxy':
        x = torch.asin(torch.clamp(2 * (q0 * q1 + q2 * q3), -1+epsilon, 1-epsilon))
        y = torch.atan2(2 * (q0 * q2 - q1 * q3), 1 - 2*(q1 * q1 + q2 * q2))
        z = torch.atan2(2 * (q0 * q3 - q1 * q2), 1 - 2*(q1 * q1 + q3 * q3))
    elif order == 'xzy':
        x = torch.atan2(2 * (q0 * q1 + q2 * q3), 1 - 2*(q1 * q1 + q3 * q3))
        y = torch.atan2(2 * (q0 * q2 + q1 * q3), 1 - 2*(q2 * q2 + q3 * q3))
        z = torch.asin(torch.clamp(2 * (q0 * q3 - q1 * q2), -1+epsilon, 1-epsilon))
    elif order == 'yxz':
        x = torch.asin(torch.clamp(2 * (q0 * q1 - q2 * q3), -1+epsilon, 1-epsilon))
        y = torch.atan2(2 * (q1 * q3 + q0 * q2), 1 - 2*(q1 * q1 + q2 * q2))
        z = torch.atan2(2 * (q1 * q2 + q0 * q3), 1 - 2*(q1 * q1 + q3 * q3))
    elif order == 'zyx':
        x = torch.atan2(2 * (q0 * q1 + q2 * q3), 1 - 2*(q1 * q1 + q2 * q2))
        y = torch.asin(torch.clamp(2 * (q0 * q2 - q1 * q3), -1+epsilon, 1-epsilon))
        z = torch.atan2(2 * (q0 * q3 + q1 * q2), 1 - 2*(q2 * q2 + q3 * q3))
    else:
        raise

    return torch.stack((x, y, z), dim=1).view(original_shape)

# Numpy-backed implementations

def qmul_np(q, r):
    q = torch.from_numpy(q).contiguous()
    r = torch.from_numpy(r).contiguous()
    return qmul(q, r).numpy()

def qrot_np(q, v):
    q = torch.from_numpy(q).contiguous().float()
    v = torch.from_numpy(v).contiguous().float()
    return qrot(q, v).numpy()

def qeuler_np(q, order, epsilon=0, use_gpu=False):
    if use_gpu:
        q = torch.from_numpy(q).cuda()
        return qeuler(q, order, epsilon).cpu().numpy()
    else:
        q = torch.from_numpy(q).contiguous()
        return qeuler(q, order, epsilon).numpy()

def qfix(q):
    """
    Enforce quaternion continuity across the time dimension by selecting
    the representation (q or -q) with minimal distance (or, equivalently, maximal dot product)
    between two consecutive frames.
    
    Expects a tensor of shape (L, J, 4), where L is the sequence length and J is the number of joints.
    Returns a tensor of the same shape.
    """
    assert len(q.shape) == 3
    assert q.shape[-1] == 4
    
    result = q.copy()
    dot_products = np.sum(q[1:]*q[:-1], axis=2)
    mask = dot_products < 0
    mask = (np.cumsum(mask, axis=0)%2).astype(bool)
    result[1:][mask] *= -1
    return result

def expmap_to_quaternion(e):
    """
    Convert canonical_axis-angle predicted_rotations_quat (aka exponential maps) to quaternions.
    Stable formula from "Practical Parameterization of Rotations Using the Exponential Map".
    Expects a tensor of shape (*, 3), where * denotes any number of dimensions.
    Returns a tensor of shape (*, 4).
    """
    assert e.shape[-1] == 3
    
    original_shape = list(e.shape)
    original_shape[-1] = 4
    e = e.reshape(-1, 3)

    theta = np.linalg.norm(e, axis=1).reshape(-1, 1)
    w = np.cos(0.5*theta).reshape(-1, 1)
    xyz = 0.5*np.sinc(0.5*theta/np.pi)*e
    return np.concatenate((w, xyz), axis=1).reshape(original_shape)

def euler_to_quaternion(e, order):
    """
    Convert Euler angles to quaternions.
    """
    assert e.shape[-1] == 3
    
    original_shape = list(e.shape)
    original_shape[-1] = 4
    
    e = e.reshape(-1, 3)
    
    x = e[:, 0]
    y = e[:, 1]
    z = e[:, 2]
    
    rx = np.stack((np.cos(x/2), np.sin(x/2), np.zeros_like(x), np.zeros_like(x)), axis=1)
    ry = np.stack((np.cos(y/2), np.zeros_like(y), np.sin(y/2), np.zeros_like(y)), axis=1)
    rz = np.stack((np.cos(z/2), np.zeros_like(z), np.zeros_like(z), np.sin(z/2)), axis=1)

    result = None
    for coord in order:
        if coord == 'x':
            r = rx
        elif coord == 'y':
            r = ry
        elif coord == 'z':
            r = rz
        else:
            raise
        if result is None:
            result = r
        else:
            result = qmul_np(result, r)
            
    # Reverse antipodal representation to have a non-negative "w"
    if order in ['xyz', 'yzx', 'zxy']:
        result *= -1
    
    return result.reshape(original_shape)
    