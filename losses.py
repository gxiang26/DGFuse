
import torch
import torch.nn.functional as F


def _make_sobel(device, dtype):
    kx = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]],
                      device=device, dtype=dtype).view(1, 1, 3, 3) / 8.0   # ← /8
    ky = torch.tensor([[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]],
                      device=device, dtype=dtype).view(1, 1, 3, 3) / 8.0   # ← /8
    return kx, ky


def _luma(x: torch.Tensor) -> torch.Tensor:
    if x.shape[1] == 1:
        return x
    if x.shape[1] == 3:
        w = torch.tensor([0.299, 0.587, 0.114], device=x.device, dtype=x.dtype)
        return (x * w.view(1, 3, 1, 1)).sum(dim=1, keepdim=True)
    return x.mean(dim=1, keepdim=True)


def sobel_magnitude(x: torch.Tensor) -> torch.Tensor:
    l = _luma(x)
    kx, ky = _make_sobel(x.device, x.dtype)
    gx = F.conv2d(l, kx, padding=1)
    gy = F.conv2d(l, ky, padding=1)
    return torch.sqrt(gx * gx + gy * gy + 1e-8)


def flow_matching_loss(v_pred, v_star):
    return F.mse_loss(v_pred, v_star)


def gradient_loss(if_pred, ir, vi):
    g_if = sobel_magnitude(if_pred)
    g_ir = sobel_magnitude(ir)
    g_vi = sobel_magnitude(vi)
    g_max = torch.maximum(g_ir, g_vi)
    return (g_if - g_max).abs().mean()


def total_loss(v_pred, v_star, x_hat, ir, vi, lambda_fm=1.3, lambda_grad=0.5):
    l_fm = flow_matching_loss(v_pred, v_star)
    l_grad = gradient_loss(x_hat, ir, vi)
    return lambda_fm * l_fm + lambda_grad * l_grad, {
        "l_fm": l_fm.detach(),
        "l_grad": l_grad.detach(),
    }