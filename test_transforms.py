import torch
from transforms import (
    LKDTransform,
    LRTransform,
    ICKDTransform,
    CWDTransform,
    SPTransform,
    MaskedTransform,
    AttentionTransform,
)


def run_smoke():
    torch.manual_seed(0)
    B, C, H, W = 3, 16, 15, 17
    feat = torch.randn(B, C, H, W)

    # LKD
    lkd = LKDTransform(in_channels=C, k=4, mode='cosine', out_dim=32)
    v, aux = lkd(feat)
    assert v.shape == (B, 32)
    assert aux['intra'].shape == (B, 16, 16)

    # LR
    t1 = torch.randn(B, 8, H, W)
    t2 = torch.randn(B, 4, H, W)
    s = torch.randn(B, 12, H, W)
    lr = LRTransform(in_channels_list=[8, 4, 12], out_dim=64)
    v, _ = lr([t1, t2], s)
    assert v.shape == (B, 64)

    # ICKD
    ickd = ICKDTransform(in_channels=C, out_dim=20)
    v, aux = ickd(feat)
    assert v.shape == (B, 20)
    assert aux['ICC'].shape == (B, C, C)

    # CWD
    cwd = CWDTransform(in_channels=C, out_dim=10)
    v, aux = cwd(feat)
    assert v.shape == (B, 10)
    assert aux['probs'].shape == (B, C, H, W)
    assert torch.all(aux['probs'] >= 0)

    # SP
    sp = SPTransform(in_channels=C, out_dim=11)
    v, aux = sp(feat)
    assert v.shape == (B, 11)
    assert aux['G'].shape == (B, C, C)

    # Masked
    mt = MaskedTransform(in_channels=C, out_dim=7, mask_ratio=0.3)
    v, aux = mt(feat)
    assert v.shape == (B, 7)
    assert aux['mask'].shape == (B, 1, H, W)

    # Attention
    att = AttentionTransform(p=2.0, out_dim=9)
    v, _ = att(feat)
    assert v.shape == (B, 9)

    print('All transform smoke tests passed.')


if __name__ == '__main__':
    run_smoke()
