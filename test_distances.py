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
from distance_functions import FrobeniusDistance, L2Distance, KLDivergenceDistance


def run_smoke():
    torch.manual_seed(0)
    B, C, H, W = 3, 16, 14, 13
    feat = torch.randn(B, C, H, W)
    feat2 = torch.randn(B, C, H, W)

    # Instantiate transforms
    lkd = LKDTransform(in_channels=C, k=4, mode='cosine', out_dim=32)
    sp = SPTransform(in_channels=C, out_dim=11)
    ickd = ICKDTransform(in_channels=C, out_dim=20)
    cwd = CWDTransform(in_channels=C, out_dim=10)
    att = AttentionTransform(p=2.0, out_dim=9)

    # Compute student/teacher outputs
    lkd_s, lkd_t = lkd(feat), lkd(feat)
    sp_s, sp_t = sp(feat), sp(feat)
    ickd_s, ickd_t = ickd(feat), ickd(feat)
    cwd_s, cwd_t = cwd(feat), cwd(feat)
    att_s, att_t = att(feat), att(feat)

    # Distance modules
    frob = FrobeniusDistance()
    l2 = L2Distance()
    kld = KLDivergenceDistance(eps=1e-6)

    # Frobenius over matrices
    assert frob(lkd_s, lkd_t).item() < 1e-6
    assert frob(sp_s, sp_t).item() < 1e-6
    assert frob(ickd_s, ickd_t).item() < 1e-6

    # Frobenius fallback to vectors (Attention has no matrix metadata)
    assert frob(att_s, att_t).item() < 1e-6

    # L2 distance over vectors
    assert l2(att_s, att_t).item() < 1e-6

    # KL divergence on probs (CWD)
    assert kld(cwd_s, cwd_t).item() < 1e-6
    # And positive when distributions differ
    assert kld(cwd(feat), cwd(feat2)).item() >= 0.0

    print('All distance function smoke tests passed.')


if __name__ == '__main__':
    run_smoke()
