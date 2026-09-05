import torch
from diffpfa.IFA.channel.pfa_channel import process_cztnufft
from diffpfa.IFA.PFA import _apply_ifft_and_deconv

device = "cuda" if torch.cuda.is_available() else "cpu"

N_pulses, N_samples = 32, 64
N_u, N_r = 32, 32
L_u, L_r = 100.0, 100.0
fxc = 9.6e9
k_ctr_u, k_ctr_r = 0.0, 64.0
cot_theta = torch.linspace(-0.002, 0.002, N_pulses, device=device, dtype=torch.float64)
Kr_radial = torch.linspace(63.0, 65.0, N_samples, device=device, dtype=torch.float64).unsqueeze(0).expand(N_pulses, N_samples)
Ku_radial = cot_theta.unsqueeze(1) * Kr_radial

def forward_pfa(sig):
    grid_2d = process_cztnufft(
        signal=sig,
        fxc=fxc,
        pvp={},
        Ku=Ku_radial,
        Kr=Kr_radial,
        N_u=N_u,
        N_r=N_r,
        L_u=L_u,
        L_r=L_r,
        k_ctr_u=k_ctr_u,
        k_ctr_r=k_ctr_r,
        batch_size=N_r,
        device=device
    )
    img = _apply_ifft_and_deconv(grid_2d, N_u, N_r, device).T
    return img

torch.manual_seed(42)
sig = torch.randn(N_pulses, N_samples, dtype=torch.complex64, device=device, requires_grad=True)
img_init = forward_pfa(sig)
img_target = img_init.detach().clone()
img_target[16, 16] += 5.0 + 2.0j

# Optimize sig to produce img_target using Adam
optimizer = torch.optim.Adam([sig], lr=1e-3)
losses = []
for step in range(15):
    optimizer.zero_grad()
    img_current = forward_pfa(sig)
    loss = torch.sum(torch.abs(img_current - img_target) ** 2)
    loss.backward()
    optimizer.step()
    losses.append(loss.item())

print(f"Step 0 loss: {losses[0]:.4f}")
print(f"Step 5 loss: {losses[5]:.4f}")
print(f"Step 10 loss: {losses[10]:.4f}")
print(f"Step 14 loss: {losses[14]:.4f}")
assert losses[-1] < losses[0]
print(f"Loss reduced from {losses[0]:.4f} to {losses[-1]:.4f} (Reduction factor: {losses[0]/losses[-1]:.2f}x)")
