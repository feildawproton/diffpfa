import torch

# Demonstrate how to pass 2D k_start and k_step to czt
def demo():
    batch = 10
    N_pulses = 100
    L_u = 10.0
    M_u = 120
    
    # Kr for each batch element
    Kr_cart = torch.linspace(10, 20, batch).unsqueeze(1) # shape (batch, 1)
    
    # cot_theta for each pulse (slow-time variation)
    cot_theta = torch.linspace(0.8, 1.2, N_pulses)
    
    # ku_start and ku_step for each column (batch)
    ku_start = Kr_cart * cot_theta[0]
    ku_step = Kr_cart * ((cot_theta[-1] - cot_theta[0]) / max(N_pulses - 1, 1))
    
    print("ku_start shape:", ku_start.shape)
    print("ku_step shape:", ku_step.shape)
    # They are (batch, 1), which broadcasts correctly in CZT over dim=-1 for an input of (batch, N_pulses)

if __name__ == "__main__":
    demo()
