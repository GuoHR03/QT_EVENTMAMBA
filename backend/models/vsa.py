import torch

def Decode_VSA(VSA,A,isELL = True):
    """
    Docstring for Decode_VSA 解码VSA向量,将其转为[x,y,a,b,angle]
    
    :param VSA: Description
    :param A: Description
    :isELL: 是否是对整个椭圆信息进行解码
    """
    def invert_j0_approx(y):
        """
        三阶近似逆函数 (Order-3 Approximation)
        在 x < 2.0 时极其精准。
        """
        y = torch.clamp(y, min=0.0, max=1.0)
        z = 1.0 - y
        sqrt_z = torch.sqrt(z)
        poly = 1.0 + 0.125 * z + 13 / 384 * (z ** 2)
        x = 2.0 * sqrt_z * poly
        return x / 10.0        #10
    
    def solve_pos_from_phase(phase, A_pinv):
        pos = phase @ A_pinv
        return pos[:,0], pos[:,1]

    def solve_abuv_from_R(R, A):
        if R.dim() == 1:
            R = R.unsqueeze(0)  
        B = R.shape[0]
        y = (R ** 2).unsqueeze(-1) 
        x_coords = A[0, :]
        y_coords = A[1, :]
        
        Phi = torch.stack([x_coords**2, 2*x_coords*y_coords, y_coords**2], dim=1)
        
        if Phi.device != R.device:
            Phi = Phi.to(R.device)
            
        Phi_batch = Phi.unsqueeze(0).expand(B, -1, -1)
        
        coeffs = torch.linalg.lstsq(Phi_batch, y).solution
        
        coeffs = coeffs.squeeze(-1)
        
        m11 = coeffs[:, 0]
        m12 = coeffs[:, 1]
        m22 = coeffs[:, 2]
        
        M = torch.stack([
            torch.stack([m11, m12], dim=1),
            torch.stack([m12, m22], dim=1)
        ], dim=1)
        
        eigenvalues, eigenvectors = torch.linalg.eigh(M)
        
        eps = 1e-8
        res_b = torch.sqrt(torch.clamp(torch.abs(eigenvalues[:, 0]), min=eps)) # (B,)
        res_a = torch.sqrt(torch.clamp(torch.abs(eigenvalues[:, 1]), min=eps)) # (B,)        

        vec_u = eigenvectors[:, :, 1].clone()

        mask_u = vec_u[:, 0] < 0
        vec_u = torch.where(mask_u.unsqueeze(1), -vec_u, vec_u)

        recovered_angle = torch.atan2(vec_u[:, 1], vec_u[:, 0])  
        is_circle = torch.abs(res_a - res_b) < 1e-4
        if is_circle.any():
            recovered_angle = torch.where(is_circle, torch.zeros_like(recovered_angle), recovered_angle)
        half_pi = torch.pi / 2
        eps_angle = 1e-5
        mask_boundary = recovered_angle > (half_pi - eps_angle)
        recovered_angle = torch.where(mask_boundary, recovered_angle - torch.pi, recovered_angle)
        
        return res_a, res_b, recovered_angle
    A_pinv = 0.01 * A.T
    magnitude = torch.abs(VSA)
    phase = torch.angle(VSA)
    rec_x, rec_y = solve_pos_from_phase(phase, A_pinv)
    if isELL == True:
        rec_R = invert_j0_approx(magnitude)
        rec_a, rec_b, rec_angle = solve_abuv_from_R(rec_R, A)
    else:
        rec_a = torch.zeros_like(rec_x)
        rec_b = torch.zeros_like(rec_x)
        rec_angle = torch.zeros_like(rec_x)
    label = torch.stack([rec_x,rec_y,rec_a,rec_b,rec_angle],dim = -1)
    return label
