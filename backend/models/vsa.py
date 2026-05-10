#用于VSA编码解码

import torch
import random
import numpy as np



def generate_random_matrix_A(m, d, scale=10):
    H = torch.randn(d, m) 
    Q, _ = torch.linalg.qr(H)
    return Q.T * scale

def Encode_VSA(label,A,isELL = True):
    """
    Docstring for generate_VSA 将椭圆的信息编码为VSA向量
    :param label: 参数 [x,y,a,b,angle] x,y,a,b归一化至[0,1],并且a>=b,angle被归一化至[-0.5pi,0.5pi)
    :param A: Description
    :isELL: 是否是对整个椭圆信息进行编码
    """
    x = label[:,0]
    y = label[:,1]
    a = label[:,2].unsqueeze(1)
    b = label[:,3].unsqueeze(1)
    angle = label[:,4]
    if isELL == True:
        sin_a = torch.sin(angle)
        cos_a = torch.cos(angle)
        u_vector = torch.stack([cos_a, sin_a], dim=1)
        v_vector = torch.stack([-sin_a, cos_a], dim=1)
        R = torch.sqrt(a**2*(u_vector @ A)**2+b**2*(v_vector @ A)**2)
        Magnitude = torch.special.bessel_j0(10 * R)                 #10
        Xc = torch.stack([x,y],dim = 1)
        X = Xc @ A
        if X.max()>torch.pi or X.min()<-torch.pi:
            print("出现相位卷绕")
        Phase = torch.exp(1j * (Xc @ A))
        VSA = Phase * Magnitude
    else:
        Xc = torch.stack([x,y],dim = 1)
        X = Xc @ A
        if X.max()>torch.pi or X.min()<-torch.pi:
            print("出现相位卷绕")
        Phase = torch.exp(1j * (Xc @ A))
        VSA = Phase
    return VSA

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
        B, D = R.shape
        y = (R ** 2).unsqueeze(-1) 
        A_rows = A
        x_coords = A_rows[0, :] 
        y_coords = A_rows[1, :] 
        
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

        vec_v = eigenvectors[:, :, 0].clone() 
        vec_u = eigenvectors[:, :, 1].clone()

        mask_u = vec_u[:, 0] < 0
        vec_u = torch.where(mask_u.unsqueeze(1), -vec_u, vec_u)

        mask_v = vec_v[:, 1] > 0
        vec_v = torch.where(mask_v.unsqueeze(1), -vec_v, vec_v)
        
        recovered_angle = torch.atan2(vec_u[:, 1], vec_u[:, 0])  
        is_circle = torch.abs(res_a - res_b) < 1e-4
        if is_circle.any():
            recovered_angle = torch.where(is_circle, torch.zeros_like(recovered_angle), recovered_angle)
        half_pi = torch.pi / 2
        eps_angle = 1e-5
        mask_boundary = recovered_angle > (half_pi - eps_angle)
        recovered_angle = torch.where(mask_boundary, recovered_angle - torch.pi, recovered_angle)
        
        return res_a, res_b, vec_u, vec_v, recovered_angle  
    A_pinv = 0.01 * A.T
    magnitude = torch.abs(VSA)
    phase = torch.angle(VSA)
    rec_x, rec_y = solve_pos_from_phase(phase, A_pinv)
    if isELL == True:
        rec_R = invert_j0_approx(magnitude)
        rec_a, rec_b, _, _, rec_angle = solve_abuv_from_R(rec_R, A)
    else:
        rec_a = torch.zeros_like(rec_x)
        rec_b = torch.zeros_like(rec_x)
        rec_angle = torch.zeros_like(rec_x)
    label = torch.stack([rec_x,rec_y,rec_a,rec_b,rec_angle],dim = -1)
    return label


if __name__=="__main__":
    A = generate_random_matrix_A(2,512)
    label  = torch.tensor([0.3,0.5,0.1,0.09,1])
    label = label.unsqueeze(0)
    VSA = Encode_VSA(label,A)
    test = Decode_VSA(VSA,A)
    print(test)