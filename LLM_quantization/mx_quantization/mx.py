import numpy as np
import torch


def mx_quant(model, mx_dict):
    m = mx_dict['m']
    k_1 = mx_dict['k1']
    k_2 = mx_dict['k2']
    d_1 = mx_dict['d1']
    d_2 = mx_dict['d2']
    avg_qsnr = 0
    ind_f = 0
    sum_err = 0
    sum_signal = 0
    total_elements = 0
    total_bits = 0
    for f in model.parameters():
        f.data, f.err, f.signal, _, _, per_layer_elements, per_layer_bits = MX(f.data, m, k_1, k_2, d_1, d_2, None, None)
        total_elements += per_layer_elements
        total_bits += per_layer_bits
        sum_err += f.err
        sum_signal += f.signal
        ind_f += 1
    avg_bitwidth = total_bits/total_elements
    avg_qsnr = 10*torch.log10(sum_signal/sum_err)
    return avg_qsnr, avg_bitwidth, model




def MX(vec, m, k_1, k_2, d_1, d_2, s, s_g):
    total_bitwidth = 0
    # vec: weight matrix per layer
    # good to reshape into k_1 * k_2 
    # print(k_1, k_2)
    if k_1 != 'inf':
        assert k_1 % k_2 == 0
    def _reshape_with_padding(x, k_1):
        if len(x.shape) == 2:
            n, d = x.shape
        elif len(x.shape) == 1:
            n = x.shape[0]
            d = 1
        else:
            raise NotImplementedError
        total_elements = n * d
        pad_size = (k_1 - (total_elements % k_1))
        padded_x = torch.cat([x.flatten(), torch.zeros(pad_size, device=x.device, dtype=x.dtype)])
        d_pad = (total_elements + pad_size) // k_1
        return padded_x.view(k_1, d_pad), k_1, d_pad, total_elements
    
    def _remove_padding(x, original_shape, padded_vec):
        if len(original_shape) == 2:
            n, d = original_shape
        elif len(original_shape) == 1:
            n = original_shape[0]
            d = 1
        else:
            raise NotImplementedError
        return x.t().flatten().reshape(padded_vec[1], padded_vec[0]).t().flatten()[:n * d].view(n, d)

    if k_1 == 'inf':
        pass
    else:
        padded_vec, k_1, d_pad, total_elements = _reshape_with_padding(vec, k_1)
    total_bits = m * total_elements + d_1*(d_pad) + d_2*(k_1//k_2)*d_pad
    
    if s is None:
        if k_1 == 'inf':
            raise NotImplementedError
        else:
            s_kron = torch.kron(torch.max(torch.abs(padded_vec), dim=0)[0], torch.ones(k_1//k_2).to(padded_vec.device) )
            if m <= 3: # empirically observed that rounding is better with extremely coarse case (via small experiments)
                s = torch.clip(torch.round(torch.log2(s_kron/(2**(m-1)-1))), -2**(d_1-1), 2**(d_1-1)-1)
                padded_vec_k_2 = padded_vec.t().flatten().reshape(-1, k_2).t()  #padded_vec.view(k_2, -1)
                s_g_tmp = torch.max(torch.abs(padded_vec_k_2), dim=0)[0]
                s_g = torch.clip( torch.round(torch.log2(s_g_tmp/(2**(m-1)-1)))-s, -2**d_2+1,0)
            else:
                s = torch.clip(torch.ceil(torch.log2(s_kron/(2**(m-1)-1))), -2**(d_1-1), 2**(d_1-1)-1)
                padded_vec_k_2 = padded_vec.t().flatten().reshape(-1, k_2).t()  #padded_vec.view(k_2, -1)
                s_g_tmp = torch.max(torch.abs(padded_vec_k_2), dim=0)[0]
                s_g = torch.clip( torch.ceil(torch.log2(s_g_tmp/(2**(m-1)-1)))-s, -2**d_2+1,0)
    else:
        if k_1 == 'inf':
            pass
        else:
            padded_vec_k_2 = padded_vec.t().flatten().reshape(-1, k_2).t() 
    if k_1 == 'inf':
        w = vec
        result = torch.sign( w ) *torch.pow(2, s) *torch.clip(torch.round(torch.abs(w) / torch.pow(2, s) ), 0, 2**(m-1)-1)
        err = torch.norm(w-result)**2
        signal = torch.norm(w)**2
        s_g = None
    else:
        quantized_tmp = torch.sign(  padded_vec_k_2)*torch.pow(2, s+s_g) *torch.clip(torch.round(torch.abs(padded_vec_k_2) / torch.pow(2, s+s_g) ), 0, 2**(m-1)-1)
        result = torch.squeeze(_remove_padding(quantized_tmp, vec.shape, padded_vec.shape))
        err = torch.norm(vec-result)**2
        signal = torch.norm(vec)**2
        
    return result, err, signal, s, s_g, total_elements, total_bits
