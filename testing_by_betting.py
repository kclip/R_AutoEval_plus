import numpy as np
from scipy.stats import beta
c = 3/4
eps = 0.00000001

def ppi_money(nu, rho, tilde_L, tilde_L_Delta, hat_L, alpha):
    R_rho = rho*tilde_L + hat_L - rho*tilde_L_Delta
    return 1 - nu*(R_rho - alpha), R_rho

def val_p_value_naive(losses, alpha):
    # trust the mean -- define extreme p-value for consistency in the code
    if np.mean(losses) < alpha:
        return -0.1
    else:
        return 1.1 
def WSR_PPI_p_value(loss_vec,loss_vec_n_simul,loss_vec_N_simul,alpha,delta,rhos, betting_mode='ONS',num_grid_UP=10000):
    def _WSR(t, n, alpha, delta, per_rho_running_sum, per_rho_running_sq_sum):
        per_rho_mu_t = per_rho_running_sum/t
        per_rho_var_t = per_rho_running_sq_sum/t
        per_rho_nu_t_tilde = np.sqrt( (2*np.log(1/delta))/( n*per_rho_var_t) )
        per_rho_nu_t_ucb = np.clip(np.expand_dims(per_rho_nu_t_tilde, axis=1), a_min=0.0, a_max=c/(1 -alpha+np.expand_dims(rhos,axis=1))) # broadcasting
        return np.squeeze(per_rho_nu_t_ucb, axis=1)

    def _UP(alpha, prev_obs, E_UP, rhos):
        def _e_bet(bet, alpha, rhos, obs):
            return 1 + (-np.expand_dims(obs, axis=1)+alpha)*bet/(1+np.expand_dims(rhos,axis=1)-alpha)
        if prev_obs is None:
            curr_UP = np.ones(E_UP.shape)
        else:
            curr_UP = _e_bet(bet_grid, alpha, rhos, prev_obs)
        E_UP *= curr_UP
        per_rho_nu_t_ucb = np.sum(E_UP*bet_grid*beta_pdf_grid, axis=1)/np.sum(E_UP*beta_pdf_grid, axis=1)
        per_rho_nu_t_ucb /= (1 -alpha+rhos) # for consistent expression with WSR
        return per_rho_nu_t_ucb, E_UP

    def _update_weight(init_weight, individual_e_process):
        unnormalized = init_weight * individual_e_process
        return unnormalized/np.sum(unnormalized)

    def _combine(curr_moneys, weights):
        return np.sum(curr_moneys*weights)

    tilde_L_t_list = []
    hat_L_t_list = []
    tilde_L_Delta_list = []
    n = loss_vec.shape[0]
    N = loss_vec_N_simul.shape[0]
    used_N = 0
    used_n = 0
    Z_i_sq_sum = 0
    init_weights = np.array([1/rhos.shape[0]]*rhos.shape[0])
    weights = np.array([1/rhos.shape[0]]*rhos.shape[0])
    e_process = 1
    if betting_mode == 'UP':
        bet_grid = np.expand_dims(np.linspace(0+eps,1-eps,num_grid_UP), axis=0)
        beta_pdf_grid = beta.pdf(bet_grid, 0.5, 0.5)
        beta_pdf_grid = beta_pdf_grid/np.sum(beta_pdf_grid) # deal with discretization
        E_UP = np.ones([rhos.shape[0], num_grid_UP])
        nus, E_UP = _UP(alpha, None, E_UP, rhos)
    elif betting_mode == 'WSR':
        per_rho_running_sum = 0.5*np.ones([rhos.shape[0]])
        per_rho_running_sq_sum = 0.25*np.ones([rhos.shape[0]])
        nus = _WSR(1, n, alpha, delta, per_rho_running_sum, per_rho_running_sq_sum)
    else:
        raise NotImplementedError
    L = 0
    individual_e_process = 1
    for t in range(n):
        n_t = 1
        N_t = int(N//n)
        tilde_L_t_list.append(  np.mean(loss_vec_N_simul[used_N:used_N+N_t]))
        hat_L_t_list.append( np.mean(loss_vec[ used_n:used_n+n_t ]) )
        tilde_L_Delta_list.append( np.mean(loss_vec_n_simul[ used_n:used_n+n_t ]) )
        used_N += N_t
        used_n += n_t
        tilde_L_t_np = np.array(tilde_L_t_list)
        hat_L_t_np = np.array(hat_L_t_list)
        tilde_L_Delta_np = np.array(tilde_L_Delta_list)        
        if N == 0:
            fake_loss_curr = 0
        else:
            fake_loss_curr = tilde_L_t_np[t]
        curr_moneys, R_rhos = ppi_money(nus, rhos, fake_loss_curr, tilde_L_Delta_np[t], hat_L_t_np[t], alpha)       
        individual_e_process *= curr_moneys
        # combining
        combined_moneys = _combine(curr_moneys, weights)
        e_process *= combined_moneys
        if rhos.shape[0] == 1:
            pass
        else:
            weights = _update_weight(init_weights, individual_e_process)
        if np.any(np.isnan(weights)):
            weights = init_weights
        if betting_mode == 'UP':
            nus, E_UP = _UP(alpha, R_rhos, E_UP, rhos)
        elif betting_mode == 'WSR':
            per_rho_running_sum += R_rhos
            per_rho_running_sq_sum += (R_rhos- (per_rho_running_sum/(t+2)))**2 # at t=0, we have one sample, and one init. guess, so we need to divide it by 2
            nus = _WSR(t+2, n, alpha, delta, per_rho_running_sum, per_rho_running_sq_sum)
        else:
            raise NotImplementedError
        if e_process >= 1/(delta-eps):
            return (1/e_process)
        else:
            pass
    return 1/e_process # we can of course choose 1/max e_process, but here no difference as we are focusing on Bonferroni correction (what matters is whether the value exceeds 1/delta or not, not its value itself)