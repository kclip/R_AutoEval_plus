
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import pickle
import tqdm
from scipy.stats import beta
eps = 0.0000000001

class PP_CS:
    def __init__(self, rv_gen, noise_add, n, N, delta, f_noise=0.0, labelled_data=None, unlabelled_data=None, betting_mode='WSR'): #f_noise: std
        self.n = n
        self.N = N
        self.betting_mode = betting_mode
        if labelled_data is None: # deprecated, always get the data set first
            self.X_n = self.noisy_treat(rv_gen(n))
        else:
            self.X_n = labelled_data
        if unlabelled_data is None: # deprecated, always get the data set first
            self.tilde_X_N = self.noisy_treat(noise_add(rv_gen(N), f_noise))
        else:
            self.tilde_X_N = self.noisy_treat(noise_add(unlabelled_data, f_noise))
        self.tilde_X_n = self.noisy_treat(noise_add(self.X_n, f_noise))
        self.delta = delta
    @staticmethod
    def noisy_treat(A):
        return np.clip(A, a_min=0, a_max=1)

    def reset(self, mode, M, num_grid, num_grid_UP=10000):
        if mode == 'LTT':
            self.rho_cand_set = np.array([0.0])
            self.c = 3/4
        elif mode == 'PPLTT':
            self.rho_cand_set = np.array([1.0])
            self.c = 3/4
        else:
            self.rho_cand_set = np.linspace(0,1,M)
            self.c = 3/4
        self.mode = mode
        self.t = 1
        self.n_t = 0
        self.N_t = 0
        self.q_t = 1.0
        if self.betting_mode == 'WSR':
            self.per_rho_running_sum = 0.5*np.ones([M])
            self.per_rho_running_sq_sum = 0.25*np.ones([M])
        elif self.betting_mode == 'UP':
            self.E_UP_ucb = np.ones([M, num_grid, num_grid_UP])
            self.E_UP_lcb = np.ones([M, num_grid, num_grid_UP])
            self.bet_grid = np.expand_dims(np.linspace(0+eps,1-eps,num_grid_UP), axis=0)
            self.prev_obs = None
            self.beta_pdf_grid = beta.pdf(self.bet_grid, 0.5, 0.5)
            self.beta_pdf_grid = self.beta_pdf_grid/np.sum(self.beta_pdf_grid) # deal with discretization
        else:
            raise NotImplementedError
        self.E_grid_lcb = np.ones([num_grid])
        self.E_grid_ucb = np.ones([num_grid])
        self.lcb = 0.0
        self.ucb = 1.0
        self.per_rho_E_grid_lcb = np.ones([M, num_grid])
        self.per_rho_E_grid_ucb = np.ones([M, num_grid])
        self.w_grid_lcb = np.ones([M, num_grid])/M
        self.w_grid_ucb = np.ones([M, num_grid])/M
        self.num_grid = num_grid

    def compute_PP_mean(self):
        self.b_t = 1
        self.lab_t = self.X_n[self.n_t]
        self.corr_t = self.tilde_X_n[self.n_t]
        num_unlabelled_used_now = int(np.clip((np.floor(self.q_t*(self.N-self.N_t)/(self.n-self.n_t))), a_min=1, a_max=self.N-self.N_t))
        self.pred_t = np.mean(self.tilde_X_N[self.N_t:self.N_t+num_unlabelled_used_now])
        self.n_t += self.b_t
        self.N_t += num_unlabelled_used_now
        self.t += 1
        return self.rho_cand_set*self.pred_t + (self.lab_t - self.rho_cand_set*self.corr_t )*self.b_t/self.q_t

    @staticmethod
    def e_bet_ucb(bet, m_grid, rho, obs):
        assert len(obs.shape) == 1
        assert len(rho.shape) == 1
        assert len(m_grid.shape) == 1
        tmp1 = -np.expand_dims(obs, axis=1) + np.expand_dims(m_grid, axis=0) 
        tmp2 = np.expand_dims(rho,axis=1)-np.expand_dims(m_grid, axis=0)
        return 1 + np.expand_dims(tmp1/(1+tmp2), axis=2)*bet.reshape(1, 1, -1) 

    @staticmethod
    def e_bet_lcb(bet, m_grid, rho, obs): # this can be replaced with ucb function by considering m+M-obs; such explicit lcb function might be useful when one wishes to use different betting for ucb and lcb
        assert len(obs.shape) == 1
        assert len(rho.shape) == 1
        assert len(m_grid.shape) == 1
        tmp1 = -np.expand_dims(obs, axis=1) + np.expand_dims(m_grid, axis=0)
        tmp2 = np.expand_dims(rho,axis=1)+np.expand_dims(m_grid, axis=0)
        return 1 + np.expand_dims(-tmp1/tmp2, axis=2)*bet.reshape(1, 1, -1) 

    def e_process_lcb_ucb(self, m_grid, theta):
        if self.betting_mode == 'WSR':
            self.per_rho_mu_t = self.per_rho_running_sum/self.t
            self.per_rho_var_t = self.per_rho_running_sq_sum/self.t
            self.per_rho_nu_t_tilde = np.sqrt( (2*np.log(2/self.delta))/(self.n*self.per_rho_var_t) ) # fixed n
            # https://arxiv.org/pdf/2412.11174 Algo 4
            self.per_rho_nu_t_ucb = np.clip(np.expand_dims(self.per_rho_nu_t_tilde, axis=1), a_min=0.0, a_max= 1/(1+2*np.expand_dims(self.rho_cand_set,axis=1))) # broadcasting
            self.per_rho_nu_t_lcb = np.clip(np.expand_dims(self.per_rho_nu_t_tilde, axis=1), a_min=0.0, a_max= 1/(1+2*np.expand_dims(self.rho_cand_set,axis=1))) # broadcasting
        else:
            if self.prev_obs is None:
                curr_UP_ucb = np.ones(self.E_UP_ucb.shape)
                curr_UP_lcb = np.ones(self.E_UP_lcb.shape)
            else:
                curr_UP_ucb = self.e_bet_ucb(self.bet_grid, m_grid, self.rho_cand_set, self.prev_obs)
                curr_UP_lcb = self.e_bet_lcb(self.bet_grid, m_grid, self.rho_cand_set, self.prev_obs)
            self.E_UP_ucb *= curr_UP_ucb
            self.E_UP_lcb *= curr_UP_lcb
            self.per_rho_nu_t_ucb = np.sum(self.E_UP_ucb*self.bet_grid.reshape(1,1,-1)*self.beta_pdf_grid.reshape(1,1,-1), axis=2)/np.sum(self.E_UP_ucb*self.beta_pdf_grid.reshape(1,1,-1), axis=2)
            self.per_rho_nu_t_ucb /= (1 - np.expand_dims(m_grid, axis=0)+np.expand_dims(self.rho_cand_set, axis=1)) # for consistent expression with other code
            self.per_rho_nu_t_lcb = np.sum(self.E_UP_lcb*self.bet_grid.reshape(1,1,-1)*self.beta_pdf_grid.reshape(1,1,-1), axis=2)/np.sum(self.E_UP_lcb*self.beta_pdf_grid.reshape(1,1,-1), axis=2)
            self.per_rho_nu_t_lcb /= (np.expand_dims(m_grid, axis=0)+np.expand_dims(self.rho_cand_set, axis=1)) # for consistent expression with other code
        per_rho_X_t = self.compute_PP_mean()
        curr_e_grid_lcb = 1+self.per_rho_nu_t_lcb*(np.expand_dims(per_rho_X_t,axis=1)-m_grid)
        curr_e_grid_ucb = 1-self.per_rho_nu_t_ucb*(np.expand_dims(per_rho_X_t,axis=1)-m_grid)
        self.E_grid_lcb *= np.sum( self.w_grid_lcb*curr_e_grid_lcb, axis=0)
        self.E_grid_ucb *= np.sum( self.w_grid_ucb*curr_e_grid_ucb, axis=0)
        self.lcb_ind = np.where(self.E_grid_lcb<1/(theta*self.delta))[0][0] if np.any(self.E_grid_lcb < 1/(theta*self.delta)) else 0
        self.ucb_ind = np.where(self.E_grid_ucb<1/((1-theta)*self.delta))[0][-1] if np.any(self.E_grid_ucb < 1/((1-theta)*self.delta)) else self.num_grid-1
        
        if self.lcb_ind >= 1:
            curr_lcb = m_grid[self.lcb_ind-1] 
        else:
            curr_lcb = 0.0
        if self.ucb_ind <= self.num_grid-2:
            curr_ucb = m_grid[self.ucb_ind+1]
        else:
            curr_ucb = 1.0
        if curr_lcb > self.lcb:
            self.lcb = curr_lcb 
        if curr_ucb  < self.ucb:
            self.ucb = curr_ucb 
        self.per_rho_E_grid_lcb *= curr_e_grid_lcb
        self.per_rho_E_grid_ucb *= curr_e_grid_ucb
        self.w_grid_lcb = self.per_rho_E_grid_lcb/np.sum(self.per_rho_E_grid_lcb, axis=0)
        self.w_grid_ucb = self.per_rho_E_grid_ucb/np.sum(self.per_rho_E_grid_ucb, axis=0)
        if self.betting_mode == 'WSR':
            self.per_rho_running_sum += per_rho_X_t
            self.per_rho_running_sq_sum += (per_rho_X_t- (self.per_rho_running_sum/self.t))**2
        else:
            self.prev_obs = per_rho_X_t

    def forward(self, mode, num_grid = 10000, theta=0.5):
        m_grid = np.linspace(0,1,num_grid)
        if 'PPLTT++' in mode:
            if 'aPPLTT' in mode:
                M = int(mode[8:])
            else:
                M = int(mode[7:])
        else:
            M = 1
        self.reset(mode, M, num_grid)
        while self.n_t < self.n and self.N_t < self.N:
            self.e_process_lcb_ucb(m_grid, theta)
        return (self.lcb, self.ucb)
