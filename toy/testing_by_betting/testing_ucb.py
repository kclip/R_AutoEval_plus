import numpy as np
from scipy.stats import beta
eps = 0.0000000001
class Testing:
    def __init__(self, rv_gen, noise_add, r, delta, f_noise=0.0, betting_mode='UP', if_vis_weight=False):
        self.r = r
        self.delta = delta
        self.rv_gen = rv_gen
        self.noise_add = noise_add
        self.f_noise = f_noise
        self.if_vis_weight = if_vis_weight
        self.betting_mode = betting_mode
    @staticmethod
    def noisy_treat(A):
        return np.clip(A, a_min=0, a_max=1)
    def reset(self, alpha, mode, M, num_grid_UP=10000):
        self.alpha = alpha
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
        self.per_rho_running_sum = 0.5*np.ones([M])
        self.per_rho_running_sq_sum = 0.25*np.ones([M])
        self.per_rho_E = np.ones([M,1])
        self.w = np.ones([M,1])/M
        self.E = 1
        self.E_UP = np.ones([M, num_grid_UP])
        self.bet_grid = np.expand_dims(np.linspace(0+eps,1-eps,num_grid_UP), axis=0)
        self.prev_obs = None

        if self.if_vis_weight:
            self.w_history = []

        self.beta_pdf_grid = beta.pdf(self.bet_grid, 0.5, 0.5)
        self.beta_pdf_grid = self.beta_pdf_grid/np.sum(self.beta_pdf_grid) # deal with discretization

    
    def compute_PP_mean(self):
        tmp_1, tmp_2 = self.rv_gen(1, self.r)
        self.X_n = tmp_1
        self.tilde_X_N = self.noise_add(tmp_2, self.f_noise)
        self.tilde_X_n = self.noise_add(self.X_n, self.f_noise)
        self.n_t += 1
        self.t += 1
        return self.rho_cand_set*np.mean(self.tilde_X_N) + (self.X_n - self.rho_cand_set*self.tilde_X_n )
    
    @staticmethod
    def e_bet(bet, alpha, rho, obs):
        return 1 + (-np.expand_dims(obs, axis=1)+alpha)*bet/(1+np.expand_dims(rho,axis=1)-alpha)

    def e_process(self, n=None):
        if self.betting_mode == 'UP':
            if self.prev_obs is None:
                curr_UP = np.ones(self.E_UP.shape)
            else:
                curr_UP = self.e_bet(self.bet_grid, self.alpha, self.rho_cand_set, self.prev_obs)
            self.E_UP *= curr_UP
            self.per_rho_nu_t_ucb = np.sum(self.E_UP*self.bet_grid*self.beta_pdf_grid, axis=1)/np.sum(self.E_UP*self.beta_pdf_grid, axis=1)
            self.per_rho_nu_t_ucb /= (1 -self.alpha+self.rho_cand_set) # for consistent expression with other code
            self.per_rho_nu_t_ucb = np.expand_dims(self.per_rho_nu_t_ucb, axis=1)
        else:
            self.per_rho_mu_t = self.per_rho_running_sum/self.t
            self.per_rho_var_t = self.per_rho_running_sq_sum/self.t
            if n is None: # toy experiments on sample complexity assume this case as n is not fixed a priori; we incrementally increase n until it rejects the null
                self.per_rho_nu_t_tilde = np.sqrt( (2*np.log(2/self.delta))/( (self.t)*np.log(self.t+1) *self.per_rho_var_t) ) # continous n
            else: # not using this option. See PP_CS.py where the fixed_n option is used
                self.per_rho_nu_t_tilde = np.sqrt( (2*np.log(2/self.delta))/( n*self.per_rho_var_t) ) # fixed n
            self.per_rho_nu_t_ucb = np.clip(np.expand_dims(self.per_rho_nu_t_tilde, axis=1), a_min=0.0, a_max=self.c/( 1 -self.alpha+np.expand_dims(self.rho_cand_set,axis=1))) # broadcasting
        per_rho_X_t = self.compute_PP_mean()
        curr_e = 1-self.per_rho_nu_t_ucb*(np.expand_dims(per_rho_X_t,axis=1)-self.alpha)
        self.E *= np.sum( self.w*curr_e, axis=0)
        self.per_rho_E *= curr_e
        self.w = self.per_rho_E/np.sum(self.per_rho_E, axis=0)
        if self.if_vis_weight:
            self.w_history.append(np.expand_dims(self.w, axis=1))
        else:
            pass
        if self.betting_mode == 'UP':
            self.prev_obs = per_rho_X_t
        else:
            self.per_rho_running_sum += per_rho_X_t
            self.per_rho_running_sq_sum += (per_rho_X_t- (self.per_rho_running_sum/self.t))**2
    def forward(self, alpha, mode):
        if 'PPLTT++' in mode:
            M = int(mode[7:])
        else:
            M = 1
        self.reset(alpha, mode, M)

        while self.E < 1/self.delta:
            self.e_process()
        return self.n_t
    def forward_for_vis(self, alpha, mode, n):
        if 'PPLTT++' in mode:
            M = int(mode[7:])
        else:
            M = 1
        self.reset(alpha, mode, M)
        while self.n_t <= n:
            self.e_process()
        return np.concatenate(self.w_history, axis=1)