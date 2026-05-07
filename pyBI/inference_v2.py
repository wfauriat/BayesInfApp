from warnings import warn
from typing import Optional

import numpy as np ## MAIN REQUIREMENT
from itertools import combinations, chain ## GENERIC TOOLS

from matplotlib.pyplot import subplots ## FOR VISUALISATION

from scipy.stats import gaussian_kde ## ONLY FOR POSTERIOR VISUALISATION

from pyBI.base import rnvmultiD
from pyBI.base import ObsVar, RandVar


###############################################################################
## class InfAlgo:
############################################################################### 
class InfAlgo:
    """
    Generic inference algorithm object
    """
    def __init__(self, N: int, Nthin: int = 1, Nburn: int = 0.1,
                 is_adaptive: bool = True, verbose: bool = True) -> None:
        """
        Instanciation of an MCMC-based inference algorithm

        Parameters to be defined whith first instanciation
        ----------
        N : int 
            Size of the generated sample
        Nthin : int (default 1)
            Number of steps between the sample that are eventually kept
        is_adaptive : bool (default True)
            Enables adpatative tuning of proposal (uni or multi-directional)
        verbose : bool (default True)
            Display additionnal information during inference and at the
            end of the inference

        """
        self.Nburn: int = int(N*0.1) if Nburn == 0.1 else Nburn
        self.Nthin: int = Nthin if Nthin else 1
        self.N: int = N + self.Nburn
        self.log_state: float = 0.
        self.prior_state: float = 0.
        self.is_adaptive: bool = is_adaptive
        self.verbose: bool = verbose
        self.ran_chain: bool = False

    def __str__(self):
        strout = ""
        strout += str(type(self)) + '\n'
        strout += 'Number of generated points : '  + \
              str(self.N - self.Nburn) + '\n'
        strout += 'Thinning : '  + str(self.Nthin) + '\n'
        strout += 'Number of retained points : '  + \
              str(int((self.N - self.Nburn)/self.Nthin)) + '\n'
        if self.ran_chain:
            strout += 'acceptation rate : ' + str(self.tacc) + '\n'
            strout += 'MAP : ' + ", ".join(["{:.3f}".format(el) 
                                            for el in self.MAP]) + '\n'
            return strout
        else: return strout

    # def __doc__():
    #     return "see __init__()"

    def initialize(self, obsObj, varObj, discrObj=None,
                    svar: float = 1., sdisc: float = 1.,
                    Lblock: Optional[np.ndarray] = None):
        """
        Definition of the necessary objects to instantiate the inference
        problem (prior and likelihood functions).

        If discrObj is None, the noise covariance is taken to be FIXED and
        provided by obsObj (e.g. a KnownCovObs instance). The chain then
        has Ndim slots instead of Ndim+1, and obsObj.loglike(par) is called
        directly. Otherwise the legacy iid-sigma path (loglike_uv with a
        nuisance sigma) is used.
        """
        self.Ndim = len(varObj)
        self.varObj = varObj
        self.discrObj = discrObj
        self.obsObj = obsObj
        self._fixed_cov = (discrObj is None)
        chain_w = self.Ndim if self._fixed_cov else self.Ndim + 1
        self._chain_w = chain_w
        self.MCchain = np.zeros((self.N, chain_w))
        self.llchain = np.zeros(self.N)
        self.svar = svar*np.ones(self.Ndim) if isinstance(svar,(int,float)) \
            else np.asarray(svar, dtype=float)
        if not self._fixed_cov:
            self.sdisc = sdisc if sdisc else 1.
            if sdisc is not None: self.sdisc = sdisc
            else: self.sdisc = discrObj.param
        if isinstance(Lblock,(int,float)):
            self.Lblock = np.eye(self.Ndim)
        elif Lblock is None:
            self.Lblock = np.diag(self.svar)
        else: self.Lblock = Lblock
        self.MCchain[0,:self.Ndim] = [varObj[i].draw()
                                      for i in range(len(varObj))]
        if not self._fixed_cov:
            self.MCchain[0, self.Ndim] = discrObj.draw()
        self.state(0, set_state=True)

    def state(self, i, set_state=False):
        """
        Evaluation of current step:
        Computation of log-likelihood given current state
        Computation of log-prior of current state
        """
        par = self.MCchain[i, :self.Ndim]
        if self._fixed_cov:
            log_state = self.obsObj.loglike(par)
            prior_state = float(np.sum(
                [self.varObj[k].logprior(self.MCchain[i, k])
                 for k in range(self.Ndim)]))
        else:
            log_state = self.obsObj.loglike_uv(par,
                                            self.MCchain[i, self.Ndim])
            prior_state = np.sum([self.varObj[k].logprior(self.MCchain[i,k])
                                  for k in range(len(self.varObj))]) + \
                              self.discrObj.logprior(self.MCchain[i,self.Ndim])
        if set_state:
            self.llchain[i] = log_state
            self.log_state = log_state
            self.prior_state = prior_state
        else: return log_state, prior_state
    
    def move_prop(self, x0=None, i=0, Lblock=None):
        """
        Proposal of a new state value given current state
        """
        if Lblock is not None: Ltmp = Lblock # rather for testing purposes
        else: Ltmp = self.Lblock# default behaviour
        if x0 is not None: # default behaviour
            xprop = list(rnvmultiD(np.array(x0[:self.Ndim]), Ltmp))
            sprop = self.discrObj.proposal(x0[self.Ndim], self.sdisc)
        else: # rather for visualisation purpose
            xprop = list(rnvmultiD(np.array(self.MCchain[i,:self.Ndim]), Ltmp))
            sprop = self.discrObj.proposal(self.MCchain[i,self.Ndim],
                                            self.sdisc)
        return xprop + [sprop]
    
    def move_uni(self, di, x0):
        """
        Proposal of a new state value in a given direction (given current state)
        """
        xprop = list(x0[:self.Ndim])
        if self._fixed_cov:
            xprop[di] = self.varObj[di].proposal(x0[di], self.svar[di])
            return xprop
        sprop = x0[self.Ndim]
        if di < self.Ndim:
            xprop[di] = self.varObj[di].proposal(x0[di], self.svar[di])
        else: sprop = self.discrObj.proposal(x0[di], self.sdisc)
        return xprop + [sprop]
    
    def store(self, i, xprop, sprop=None):
        """
        Store current proposal in the MCchain
        """
        self.MCchain[i,:self.Ndim] = xprop
        if (not self._fixed_cov) and (sprop is not None):
            self.MCchain[i, self.Ndim] = sprop
        
    def stay(self, i, j=None):
        """
        Discard new proposed state (unidirection or multidirectional)
        and come back to previous (or current) state
        """
        if j is None:
            self.MCchain[i] = self.MCchain[i-1]
            self.llchain[i] = self.llchain[i-1]
        else: 
            self.MCchain[i,j] = self.MCchain[i-1,j]
            self.llchain[i] = self.log_state

    def thin_and_sort(self):
        """
        Post process result of carried out inference
        """
        if not self.ran_chain: 
            self.ran_chain = True
            self.raw_chain = self.MCchain
            self.cut_chain = self.raw_chain[self.Nburn::self.Nthin]
            self.cut_llchain = self.llchain[self.Nburn::self.Nthin]
            self.sorted_indices = np.argsort(self.cut_llchain)
            self.idx_chain = self.cut_chain[self.sorted_indices,:]
            self.MAP = self.idx_chain[-1]
            return self.idx_chain, self.cut_llchain[self.sorted_indices]
        else: 
            warn("Already ran and sorted")
            return self.idx_chain, self.cut_llchain[self.sorted_indices]
        
    def post_obs(self, model=None) -> np.ndarray:
        """
        Propagate posterior through the prevision model
        """
        Mtmp = model if model is not None else self.obsObj.prev_model
        postY = np.zeros((int((self.N - self.Nburn)/self.Nthin),
                          self.obsObj.dimdata))
        for i in range(postY.shape[0]):
            mu = np.ravel(Mtmp(self.obsObj.cond_var,
                              self.idx_chain[i,:self.Ndim]))
            if self._fixed_cov:
                kind = getattr(self.obsObj, 'kind', None)
                if kind == 'diag':
                    noise = np.random.randn(*mu.shape) * \
                            np.sqrt(self.obsObj.sigma2)
                else:
                    noise = 0.0
                postY[i,:] = mu + noise
            else:
                postY[i,:] = mu + \
                    np.random.randn()*self.idx_chain[i,self.Ndim]
        return postY
    
    def post_visuobs(self, di=0):
        """
        Visualize (or propagate) posterior in the observational space
        """
        fig, ax = subplots()
        postY = self.post_obs()
        ax.plot(self.obsObj.cond_var[:,di], postY.T, '.k')
        ax.plot(self.obsObj.cond_var[:,di], self.obsObj.obs[0,:], 'or')
        return fig, ax
    
    def post_visupar(self):
        """
        Visualize posterior in the parameter space
        """
        MC = self.idx_chain
        LL = self.cut_llchain[self.sorted_indices]
        labels = ['v' + str(i) for i in range(self.Ndim)]
        if not self._fixed_cov:
            labels = labels + ['d']
        n_axes = self._chain_w
        quadrans = list(combinations(labels, 2))
        quad_idx = list(combinations(list(range(n_axes)), 2))
        Nbplot = int(np.ceil(np.sqrt(len(quadrans))))
        k = 0
        fig, ax = subplots(Nbplot, Nbplot)
        for i in range(Nbplot):
            for j in range(Nbplot):
                if k<len(quadrans):
                    ax[i,j].scatter(MC[:,quad_idx[k][0]], MC[:,quad_idx[k][1]],
                                    c=LL, cmap='jet', marker='.')
                    ax[i,j].set_xlabel(quadrans[k][0])
                    ax[i,j].set_ylabel(quadrans[k][1])
                    ax[i,j].plot(self.MAP[quad_idx[k][0]],
                                  self.MAP[quad_idx[k][1]], 'dk')
                    k += 1
                else: ax[i,j].set_visible(False)
        fig.tight_layout()
        return fig, ax
    
    def diag_chain(self, di=0, show_prior=True, axf=None):
        """
        Visualize the diagnostic associated to a given dimension of the 
        sampled chain
        """
        MC = self.cut_chain
        Nc = int(self.cut_chain.shape[0]/4)
        labels = ['v' + str(i) for i in range(self.Ndim)]
        if not self._fixed_cov:
            labels = labels + ['d']
        xi = np.linspace(MC[:,di].min(), MC[:,di].max())
        kdes = [gaussian_kde(MC[i*Nc:(i+1)*Nc,di]) for i in range(0,4)]
        if axf is None:
            fig, ax = subplots(2,1)
            ax[0].plot(self.raw_chain[:(1*Nc*self.Nthin + self.Nburn),di],
                                    '-k', lw=0.5)
            ax[0].axvline(x=self.Nburn, ls='--', color='r')
            ax[1].hist(self.idx_chain[:,di], edgecolor='b', alpha=0.4)
            axx = ax[1].twinx()
            ax[1].set_xlabel(labels[di])
        else: 
            axf.hist(self.idx_chain[:,di], edgecolor='b', alpha=0.4)
            axf.set_xlabel(labels[di])
            axx = axf.twinx()
        for i in range(4):
            axx.plot(xi, kdes[i].pdf(xi), 'k')
        if show_prior:
            if di < self.Ndim:
                for i in range(4):
                    xp = np.linspace(
                        self.varObj[di].min,self.varObj[di].max, 100)
                    axx.plot(xp, np.exp([self.varObj[di].logprior(x) 
                                        for x in xp]), 'r')
            elif not self._fixed_cov:
                for i in range(4):
                    xp = np.linspace(
                        self.discrObj.min,self.discrObj.max, 100)
                    axx.plot(xp, np.exp([self.discrObj.logprior(x) 
                                        for x in xp]), 'r')
        if axf is None: return fig, ax
        else: return axf
    
    def hist_alldim(self, figsize=(10,4)):
        fig, ax = subplots(1,self._chain_w, figsize=figsize)
        if self._chain_w == 1:
            ax = [ax]
        for i in range(self._chain_w):
            self.diag_chain(di=i, axf=ax[i])
        fig.tight_layout()

    def simple_pairplot(self, d1, d2):
        fig, ax = subplots()
        ax.scatter(self.idx_chain[:,d1], self.idx_chain[:,d2],
                c=self.cut_llchain[self.sorted_indices], cmap='jet')


###############################################################################
## class MHalgo(InfAlgo):
############################################################################### 
class MHalgo(InfAlgo):
    """
    Subclass for instanciating a standard Metropolis Hasting inference 
    """
    def __init__(self, N, Nthin=1, Nburn=0.1, is_adaptive=True,
                 verbose=True):
        super().__init__(N, Nthin, Nburn, is_adaptive, verbose)

    def runInference(self):
        self.nacc = np.zeros(2) 
        for i in range(1,self.N):
            if self._fixed_cov:
                # Block proposal on parameters only
                xprop = list(rnvmultiD(np.array(self.MCchain[i-1, :self.Ndim]),
                                       self.Lblock))
                self.store(i, xprop)
            else:
                prop = self.move_prop(self.MCchain[i-1])
                self.store(i, prop[:self.Ndim], self.MCchain[i-1,self.Ndim])
            llprop, lpprop = self.state(i)
            ldiff = llprop + lpprop - self.log_state - self.prior_state
            if ldiff > np.log(np.random.rand()):
                if i>self.Nburn: self.nacc[0] += 1
                self.llchain[i] = llprop
                self.log_state = llprop
                self.prior_state = lpprop
            else: self.stay(i)
            if not self._fixed_cov:
                prop = self.move_prop(self.MCchain[i])
                self.store(i, self.MCchain[i,:self.Ndim], prop[self.Ndim])
                llprop, lpprop = self.state(i)
                ldiff = llprop + lpprop - self.log_state - self.prior_state
                if ldiff > np.log(np.random.rand()):
                    if i>self.Nburn: self.nacc[1] += 1
                    self.llchain[i] = llprop
                    self.log_state = llprop
                    self.prior_state = lpprop
                else: self.stay(i,self.Ndim)
            if (i%1000 == 0) & (i<=self.Nburn) & (self.verbose):
                print('Burn-in : ' + str(i))
            if (i%500 == 0) & (i<self.Nburn) & (self.is_adaptive):
                covProp = np.cov(self.MCchain[i-500:i,:self.Ndim].T) 
                LLTprop = np.linalg.cholesky(covProp * 2.38**2/(self.Ndim-1) +
                                            np.eye(self.Ndim)*1e-8)
                self.Lblock = LLTprop
                if not self._fixed_cov:
                    self.sdisc = np.std(self.MCchain[i-500:i,self.Ndim])*1
            if (i%1000 == 0) & (i>=self.Nburn) & (self.verbose):
                print('Chain : ' + str(i - self.Nburn))
        MCS = self.thin_and_sort()
        self.tacc = self.nacc/(self.N - self.Nburn)
        if self.verbose: print('acceptation rate', self.tacc)
        return MCS[0], MCS[1]


###############################################################################
## class MHwGalgo(InfAlgo):
############################################################################### 
class MHwGalgo(InfAlgo):
    """
    Subclass for instanciating a Metropolis within Gibbs inference
    """
    def __init__(self, N, Nthin=1, Nburn=0.1, is_adaptive=True,
                 verbose=False):
        super().__init__(N, Nthin, Nburn, is_adaptive, verbose)
        # self.target_acc = 0.3

    def runInference(self):
        n_blocks = self.Ndim if self._fixed_cov else self.Ndim + 1
        self.nacc = np.zeros(n_blocks)
        self.interval_acc = np.zeros(n_blocks)
        TUNE_INTERVAL = 500
        for i in range(1,self.N):
            self.MCchain[i] = self.MCchain[i-1]
            id_rnd = np.random.permutation(n_blocks)
            for idx in id_rnd:
                prop = self.move_uni(idx, self.MCchain[i])
                if self._fixed_cov:
                    self.store(i, prop[:self.Ndim])
                else:
                    self.store(i, prop[:self.Ndim], prop[self.Ndim])
                llprop, lpprop = self.state(i)
                ldiff = llprop + lpprop - self.log_state - self.prior_state
                if ldiff > np.log(np.random.rand()):
                    if i>self.Nburn: self.nacc[idx] += 1
                    if i<self.Nburn: self.interval_acc[idx] += 1
                    self.llchain[i] = llprop
                    self.log_state = llprop
                    self.prior_state = lpprop
                else: self.stay(i,idx)
            if (i%1000 == 0) & (i<=self.Nburn) & (self.verbose):
                print('Burn-in : ' + str(i))
            if (i%TUNE_INTERVAL == 0) & (i<self.Nburn) & (self.is_adaptive):
                for idx in range(self.Ndim):
                    self.svar[idx] = self.tune_scale(self.svar[idx],
                                                     self.interval_acc[idx], 
                                                    TUNE_INTERVAL)
                if not self._fixed_cov:
                    self.sdisc = self.tune_scale(self.sdisc, 
                                                 self.interval_acc[self.Ndim],
                                                  TUNE_INTERVAL)
                self.interval_acc = np.zeros(n_blocks)
                # OPTION 1 !!!!!!
                # for idx in id_rnd:
#                     if idx != self.Ndim:
#                         self.svar[idx] = np.std(self.MCchain[i-500:i,idx])*2.38**2
#                     else:
#                         self.sdisc = np.std(self.MCchain[i-500:i,idx])*1
#                 self.Lblock = np.diag(self.svar)

                # OPTION 2 !!!!!!
                # t = i // TUNE_INTERVAL 
                # gamma_t = 1.0 / (t + t0)**ALPHA 
                # acc_rate = interval_acc[idx] / TUNE_INTERVAL
                # log_sigma_t = np.log(self.svar[idx] if idx != self.Ndim \
                #                      else self.sdisc)
                # log_sigma_t += gamma_t * (acc_rate - self.target_acc)
                # new_sigma = np.exp(log_sigma_t)
                # if idx != self.Ndim:
                #     self.svar[idx] = new_sigma 
                # else:
                #     self.sdisc = new_sigma
            if (i%1000 == 0) & (i>=self.Nburn) & (self.verbose):
                print('Chain : ' + str(i - self.Nburn))


        MCS = self.thin_and_sort()
        self.tacc = self.nacc/(self.N - self.Nburn)
        if self.verbose: print('acceptation rate', self.tacc)
        return MCS[0], MCS[1]
    
    def tune_scale(self, sin, nacc, interv):
        sout = sin * 1 # if between 0.2 and 0.5
        if nacc / interv < 0.2:
            sout = sin * 0.9
        if nacc / interv < 0.05:
            sout = sin * 0.5
        if nacc / interv < 0.001:
            sout = sin * 0.1
        if nacc / interv > 0.5:
            sout = sin * 1.1
        if nacc / interv > 0.75:
            sout = sin * 2
        if nacc / interv > 0.95:
            sout = sin * 10
        return sout



###############################################################################
## class InfAlgo2: (SHOULD BE REWRITTEN TO ACCOUNT FOR ALGO MODIF IN 1)
## EXPERIMENTAL AND NOT UP TO DATE
############################################################################### 

class InfAlgo2:
    """
    Generic inference algorithm object
    """
    def __init__(self, N: int, Nthin: int = 1, Nburn: int = 0.1,
                 is_adaptive: bool = True, verbose: bool = True) -> None:
        """
        Instanciation of an MCMC-based inference algorithm

        Parameters to be defined whith first instanciation
        ----------
        N : int 
            Size of the generated sample
        Nthin : int (default 1)
            Number of steps between the sample that are eventually kept
        is_adaptive : bool (default True)
            Enables adpatative tuning of proposal (uni or multi-directional)
        verbose : bool (default True)
            Display additionnal information during inference and at the
            end of the inference

        """
        self.Nburn: int = int(N*0.1) if Nburn == 0.1 else Nburn
        self.Nthin: int = Nthin
        self.N: int = N + self.Nburn
        self.log_state: float = 0.
        self.prior_state: float = 0.
        self.is_adaptive: bool = is_adaptive
        self.verbose: bool = verbose
        self.ran_chain: bool = False

    def __str__(self):
        strout = ""
        strout += str(type(self)) + '\n'
        strout += 'Number of generated points : '  + \
              str(self.N - self.Nburn) + '\n'
        strout += 'Thinning : '  + str(self.Nthin) + '\n'
        strout += 'Number of retained points : '  + \
              str(int((self.N - self.Nburn)/self.Nthin)) + '\n'
        if self.ran_chain:
            strout += 'acceptation rate : ' + str(self.tacc) + '\n'
            strout += 'MAP : ' + ", ".join(["{:.3f}".format(el) 
                                            for el in self.MAP]) + '\n'
            return strout
        else: return strout

    # def __doc__():
    #     return "see __init__()"

    def initialize(self, obsObj, varObj, svar):
        self.obsObj = obsObj
        self.varObj = varObj
        self.vdim = np.sum([len(self.varObj[el]) for el in self.varObj.keys()])
        self.svar = svar
        self.MCchain = np.zeros((self.N, self.vdim))
        self.llchain = np.zeros(self.N)
        self.MCchain[0,:] = self.draw_all_vars()
        self.state(0, set_state=True)
    
    def draw_all_vars(self):
        rnd = np.zeros(self.vdim)        
        i = 0
        for el in self.varObj.keys():
            for k in range(len(self.varObj[el])):
                rnd[i] = self.varObj[el][k].draw()
                i+=1
        return rnd

    def place_var(self):
        place_g = len(self.varObj['gpar'])
        place_k = len(self.varObj['kpar'])
        place_s = len(self.varObj['spar'])
        return place_g, place_g + place_k, place_g + place_k + place_s

    def category_from_chain(self, x):
        gpar = list(x[:self.place_var()[0]])
        kpar = list(x[self.place_var()[0]:self.place_var()[1]])
        spar = list(x[self.place_var()[1]:self.place_var()[2]])
        return gpar, kpar, spar
    
    def chain_from_category(self, catElem):
        x = []
        for (el,j) in zip(self.varObj.keys(), range(3)):
            for k in range(len(self.varObj[el])):
                x.append(catElem[j][k])
        return x
    
    def flat_varobj(self, varobj):
        flat_list = []
        for el in varobj.keys():
            for j in range(len(varobj[el])):
                flat_list.append(varobj[el][j])
        return flat_list
    
    def proposal_all(self, x0, di=None):
        catElem = self.category_from_chain(x0)
        catProp = [[], [], []]
        curr = -1
        for (el,j) in zip(self.varObj.keys(), range(3)):
            for k in range(len(self.varObj[el])):
                curr +=1
                if di is None:
                    catProp[j].append(self.varObj[el][k].proposal(
                        catElem[j][k], self.svar[j][k]))
                elif di != curr: catProp[j].append(catElem[j][k])
                elif di == curr: catProp[j].append(self.varObj[el][k].proposal(
                        catElem[j][k], self.svar[j][k]))
        return catProp

    def state(self, i, set_state=False):
        """
        Evaluation of current step:
        Computation of log-likelihood given current state
        Computation of log-prior of current state
        """
        catElem = self.category_from_chain(self.MCchain[i])
        self.obsObj.setpar(catElem[0], catElem[1], catElem[2])
        log_state = self.obsObj.loglike()
        log_prior = [[self.varObj[el][k].logprior(catElem[j][k])
                    for k in range(len(self.varObj[el]))] 
                        for (el,j) in zip(self.varObj.keys(),range(3))]
        log_prior_sum = np.sum(list(chain(*log_prior)))
        if set_state:
            self.llchain[i] = log_state
            self.log_state = log_state
            self.prior_state = log_prior_sum
        else: return log_state, log_prior_sum
    
    def move_prop(self, x0=None, i=0, Lblock=None):
        """
        Proposal of a new state value given current state
        """
        if Lblock is not None: Ltmp = Lblock # rather for testing purposes
        else: Ltmp = self.Lblock# default behaviour
        if x0 is not None: # default behaviour
            xprop = list(rnvmultiD(np.array(x0[:self.Ndim]), Ltmp))
            sprop = self.discrObj.proposal(x0[self.Ndim], self.sdisc)
        else: # rather for visualisation purpose
            xprop = list(rnvmultiD(np.array(self.MCchain[i,:self.Ndim]), Ltmp))
            sprop = self.discrObj.proposal(self.MCchain[i,self.Ndim],
                                            self.sdisc)
        return xprop + [sprop]
    
    def move_uni(self, di, x0):
        """
        Proposal of a new state value in a given direction (given current state)
        """
        prop = self.proposal_all(x0, di=di)
        return prop
    
    def store(self, i, prop):
        """
        Store current proposal in the MCchain
        """
        self.MCchain[i,:] = self.chain_from_category(prop)
        
    def stay(self, i, j=None):
        """
        Discard new proposed state (unidirection or multidirectional)
        and come back to previous (or current) state
        """
        if j is None:
            self.MCchain[i] = self.MCchain[i-1]
            self.llchain[i] = self.llchain[i-1]
        else: 
            self.MCchain[i,j] = self.MCchain[i-1,j]
            self.llchain[i] = self.log_state

    def thin_and_sort(self):
        """
        Post process result of carried out inference
        """
        if not self.ran_chain: 
            self.ran_chain = True
            self.raw_chain = self.MCchain
            self.cut_chain = self.raw_chain[self.Nburn::self.Nthin]
            self.cut_llchain = self.llchain[self.Nburn::self.Nthin]
            self.sorted_indices = np.argsort(self.cut_llchain)
            self.idx_chain = self.cut_chain[self.sorted_indices,:]
            self.MAP = self.idx_chain[-1]
            return self.idx_chain, self.cut_llchain[self.sorted_indices]
        else: 
            warn("Already ran and sorted")
            return self.idx_chain, self.cut_llchain[self.sorted_indices]
        
    def post_obs(self, model=None) -> np.ndarray:
        """
        Propagate posterior through the prevision model
        """
        Mtmp = model if model is not None else self.obsObj.prev_model
        postY = np.zeros((int((self.N - self.Nburn)/self.Nthin),
                          self.obsObj.dimdata))
        for i in range(postY.shape[0]):
            postY[i,:] = Mtmp(self.obsObj.cond_var,
                             self.idx_chain[i,:self.Ndim]) + \
                        np.random.randn()*self.idx_chain[i,self.Ndim]
        return postY
    
    def post_visuobs(self, di=0):
        """
        Visualize (or propagate) posterior in the observational space
        """
        fig, ax = subplots()
        postY = self.post_obs()
        ax.plot(self.obsObj.cond_var[:,di], postY.T, '.k')
        ax.plot(self.obsObj.cond_var[:,di], self.obsObj.obs[0,:], 'or')
        return fig, ax
    
    def post_visupar(self):
        """
        Visualize posterior in the parameter space
        """
        MC = self.idx_chain
        LL = self.cut_llchain[self.sorted_indices]
        quadrans = list(combinations(
                ['v' + str(i) for i in range(self.vdim)],2))
        quad_idx = list(combinations(list(range(self.vdim)),2))
        Nbplot = int(np.ceil(np.sqrt(len(quadrans))))
        k = 0
        fig, ax = subplots(Nbplot, Nbplot)
        for i in range(Nbplot):
            for j in range(Nbplot):
                if k<len(quadrans):
                    ax[i,j].scatter(MC[:,quad_idx[k][0]], MC[:,quad_idx[k][1]],
                                    c=LL, cmap='jet', marker='.')
                    ax[i,j].set_xlabel(quadrans[k][0])
                    ax[i,j].set_ylabel(quadrans[k][1])
                    ax[i,j].plot(self.MAP[quad_idx[k][0]],
                                  self.MAP[quad_idx[k][1]], 'dk')
                    k += 1
                else: ax[i,j].set_visible(False)
        fig.tight_layout()
        return fig, ax
    
    def diag_chain(self, di=0, show_prior=True, axf=None):
        """
        Visualize the diagnostic associated to a given dimension of the 
        sampled chain
        """
        MC = self.cut_chain
        Nc = int(self.cut_chain.shape[0]/4)
        xi = np.linspace(MC[:,di].min(), MC[:,di].max())
        kdes = [gaussian_kde(MC[i*Nc:(i+1)*Nc,di]) for i in range(0,4)]
        flat_varobj = self.flat_varobj(self.varObj)
        if axf is None:
            fig, ax = subplots(2,1)
            ax[0].plot(self.raw_chain[:(1*Nc*self.Nthin + self.Nburn),di],
                                    '-k', lw=0.5)
            ax[1].hist(self.idx_chain[:,di], edgecolor='b', alpha=0.4)
            axx = ax[1].twinx()
        else: 
            axf.hist(self.idx_chain[:,di], edgecolor='b', alpha=0.4)
            axx = axf.twinx()
        for i in range(4):
            axx.plot(xi, kdes[i].pdf(xi), 'k')
        if show_prior:
                for i in range(self.vdim):
                    xp = np.linspace(
                        flat_varobj[di].min,flat_varobj[di].max, 100)
                    axx.plot(xp, np.exp([flat_varobj[di].logprior(x) 
                                        for x in xp]), 'r')
        if axf is None: return fig, ax
        else: return axf
    
    def hist_alldim(self, figsize=(10,4)):
        fig, ax = subplots(1,self.vdim, figsize=figsize)
        for i in range(self.vdim):
            self.diag_chain(di=i, axf=ax[i])
        fig.tight_layout()

    def simple_pairplot(self, d1, d2):
        fig, ax = subplots()
        ax.scatter(self.idx_chain[:,d1], self.idx_chain[:,d2],
                c=self.cut_llchain[self.sorted_indices], cmap='jet')



###############################################################################
## class MHwGalgo2(InfAlgo2):
############################################################################### 
class MHwGalgo2(InfAlgo2):
    """
    Subclass for instanciating a Metropolis within Gibbs inference
    """
    def __init__(self, N, Nthin=None, Nburn=0.1, is_adaptive=True,
                 verbose=False):
        super().__init__(N, Nthin, Nburn, is_adaptive, verbose)

    def runInference(self):
        self.nacc = np.zeros(self.vdim) 
        for i in range(1,self.N):
            self.MCchain[i] = self.MCchain[i-1]
            id_rnd = np.random.permutation(self.vdim)
            for idx in id_rnd:
                prop = self.move_uni(idx, self.MCchain[i])
                self.store(i, prop)
                llprop, lpprop = self.state(i)
                ldiff = llprop + lpprop - self.log_state - self.prior_state
                if ldiff > np.log(np.random.rand()):
                    if i>self.Nburn: self.nacc[idx] += 1
                    self.llchain[i] = llprop
                    self.log_state = llprop
                    self.prior_state = lpprop
                else: self.stay(i,idx)
            if (i%1000 == 0) & (self.verbose): print(i)
            if (i%1000 == 0) & (i<=self.Nburn) & (self.is_adaptive):
                for idx in id_rnd:
                    new_svar = np.std(self.MCchain[i-1000:i,idx])*2
                    catSvar = self.chain_from_category(self.svar)
                    catSvar[idx] = new_svar
                    self.svar = self.category_from_chain(catSvar)
        MCS = self.thin_and_sort()
        self.tacc = self.nacc/(self.N - self.Nburn)
        if self.verbose: print('acceptation rate', self.tacc)
        return MCS[0], MCS[1]
    