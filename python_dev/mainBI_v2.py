#%%
import numpy as np
import scipy.stats as sst

import matplotlib.pyplot as plt

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel

from pyBI.base_v2 import UnifVar, NormVar, InvGaussVar, HalfNormVar, ObsVar
from pyBI.base_v2 import KnownCovObs
from pyBI.inference_v2 import MHalgo, MHwGalgo

np.set_printoptions(suppress=True, precision=5)

#%%####
# FOR DEVELOPMENT PURPOSES
#

# import importlib
# import pyBI.base
# new_mod = importlib.reload(pyBI.base)
# GaussLike = new_mod.GaussLike

#%%

# casep = 0
# casep = 1
# casep = 2
casep = 3   # NEW: scalar output, KNOWN AR(1) covariance across observations
# casep = 4   # NEW: vector output (dy=2), KNOWN Kronecker covariance

inftype = 'MH'
# inftype = 'MHwG'

#%%############################################################################
# DEFINITION OF APPLICATION / CALIBRATION CASE
###############################################################################

if casep == 0:
    def modeltrue(x,b):
        return np.atleast_2d(b[0] + b[1]*x[:,0] + b[2]*x[:,0]**2 + \
                             1*x[:,1] + 0.02*x[:,0]**3)
    def modelfit(x,b):
        return np.atleast_2d(b[0] + b[1]*x[:,0] + b[2]*x[:,0]**2)[0]

    b0 = [2, -1, 2, 0]
    nslvl = 0.1
    nsp1 = 0.2
    biasp1 = -1

    xplot = np.repeat(np.c_[np.linspace(0,6,50)],2, axis=1)
    xplot[:,1] = 1

    xmes = np.hstack([np.c_[[0, 0.5, 1, 2, 2.5, 2.8, 4, 4.4, 5.2, 5.5]],
                    biasp1+nsp1*np.c_[np.random.randn(10)]])
    ymes = modeltrue(xmes, b0)
    ymes += np.random.randn(xmes.shape[0])*nslvl

    # DXtrain = xmes.max(axis=0) - xmes.min(axis=0)
    # lowD = DXtrain/20
    # highD = DXtrain*5
    # kernel = RBF([1.0]*2, 
    #             [(el1, el2) for el1,el2 in zip(lowD, highD)]) + \
    #     WhiteKernel(noise_level=0.001,
    #                 noise_level_bounds=(1e-6,np.var(ymes.ravel())*0.05))

    # gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=20)
    # gp.fit(xmes, ymes.ravel())

if casep == 1: ## CORRECTIONS MADE (MAY NO LONGER WORK)
    def ishigami(x,b):
        return np.atleast_2d(np.sin(x[:,0]) + b[:,0]*np.sin(x[:,1])**2 +  \
               b[:,1]*np.sin(x[:,0])*x[:,2]**4)

    Xtrain = sst.qmc.scale(sst.qmc.LatinHypercube(d=5).random(100),
                            [-3, -3, -3, 4, 0.01], [3, 3, 3, 9, 0.3])
    ytrain = ishigami(Xtrain[:,:3],Xtrain[:,3:])

    XX = np.array([[0,0,0], [-1.3,0,-1], [0.2,1,-1],
                [2,1,-2], [3.4,1,-1], [2.5,-2,-1],
                [3,3,-2], [-1,-1,2], [2.7,-2,0],
                [1,1,-2]])
    b0 = np.array([7,0.1])

    DXtrain = Xtrain.max(axis=0) - Xtrain.min(axis=0)
    lowD = DXtrain/20
    highD = DXtrain*5
    kernel = RBF([1.0]*5, 
                [(el1, el2) for el1,el2 in zip(lowD, highD)]) + \
        WhiteKernel(noise_level=0.001,
                    noise_level_bounds=(1e-6,np.var(ytrain)*0.05))

    gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=20)
    gp.fit(Xtrain, ytrain.ravel())

    def modelgp(x, b):
        return np.atleast_2d(gp.predict(np.hstack([x,
                    np.repeat(np.r_[[b]],x.shape[0], axis=0)])).ravel())

    ytrue = ishigami(XX,
                np.repeat(np.atleast_2d(b0),XX.shape[0], axis=0))
    ypred = modelgp(XX, b0)
    Q2emp = 1-np.var(ytrue-ypred)/np.var(ypred)
    ymes = ytrue
           
    if True:
        fig, ax = plt.subplots()
        ax.plot(XX[:,0], ytrue[0,:], 'or')
        ax.plot(XX[:,0], ypred[0,:], '.b')
        axx = ax.twiny()
        axx.plot(ytrue[0,:], ypred[0,:], '+k')
        axx.plot(ytrue[0,:], ytrue[0,:], '+-k')
        ax.set_title('Q2emp=' + \
                    str(np.round(Q2emp,3)))
    print('Q2emp =', Q2emp)

    xmes = XX

if casep == 2: ## CORRECTIONS MADE (MAY NO LONGER WORK)
    Nsamp = 50
    mu_true = np.array([1,2,4])
    cor_true = np.array([[1,     0.8,  -0.1],
                         [0.8,     1,  -0.5],
                         [-0.5, -0.1,    1]])
    s_true = np.array([0.2, 0.4, 0.5])
    cov_true = np.diag(s_true) @ cor_true @ np.diag(s_true)
    L_true = np.linalg.cholesky(cov_true)

    yrnd = mu_true + (L_true @ np.random.randn(mu_true.shape[0],Nsamp)).T

    def idfun(cond, par, N):
        return np.repeat(np.atleast_2d(par),N, axis=0)
    
############## NEW: scalar output, KNOWN AR(1) noise covariance
if casep == 3:
    def modelfit(x, b):
        # forward model returns (N, 1) -- enforced shape convention
        return (b[0] + b[1]*x[:,0] + b[2]*x[:,0]**2)[:, None]

    Nobs = 30
    xmes = np.linspace(0, 6, Nobs)[:, None]
    b0_true = np.array([2, -1, 2, 0])
    # b0_true = np.array([2.0, -1.0, 0.3])
    y_clean = modelfit(xmes, b0_true)

    # Build AR(1) covariance with known parameters (treated as fixed/known)
    rho_true = 0.7
    s2_true = 0.15**2
    idx_obs = np.arange(Nobs)
    Sigma_known = s2_true * rho_true ** np.abs(
        idx_obs[:, None] - idx_obs[None, :])
    L_known = np.linalg.cholesky(Sigma_known)
    ymes = y_clean + (L_known @ np.random.randn(Nobs))[:, None]

############## NEW: vector output (dy=2), KNOWN Kronecker noise
if casep == 4:
    def modelfit(x, b):
        # two outputs, both functions of beta. Returns (N, 2).
        a = b[0] + b[1]*x[:,0] + b[2]*x[:,0]**2
        c = b[0] - 0.5*b[1]*x[:,0]
        return np.column_stack([a, c])

    Nobs = 30
    xmes = np.linspace(0, 6, Nobs)[:, None]
    b0_true = np.array([2.0, -1.0, 0.3])
    y_clean = modelfit(xmes, b0_true)

    # Known per-observation covariance across the dy=2 components
    Sigma_d_known = np.array([[0.05, 0.03],
                              [0.03, 0.08]])
    Ld_known = np.linalg.cholesky(Sigma_d_known)
    ymes = y_clean + np.random.randn(Nobs, 2) @ Ld_known.T
    

#%%############################################################################
# DEFINITION OF BAYESIAN INFERENCE OBJECTS
###############################################################################

############## FULLY ANALYTICAL REGRESSION EXAMPLE
if casep == 0:
    Ndim = 3
    pl = -5
    ph = 5
    # sinvg = [0.2, -0.1, 2]
    # sexp = [0.4, 0.4, 0.05]
    # sdexp = 0.1
    sexp = [0.1, 0.1, 0.1]
    sdexp = 0.1
    covProp = np.eye(3)*1e-1
    LLTprop = np.linalg.cholesky(covProp)

    rndUs = [UnifVar([pl,ph]) for _ in range(3)]
    # rndUs = [NormVar([0, 1]) for _ in range(3)]
    # rnds = InvGaussVar(param=sinvg)
    rnds = HalfNormVar(param=0.5)
    obsvar = ObsVar(obs=np.c_[np.ravel(ymes)], prev_model=modelfit, cond_var=xmes)

    bstart = np.array([rndUs[i].draw() for i in range(3)] + \
                       [float(rnds.draw())])
    
############## ISHIGAMI AND GP MODEL SURROGATE FOR CALIBRATION
if casep == 1:
    Ndim = 2
    sinvg = [0.4, 0, 2]

    sexp = [0.2, 5e-2]
    sdexp = 0.2
    covProp = np.array([[0.2,0],[0,0.05]])
    LLTprop = np.linalg.cholesky(covProp)

    rndUs = [UnifVar([0,10]), UnifVar([0,1])]
    # rnds = InvGaussVar(param=sinvg)
    rnds = HalfNormVar(param=0.5)
    obsvar = ObsVar(obs=ymes, prev_model=modelgp, cond_var=XX)

    bstart = np.array([rndUs[i].draw() for i in range(2)] + \
                       [float(rnds.draw())])
    
############## STANDARD MULTIVARIATE PARAMETER ESTIMATION
if casep == 2:
    Ndim = 3
    sinvg = [0.2, 0.2, 1]
    sexp = [0.1, 0.1, 0.1]
    sdexp = 0.1
    rndUs = [UnifVar([0,6]), UnifVar([0,6]), UnifVar([0,6])]
    # rnds = InvGaussVar(param=sinvg)
    rnds = HalfNormVar(param=0.5)
    bstart = np.array([rndUs[i].draw() for i in range(3)] + \
                       [float(rnds.draw())])
    obsvar = ObsVar(obs=yrnd,
                prev_model=lambda v, x: idfun(v, x, yrnd.shape[0]),
                cond_var=[])

############## NEW: scalar output, AR(1) Sigma KNOWN -> KnownCovObs (full)
if casep == 3:
    Ndim = 3
    rndUs = [UnifVar([-5, 5]) for _ in range(3)]
    rnds = None     # discrObj=None signals fixed-covariance mode
    bstart = np.array([rndUs[i].draw() for i in range(3)])
    obsvar = KnownCovObs(obs=ymes, prev_model=modelfit, cond_var=xmes,
                         Sigma=Sigma_known, kind="full")

############## NEW: vector output, Sigma_d KNOWN -> KnownCovObs (kron)
if casep == 4:
    Ndim = 3
    rndUs = [UnifVar([-5, 5]) for _ in range(3)]
    rnds = None
    bstart = np.array([rndUs[i].draw() for i in range(3)])
    obsvar = KnownCovObs(obs=ymes, prev_model=modelfit, cond_var=xmes,
                         Sigma=Sigma_d_known, kind="kron")

#%%############################################################################
# INSTANTIATION AND RUN OF INFERENCE ALGORITHM
###############################################################################

NMCMC = 20000
Nburn = 10000
verbose = True

# inftype = 'MHwG'
# bstart = np.array([rndUs[i].draw() for i in range(3)] + \
#                     [float(rnds.draw())])

if inftype == 'MH':
    MCalgo = MHalgo(NMCMC, Nthin=20, Nburn=Nburn, is_adaptive=True,
                     verbose=verbose)
if inftype == 'MHwG':
    MCalgo = MHwGalgo(NMCMC, Nthin=20, Nburn=Nburn, is_adaptive=True,
                       verbose=verbose)
    
MCalgo.initialize(obsvar, rndUs, rnds)
MCalgo.MCchain[0] = bstart
MCalgo.state(0, set_state=True)
MCout, llout = MCalgo.runInference()


#%%############################################################################
# VISUALISATION OF INFERENCE RESULTS
###############################################################################

MCalgo.post_visupar()
MCalgo.hist_alldim()
MCalgo.diag_chain(0, show_prior=False)
if not MCalgo._fixed_cov:
    MCalgo.diag_chain(3, show_prior=False)

print(MCalgo)


#%%############################################################################
# VISUALISATION FOR PURE STATISTICAL INFERENCE
###############################################################################

if casep == 2:
    print(sst.multivariate_normal(
        MCalgo.MAP[:3], rnds.diagSmat(MCalgo.MAP[3], 3)).logpdf(yrnd).sum())
    print(obsvar.loglike(MCalgo.MAP[:3],rnds.diagSmat(MCalgo.MAP[3], 3)))
    ypost = sst.multivariate_normal(mean=MCalgo.MAP[:3],
                                    cov=MCalgo.discrObj.diagSmat(
                                        MCalgo.MAP[3], yrnd.shape[1])).rvs(100)

    fig, ax = plt.subplots()
    ax.scatter(ypost[:,0], ypost[:,1], marker='x', color='k')
    ax.scatter(yrnd[:,0], yrnd[:,1], marker='o', color='b')



#%%
    
# def modelfit1(x,b):
#     return np.atleast_2d(b[0] + b[1]*x[:,0] + b[2]*x[:,0]**2)[0]
# # def modelfit2(x, b):
# #     x = np.atleast_2d(x)
# #     b = np.asarray(b)
# #     return np.atleast_2d(b[0] + x @ b[1:])[0]
# def modelfit2(x, b):
#     if x.ndim == 1:
#         x = x[np.newaxis, :]
#     b = np.asarray(b)
#     y =  y = b[0] + x @ b[1:] 
#     return y
#     # return np.squeeze(y)

# ymes1 = modelfit1(xmes, b0)
# ymes2 = modelfit2(xmes, b0[:-1])

# print(ymes1)
# print(ymes2)

# postpar = MCalgo.cut_chain


# %%

# postY1 = np.array([[modelfit1(np.r_[[xx]], bb)[0] for bb in postpar[:,:-1]] \
#                         for xx in xmes])
# postY2 = np.array([[modelfit2(np.r_[[xx]], bb)[0] for bb in postpar[:,:-1]] \
#                         for xx in xmes])
# print(postY1.shape)
# print(postY2.shape)