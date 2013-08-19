import numpy as np
from scipy.optimize import *
import itertools as iter
import pylab as pl

def Kfilter00(num, y, A, mu0, Sigma0, Phi, cQ, cR, m_y, h):
    """
    # NOTE: must give cholesky decomp: cQ=chol(Q), cR=chol(R)
    # y is num by q  (time=row series=col)
    # A is a q by p matrix
    # R is q by q
    # mu0 is p by 1 
    # Sigma0, Phi, Q are p by p
    """
    Q = np.dot(np.transpose(cQ), cQ)
    R = np.dot(np.transpose(cR), cR)

    Phi = np.asmatrix(Phi)
    pdim = Phi.shape[0]    
    y = np.asmatrix(y)
    qdim = y.shape[1]
    xp = np.zeros((pdim, 1, num))
    Pp = np.zeros((pdim, pdim, num))
    xf = np.zeros((pdim, 1, num))
    Pf = np.zeros((pdim, pdim, num))
    innov = np.zeros((qdim, 1, num))
    sig = np.zeros((qdim, qdim, num))
    
    x00 = np.zeros((pdim, 1)) + mu0
    P00 = np.zeros((pdim, pdim)) + Sigma0
    xp[0, :, :] = np.dot(Phi, x00)
    Pp[0, :, :] = np.dot(Phi, np.dot(P00, np.transpose(Phi))) + Q
    sigtemp = np.dot(A[0, :], np.dot(Pp[0, :, :], A[0, :])) + R
    sig[0, :, :] = (np.transpose(sigtemp) + sigtemp) / 2
    siginv = np.linalg.inv(sig[0, :, :])
  
    K = np.dot(Pp[0, :, :], np.dot(A[0, :], siginv))
    innov[0, :, :] = y[1, ] - np.dot(A[0, :], xp[0, :, :]) - np.dot(h, np.transpose(m_y[0, :]))

    xf[0, :, :] = xp[0, :, :] + np.dot(K, innov[0, :, :])
    Pf[0, :, :] = Pp[0, :, :] - np.dot(K, np.dot(A[0, :], Pp[0, :, :]))
    sigmat = np.zeros((qdim, qdim)) + sig[0, :, :]
    # -log(likelihood)
    like = 0.5 * np.log(2 * np.pi) + 0.5 * np.log(np.linalg.det(sigmat)) + \
           0.5 * np.dot(np.transpose(innov[0, :, :]), np.dot(siginv, innov[0, :, :]))

    ########## start filter iterations ###################
    for i in range(1, num):
        xp[i, :, :] = np.dot(Phi, xf[i - 1, :, :])
        Pp[i, :, :] = np.dot(Phi, np.dot(Pf[i - 1, :, :], np.transpose(Phi))) + Q
        sigtemp = np.dot(A[i, :], np.dot(Pp[i, :, :], A[i, :])) + R
        sig[i, :, :] = (np.transpose(sigtemp) + sigtemp) / 2
        siginv = np.linalg.inv(sig[i, :, :])              
     
        ### Kalman Gain(Update)###
        K = np.dot(Pp[i, :, :], np.dot(A[i, :], siginv))
        innov[i, :, :] = y[i, ] - np.dot(A[i, :], xp[i, :, :]) - np.dot(h, np.transpose(m_y[i, :]))
        xf[i, :, :] = xp[i, :, :] + np.dot(K, innov[i, :, :])
        Pf[i, :, :] = Pp[i, :, :] - np.dot(K, np.dot(A[i, :], Pp[i, :, :]))
        sigmat = np.zeros((qdim, qdim)) + sig[i, :, :]
        like += 0.5 * np.log(2 * np.pi) + 0.5 * np.log(np.linalg.det(sigmat)) + \
                0.5 * np.dot(np.transpose(innov[i, :, :]), np.dot(siginv, innov[i, :, :]))
    
    return {'xp': xp, 'Pp' :Pp, 'xf': xf, 'Pf': Pf, 'like': like,
            'innov': innov, 'sig': sig, 'Kn': K}

def Kfilter0(num, y, A, mu0, Sigma0, Phi, cQ, cR, m_y, h):
    """
    # NOTE: must give cholesky decomp: cQ=chol(Q), cR=chol(R)
    # y is num by q  (time=row series=col)
    # A is a q by p matrix
    # R is q by q
    # mu0 is p by 1 
    # Sigma0, Phi, Q are p by p
    """
    Q = np.dot(np.transpose(cQ), cQ)
    R = np.dot(np.transpose(cR), cR)

    Phi = np.asmatrix(Phi)
    pdim = Phi.shape[0]
    y = np.asmatrix(y)
    qdim = y.shape[0]
    xp = np.zeros((num, pdim, 1))
    Pp = np.zeros((num, pdim, pdim))
    xf = np.zeros((num, pdim, 1))
    Pf = np.zeros((num, pdim, pdim))
    innov = np.zeros((num, qdim, 1))
    sig = np.zeros((num, qdim, qdim))
    
    x00 = np.zeros((pdim, 1)) + mu0
    P00 = np.zeros((pdim, pdim)) + Sigma0
    xp[0, :, :] = np.dot(Phi, x00)
    Pp[0, :, :] = np.dot(Phi, np.dot(P00, np.transpose(Phi))) + Q
    sigtemp = np.dot(A[0, :], np.dot(Pp[0, :, :], A[0, :])) + R
    sig[0, :, :] = (np.transpose(sigtemp) + sigtemp) / 2
    siginv = np.linalg.inv(sig[0, :, :])
    
    K = np.dot(Pp[0, :, :], np.dot(np.asmatrix(A[0, :]).T, siginv))
    innov[0, :, :] = y[0, 0] - np.dot(A[0, :], xp[0, :, :]) - np.dot(h, np.asmatrix(m_y[:, 0]).T)
    
    xf[0, :, :] = xp[0, :, :] + np.dot(K, innov[0, :, :])
    Pf[0, :, :] = Pp[0, :, :] - np.dot(np.dot(K, np.asmatrix(A[0, :])), Pp[0, :, :])
    sigmat = np.zeros((qdim, qdim)) + sig[0, :, :]

    # -log(likelihood)
    like = 0.5 * np.log(2 * np.pi) + 0.5 * np.log(np.linalg.det(sigmat)) + \
           0.5 * np.dot(np.transpose(innov[0, :, :]), np.dot(siginv, innov[0, :, :]))
    
    ########## start filter iterations ###################
    for i in range(1, num):
        xp[i, :, :] = np.dot(Phi, xf[i - 1, :, :])
        Pp[i, :, :] = np.dot(Phi, np.dot(Pf[i - 1, :, :], np.transpose(Phi))) + Q
        sigtemp = np.dot(A[i, :], np.dot(Pp[i, :, :], A[i, :])) + R
        sig[i, :, :] = (np.transpose(sigtemp) + sigtemp) / 2
        siginv = np.linalg.inv(sig[i, :, :])              
     
        ### Kalman Gain(Update)###
        K = np.dot(Pp[i, :, :], np.dot(np.asmatrix(A[i, :]).T, siginv))
        innov[i, :, :] = y[0, i] - np.dot(A[i, :], xp[i, :, :]) - np.dot(h, np.asmatrix(m_y[:, i]).T)
        xf[i, :, :] = xp[i, :, :] + np.dot(K, innov[i, :, :])
        Pf[i, :, :] = Pp[i, :, :] - np.dot(np.dot(K, np.asmatrix(A[i, :])), Pp[i, :, :])
        sigmat = np.zeros((qdim, qdim)) + sig[i, :, :]
        like = like + 0.5 * np.log(2 * np.pi) + 0.5 * np.log(np.linalg.det(sigmat)) + \
                      0.5 * np.dot(np.transpose(innov[i, :, :]), np.dot(siginv, innov[i, :, :]))
    
    return {'xp': xp, 'Pp' :Pp, 'xf': xf, 'Pf': Pf, 'like': like,
            'innov': innov, 'sig': sig, 'Kn': K}
    
def Ksmooth0(num, y, A, mu0, Sigma0, Phi, cQ, cR, m_y, h):
    """
    # Note: Q and R are given as Cholesky decomps
    #       cQ=chol(Q), cR=chol(R)
    #
    """
    kf = Kfilter0(num, y, A, mu0, Sigma0, Phi, cQ, cR, m_y, h)

    pdim = Phi.shape[0]
    xs = np.zeros((num, pdim, 1))
    Ps = np.zeros((num, pdim, pdim))
    
    J = np.zeros((num, pdim, pdim))
    xs[num - 1, :, :] = kf['xf'][num - 1, :, :]
    Ps[num - 1, :, :] = kf['Pf'][num - 1, :, :]
    
    "Smoothing"
    for k in reversed(range(1, num)):
        J[k - 1, :, :] = np.dot(kf['Pf'][k - 1, :, :],
                            np.dot(np.transpose(Phi), np.linalg.inv(kf['Pp'][k, :, :])))
        xs[k - 1, :, :] = kf['xf'][k - 1, :, :] + np.dot(J[k - 1, :, :], (xs[k, :, :]) - kf['xp'][k, :, :])
        Ps[k - 1, :, :] = kf['Pf'][k - 1, :, :] + \
                      np.dot(J[k - 1, :, :], np.dot(Ps[k, :, :] - kf['Pp'][k, :, :], np.transpose(J[k - 1, :, :])))

    x00 = mu0
    P00 = Sigma0
    J0 = np.zeros((pdim, pdim)) + np.dot(P00, np.dot(np.transpose(Phi), np.linalg.inv(kf['Pp'][0, :, :])))
    x0n = np.zeros((pdim, 1)) + x00 + np.dot(J0, xs[0, :, :] - kf['xp'][0, :, :])
    P0n = P00 + np.dot(J0, np.dot(Ps[k, :, :] - kf['Pp'][k, :, :], np.transpose(J0)))
    
    return {'xs': xs, 'Ps': Ps, 'x0n': x0n, 'P0n': P0n,
            'J0': J0, 'J': J, 'xp': kf['xp'], 'Pp': kf['Pp'],
            'xf': kf['xf'], 'Pf': kf['Pf'], 'like': kf['like'], 'Kn': kf['Kn']}

def Linn(num, y, Phi, mu0, Sigma0, m_y, h, para):
    try:
        para = list(iter.chain.from_iterable(para))
    except (TypeError):
        pass
    
    size = Phi.shape[0]
    A = np.zeros((num, size))  # Coefficient Vector in ObservationEquation 
    A[:, 0] = 1
    A[:, size - 1] = 1
  
    cQ1 = para[0] ** 2  # sqrt q11
    cQ2 = para[1] ** 2
    cQ = np.zeros((size, size))
    cQ[0, 0] = cQ1  # State Equation Error Component 
    cQ[size - 1, size - 1] = cQ2

    cR1 = para[2] ** 2
    cR = np.eye(1)
    cR[0, 0] = cR1  # sqrt r11 Observed Equation Error Component
    
    for i in range(m_y.shape[0]):
        h[0, i] = para[3 + i]
    
    kf = Kfilter0(num, y, A, mu0, Sigma0, Phi, cQ, cR, m_y, h)
    
    return(kf['like'])

class KalmanFilter(object):
    
    def __init__(self, num, y, mu0, Sigma0, Phi, m_y, h, initpar):
        self.num = num
        self.y = y
        self.mu0 = mu0
        self.Sigma0 = Sigma0
        self.Phi = Phi
        self.m_y = m_y
        self.h = h
        self.initpar = initpar

    def loglikelihood(self, para):
        loglikelihood = Linn(self.num, self.y, self.Phi,
                    self.mu0, self.Sigma0,
                    self.m_y, self.h, para)
        print(loglikelihood)
        return loglikelihood
        
    def optimize_params(self, method):
        return minimize(fun = self.loglikelihood,
                                x0 = self.initpar,
                                method = method)
        
if __name__ == "__main__":
    target_dir = "C:\Users\toan.nguyen\Google ドライブ\研究開発\時系列_状態空間\kalman\mykalman\"
    targets = np.loadtxt(target_dir + 'target.csv', delimiter = ',', unpack = True)
    chk = 0
    
    if chk == 0:
        y = targets[1, :]
        ncols = targets.shape[0]
        num = y.shape[0]
    
    one_vector = np.zeros((num, 1)) + 1
    if ncols >= 3:
        m_y = np.concatenate((targets[2:, ], one_vector.T))
    else:
        m_y = one_vector.T
    
    transm = np.loadtxt('TransM.csv', delimiter = ',', unpack = True)  
    Phi = transm.T       
    
    ######## Initial Parameters Setting ######### 
    size = Phi.shape[0]
    initpar = np.zeros(3 + m_y.shape[0]) + 0.1
    mu0 = np.zeros((size, 1)) + y[0]
    Sigma0 = np.eye(size) * 0.1
    h = np.zeros((1, m_y.shape[0]))
    
    kf = KalmanFilter(num, y, mu0, Sigma0, Phi, m_y, h, initpar)
    # est = kf.optimize_params('Powell')
    # print(est)
    """
    # default, CG, Newton-CG, BFGS:    error: ValueError: objects are not aligned
    
    # Nelder-Mead
    # est = [ -1.66371007e-04, 5.10463718e-08, 7.77654102e-02, -1.83669072e-02, 2.99860668e-01]
    status: 0
    nfev: 648
 success: True
     fun: -511.19699744659329
       x: array([ -1.66371007e-04,   5.10463718e-08,   7.77654102e-02,
        -1.83669072e-02,   2.99860668e-01])
 message: 'Optimization terminated successfully.'
     nit: 382
     
    # Powell: fast
    # est = [  6.24888471e-10,   8.61821954e-14,   5.35695417e-02, -1.49950795e-02,   3.96643123e-03]
     status: 0
 success: True
   direc: array([[ 1.,  0.,  0.,  0.,  0.],
       [ 0.,  1.,  0.,  0.,  0.],
       [ 0.,  0.,  1.,  0.,  0.],
       [ 0.,  0.,  0.,  1.,  0.],
       [ 0.,  0.,  0.,  0.,  1.]])
    nfev: 220
     fun: array(-554.1146586389351)
       x: array([  6.24888471e-10,   8.61821954e-14,   5.35695417e-02,
        -1.49950795e-02,   3.96643123e-03])
 message: 'Optimization terminated successfully.'
     nit: 3
     
    # Anneal: value is BIG, long runtime
    # est = [ -2.14967973,   5.37881934,  33.90043744, -40.01700198, -47.05939541]
      status: 0
 success: True
  accept: 699
    nfev: 4251
       T: array([[  5.87467845e-11]])
     fun: array([[-547.4354928]])
       x: array([[ -1.20232621e-03,  -1.01785655e-04,  -5.54236365e-02,
         -3.39332249e-02,   1.84722863e-01]])
 message: 'Points no longer changing'
     nit: 84
     
    
    
    # L-BFGS-B
    # est = [ 2.73620390e-06,  -2.44110270e-09,   5.35723161e-02, -1.50965118e-02,   9.95268403e-02]
    
    # TNC
    # est = [  7.16635700e-06,  -4.83307293e-09,   5.35689367e-02, -1.55180124e-02,   1.43285834e-01]
    
    # COBYLA: fast
    # est = [  2.13604390e-02,  -4.78535507e-05,   9.74439329e-02, 3.21175770e-01,   1.80490864e-01]
    
    # SLSQP: fast    
    # est = [  2.73312031e-08,  -6.34171048e-09,   5.35694953e-02, -1.49956432e-02,   3.98322031e-03]    
    """



    est = [-0.01190288, 0.00066779, 0.23152909, -0.01503034, 0.01186958]
    # smooth
    cQ1 = est[0] ** 2
    cQ2 = est[1] ** 2
    cQ = np.zeros((size, size))
    cQ[0, 0] = cQ1
    cQ[size - 1, size - 1] = cQ2
    
    cR1 = est[2] ** 2
    cR = np.zeros((1, 1)) + cR1
    
    A = np.zeros((num, size))
    A[:, 0] = 1
    A[:, size - 1] = 1
    for i in range(m_y.shape[0]):
        h[0, i] = est[3 + i]
    
    kf = Ksmooth0(num, y, A, mu0, Sigma0, Phi, cQ, cR, m_y, h)
        
    ex_tr = np.zeros((num, 1))
    for i in range(num):
        ex_tr[i, ] = np.dot(np.asmatrix(A[i, : ]), kf['xs'][i, :, : ])
    
    cycle = np.zeros((num, 1))
    A[:, size - 1] = 0
    for i in range(num):
        cycle[i, ] = np.dot(np.asmatrix(A[i, : ]), kf['xs'][i, :, : ])
    
    ex_var = np.zeros((num, 1))
    for i in range(num):
        ex_var[i, ] = np.dot(h, np.asmatrix(m_y[:, i]).T)
    
#    pl.figure()
#    lines_true = pl.plot(y, color = 'k')
#    lines_filt = pl.plot(ex_tr, color = 'r', linestyle = 'dashed')
#    lines_smooth = pl.plot(cycle, color = 'b', linestyle = 'dotted')
#    pl.legend((lines_true[0], lines_filt[0], lines_smooth[0]),
#           ('true', 'trend', 'cycle'),
#           loc = 'upper right'
#    )
#    pl.show()

    pl.figure()
#    lines_true = pl.plot(y, color = 'k')
    lines_trend = pl.plot(ex_tr, color = 'r', linestyle = 'dashed')
#    lines_cycle = pl.plot(cycle, color = 'b', linestyle = 'dotted')
    pl.show()
