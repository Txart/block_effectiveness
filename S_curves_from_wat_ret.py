#%%
# -*- coding: utf-8 -*-

"""
Created on Tue Apr 19 08:47:45 2022
@author: alauren
"""
# --------    RUN S ----------------------------

import numpy as np
import matplotlib.pylab as plt
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit
from scipy.interpolate import InterpolatedUnivariateSpline as interS


#%% Peat hydrol properties from Susi
def peat_hydrol_properties(x, unit='g/cm3', var='bd', ptype='A'):
    """
    Peat water retention and saturated hydraulic conductivity as a function of bulk density
    Päivänen 1973. Hydraulic conductivity and water retention in peat soils. Acta forestalia fennica 129.
    see bulk density: page 48, fig 19; degree of humification: page 51 fig 21
    Hydraulic conductivity (cm/s) as a function of bulk density(g/cm3), page 18, as a function of degree of humification see page 51 
    input:
        - x peat inputvariable in: db, bulk density or dgree of humification (von Post)  as array \n
        - bulk density unit 'g/cm3' or 'kg/m3' \n
        - var 'db' if input variable is as bulk density, 'H' if as degree of humification (von Post) \n
        - ptype peat type: 'A': all, 'S': sphagnum, 'C': Carex, 'L': wood, list with length of x 
    output: (ThetaS and ThetaR in m3 m-3)
        van Genuchten water retention parameters as array [ThetaS, ThetaR, alpha, n] \n
        hydraulic conductivity (m/s)
    """
    #paras is dict variable, parameter estimates are stored in tuples, the model is water content = a0 + a1x + a2x2, where x is
    para={}                                                                     #'bd':bulk density in g/ cm3; 'H': von Post degree of humification
    para['bd'] ={'pF0':(97.95, -79.72, 0.0), 'pF1.5':(20.83, 759.69, -2484.3),
            'pF2': (3.81, 705.13, -2036.2), 'pF3':(9.37, 241.69, -364.6),
            'pF4':(-0.06, 249.8, -519.9), 'pF4.2':(0.0, 174.48, -348.9)}
    para['H'] ={'pF0':(95.17, -1.26, 0.0), 'pF1.5':(46.20, 8.32, -0.54),
            'pF2': (27.03, 8.14, -0.43), 'pF3':(17.59, 3.22, -0.07),
            'pF4':(8.81, 3.03, -0.10), 'pF4.2':(5.8, 2.27, -0.08)}
    
    intp_pF1={}                                                                 # interpolation functions for pF1        
    intp_pF1['bd'] = interp1d([0.04,0.08,0.1,0.2],[63.,84.,86.,80.],fill_value='extrapolate')
    intp_pF1['H'] = interp1d([1.,4.,6.,10.],[75.,84.,86.,80.],fill_value='extrapolate')
    
    #Saturatated hydraulic conductivity parameters
    Kpara ={'bd':{'A':(-2.271, -9.80), 'S':(-2.321, -13.22), 'C':(-1.921, -10.702), 'L':(-1.921, -10.702)}, 
            'H':{'A':(-2.261, -0.205), 'S':(-2.471, -0.253), 'C':(-1.850, -0.278), 'L':(-2.399, -0.124)}}
    
    vg_ini=(0.88,	0.09, 0.03, 1.3)                                              # initial van Genuchten parameters (porosity, residual water content, alfa, n)

    x = np.array(x)
    prs = para[var]; pF1=intp_pF1[var]
    if unit=='kg/m3'and var=='db': x=x/1000.
    if  np.shape(x)[0] >1 and len(ptype)==1:
        ptype=np.repeat(ptype, np.shape(x)[0])        
    vgen = np.zeros((np.size(x),4))
    Ksat = np.zeros((np.size(x)))
    
    #wcont = lambda x, (a0, a1, a2): a0 + a1*x + a2*x**2.
    wcont = lambda x, *a: a[0] + a[1]*x + a[2]*x**2.
    van_g = lambda pot, *p:   p[1] + (p[0] - p[1]) / (1. + (p[2] * pot) **p[3]) **(1. - 1. / p[3])   
    #K = lambda x, (a0, a1): 10.**(a0 + a1*x) / 100.   # to m/s   
    K = lambda x, *a: 10.**(a[0] + a[1]*x) / 100.   # to m/s   
    
    potentials =np.array([0.01, 10.,32., 100.,1000.,10000.,15000. ])
    
    wc = (np.array([wcont(x,*prs['pF0']), pF1(x), wcont(x,*prs['pF1.5']), wcont(x,*prs['pF2']),
               wcont(x,*prs['pF3']), wcont(x,*prs['pF4']),wcont(x,*prs['pF4.2'])]))/100.
        
    for i,s in enumerate(np.transpose(wc)):
        vgen[i],_= curve_fit(van_g,potentials,s, p0=vg_ini)                      # van Genuchten parameters
        
    for i, a, pt in zip(range(len(x)), x, ptype):
        Ksat[i] = K(a, *Kpara[var][pt])                                          # hydraulic conductivity (cm/s -> m/s) 
    
    return vgen, Ksat

def CWTr(nLyrs, z, dz, pF, Ksat, direction='positive'):
    """
    Returns interpolation functions 
        sto=f(gwl)  profile water storage as a function ofground water level
        gwl=f(sto)  ground water level
        tra=f(gwl)  transissivity
    Input:
        nLyrs number of soil layers
        d depth of layer midpoint
        dz layer thickness
        pF van Genuchten water retention parameters: ThetaS, ThetaR, alfa, n
        Ksat saturated hydraulic conductivity in m s-1
        direction: positive or negative downwards
    """    
    #-------Parameters ---------------------
    z = np.array(z)   
    dz =np.array(dz)
    nroot = 8   # 8 good number of layers in rooting zone
    #nroot =10      #for jääli simulations
    nroot2 = 3  #2 10 cm root layers for air-filled porosity

    #--------- Connection between gwl and water storage------------
    d = 6 if direction == 'positive' else -6   
    gwl=np.linspace(0,d,150)
    if direction == 'positive':
        sto = [sum(wrc(pF, x = np.minimum(z-g, 0.0))*dz) for g in gwl]     #equilibrium head m
        storoot = [np.sum(wrc(pF, x = np.minimum(z-g, 0.0))[0:nroot]*dz[0:nroot]) for g in gwl]
        storoot2 = [np.sum(wrc(pF, x = np.minimum(z-g, 0.0))[0:nroot2]*dz[0:nroot2]) for g in gwl]
    else:
        sto = [sum(wrc(pF, x = np.minimum(z+g, 0.0))*dz) for g in gwl]     #equilibrium head m
        storoot = [np.sum(wrc(pF, x = np.minimum(z+g, 0.0))[0:nroot]*dz[0:nroot]) for g in gwl]
        storoot2 = [np.sum(wrc(pF, x = np.minimum(z+g, 0.0))[0:nroot2]*dz[0:nroot2]) for g in gwl]

    gwlToSto = interp1d(np.array(gwl), np.array(sto), fill_value='extrapolate')
    airtot = sto[0]-sto                                                         #m air in the profile
    airroot = storoot[0]-storoot                                                #m air in the rooting zone
    afproot = (storoot2[0]-storoot2)/(sum(dz[:nroot2]))                         #air-filled porosity in root layer
    ratio = airroot[1:]/airtot[1:]                                            #share of air-filled porosity in rooting zone to total air volume
    sto = list(sto); gwl= list(gwl); ratio=list(ratio); afproot = list(afproot)         
    sto.reverse(); gwl.reverse(); ratio.reverse(); afproot.reverse()
    stoToGwl =interp1d(np.array(sto), np.array(gwl), fill_value='extrapolate')
    gwlToRatio = interp1d(np.array(gwl[1:]), np.array(ratio), fill_value='extrapolate' )
    gwlToAfp= interp1d(np.array(gwl), np.array(afproot), fill_value='extrapolate' )
    C = interp1d(np.array(gwl), np.array(np.gradient(gwlToSto(gwl))/np.gradient(gwl)), fill_value='extrapolate')  #storage coefficient function      
    
    del gwl, sto, ratio, afproot
        
    #----------Transmissivity-------------------
    K=np.array(Ksat*86400.)   #from m/s to m/day
    tr =[sum(K[t:]*dz[t:]) for t in range(nLyrs)]        
    if direction=='positive':        
        gwlToTra = interS(z, np.array(tr))            
    else:
        z= list(z);  z.reverse(); tr.reverse()
        gwlToTra = interS(-np.array(z), np.array(tr))                    
    del tr
    return gwlToSto, stoToGwl, gwlToTra, C, gwlToRatio, gwlToAfp

def wrc(pF, x=None, var=None):
    """
    vanGenuchten-Mualem soil water retention curve\n
    IN:
        pF - dict['ThetaS': ,'ThetaR': ,'alpha':, 'n':,] OR
           - list [ThetaS, ThetaR, alpha, n]
        x  - soil water tension [m H2O = 0.1 kPa]
           - volumetric water content [vol/vol]
        var-'Th' is x=vol. wat. cont.
    OUT:
        res - Theta(Psii) or Psii(Theta)
    NOTE:\n
        sole input 'pF' draws water retention curve and returns 'None'. For drawing give only one pF-parameter set. 
        if several pF-curves are given, x can be scalar or len(x)=len(pF). In former case var is pF(x), in latter var[i]=pf[i,x[i]]
               
    Samuli Launiainen, Luke 2/2016
    """
    if type(pF) is dict: #dict input
        #Ts, Tr, alfa, n =pF['ThetaS'], pF['ThetaR'], pF['alpha'], pF['n']
        Ts=np.array(pF['ThetaS'].values()); Tr=np.array( pF['ThetaR'].values()); alfa=np.array( pF['alpha'].values()); n=np.array( pF['n'].values())
        m= 1.0 -np.divide(1.0,n)
    elif type(pF) is list: #list input
        pF=np.array(pF, ndmin=1) #ndmin=1 needed for indexing to work for 0-dim arrays
        Ts=pF[0]; Tr=pF[1]; alfa=pF[2]; n=pF[3] 
        m=1.0 - np.divide(1.0,n)
    elif type(pF) is np.ndarray:
        Ts, Tr, alfa, n = pF.T[0], pF.T[1], pF.T[2], pF.T[3]
        m=1.0 - np.divide(1.0,n)
    else:
        print ('Unknown type in pF')
        
    def theta_psi(x): #'Theta-->Psi'
        x=np.minimum(x,Ts) 
        x=np.maximum(x,Tr) #checks limits
        s= ((Ts - Tr) / (x - Tr))#**(1/m)
        Psi=-1e-2/ alfa*(s**(1/m)-1)**(1/n) # in m
        return Psi
        
    def psi_theta(x): # 'Psi-->Theta'
        x=100*np.minimum(x,0) #cm
        Th = Tr + (Ts-Tr)/(1+abs(alfa*x)**n)**m
        return Th           
 
    if var is 'Th': y=theta_psi(x) #'Theta-->Psi'           
    else: y=psi_theta(x) # 'Psi-->Theta'          
    return y

#%%
class StripHydrology():
    def __init__(self, spara):
        self.nLyrs = spara['nLyrs']                                                 # number of soil layers
        dz = np.ones(self.nLyrs)*spara['dzLyr']                                     # thickness of layers, m
        z = np.cumsum(dz)-dz/2.                                                # depth of the layer center point, m
        self.spara = spara
        if spara['vonP']:
            lenvp=len(spara['vonP top'])   
            vonP = np.ones(self.nLyrs)*spara['vonP bottom']
            vonP[0:lenvp] = spara['vonP top']                                      # degree of  decomposition, von Post scale
            ptype = spara['peat type bottom']*spara['nLyrs']
            lenpt = len(spara['peat type']); ptype[0:lenpt] = spara['peat type']   
            self.pF, self.Ksat = peat_hydrol_properties(vonP, var='H', ptype=ptype) # peat hydraulic properties after Päivänen 1973   
        else:
            lenbd=len(spara['bd top'])   
            bd = np.ones(self.nLyrs)*spara['bd bottom']
            bd[0:lenbd] = spara['bd top']                                      # degree of  decomposition, von Post scale
            ptype = spara['peat type bottom']*spara['nLyrs']
            lenpt = len(spara['peat type']); ptype[0:lenpt] = spara['peat type']   
            self.pF, self.Ksat = peat_hydrol_properties(bd, var='bd', ptype=ptype) # peat hydraulic properties after Päivänen 1973   

        for n in range(self.nLyrs):
            if z[n] < 0.41:
                self.Ksat[n]= self.Ksat[n]*spara['anisotropy']
            else:
                self.Ksat[n]= self.Ksat[n]*1.

        self.dwtToSto, self.stoToGwl, self.dwtToTra, self.C, self.dwtToRat, self.dwtToAfp = CWTr(self.nLyrs, z, dz, self.pF,
                                                               self.Ksat, direction='negative') # interpolated storage, transmissivity, diff water capacity, and ratio between aifilled porosoty in rooting zone to total airf porosity  functions

        self.L= spara['L']                                                     # compartemnt width, m
        self.n= spara['n']                                                     # number of computatin nodes
        self.dy = float(self.L/self.n)                                         # node width 
        sl= spara['slope']                                                     # slope %
        lev = 1.                                                               # basic level of soil surface
       
        self.ele = np.linspace(0,self.L*sl/100., self.n) + lev                 # surface rise in y direction, m
        self.dt = 1                                                             # time step, days
        self.implic = 1.# 0.5                                                  # 0-forward Euler, 1-backward Euler, 0.5-Crank-Nicolson
        self.DrIrr = False
        self.dwt = spara['initial h']                                          # h in the compartment
        self.H = (self.ele + self.dwt)  #np.empty(self.n)                      # depth to water table, negative down m
        self.Kmap = np.tile(self.Ksat, (self.n,1))                             # Ksat map (col, lyr)
        self.residence_time = np.zeros(self.n)                                 # residence time from column to the ditch, days

        print ('Peat strip initialized')

#%% params

spara = {
    'develop_scens': {
        'species': 'Pine', 'sfc': 3, 'sfc_specification': 1,
        'hdom': 20, 'vol': 150, 'age': 15, 'smc': 'Peatland',
        'nLyrs': 100, 'dzLyr': 0.05, 'L': 40, 'n': 20,
        # nLyrs kerrosten lkm, dzLyr kerroksen paksuus m, saran levys m, n laskentasolmulen lukumäärä, ditch depth pjan syvyys simuloinnin alussa m
        'ditch depth west': [-0.5],
        'ditch depth east': [-0.5],
        # ojan syvyys 20 vuotta simuloinnin aloituksesta
        'ditch depth 20y west': [-0.5],
        # ojan syvyys 20 vuotta simuloinnin aloituksesta
        'ditch depth 20y east': [-0.5],
        'scenario name': ['Control'],  # kasvunlisaykset
        'initial h': -0.2, 'slope': 0.0,
        'peat type': ['A', 'A'],
        'peat type bottom': ['A'], 'anisotropy': 10.,
        'vonP': True,
        'vonP top':  [4, 4],
        'vonP bottom': 4,
        'bd top': None, 'bd bottom': 0.16,
        'peatN': 1, 'peatP': 1, 'peatK': 1,
        'enable_peattop': True, 'enable_peatmiddle': True,
        'enable_peatbottom': True,
        'depoN': 4.0, 'depoP': 0.1, 'depoK': 1.0,
        'fertilization': {
            'application year': 2201,
            # fertilization dose in kg ha-1, decay_k in yr-1
            'N': {'dose': 0.0, 'decay_k': 0.5, 'eff': 1.0},
            'P': {'dose': 45.0, 'decay_k': 0.2, 'eff': 1.0},
            'K': {'dose': 100.0, 'decay_k': 0.3, 'eff': 1.0},
            'pH_increment': 1.0},
        'canopylayers': {'dominant': np.ones(20, dtype=int),

                         'subdominant': np.zeros(20, dtype=int),

                         'under': np.zeros(20, dtype=int)}
    }}
#%% Compute S

wt = np.arange(0, 5, 0.01) * -1
stp4 = StripHydrology(spara['develop_scens'])
storage = stp4.dwtToSto(wt) # total metres of water in soil column
S4 = np.gradient(storage) / np.gradient(wt) # From Boussinesq: S = [change in water input]/[change in wtd]

spara['develop_scens']['vonP top'] = [6, 6]
spara['develop_scens']['vonP bottom'] = 6
stp6 = StripHydrology(spara['develop_scens'])
storage = stp6.dwtToSto(wt)
S6 = np.gradient(storage) / np.gradient(wt)

spara['develop_scens']['vonP top'] = [8, 8]
spara['develop_scens']['vonP bottom'] = 8
stp8 = StripHydrology(spara['develop_scens'])
storage = stp8.dwtToSto(wt)
S8 = np.gradient(storage) / np.gradient(wt) 

#%% Plot
fig = plt.figure(num='s', figsize=(15, 8))
plt.plot(S4, wt, 'b-', label='S4')
plt.plot(S6, wt, 'r-', label='S6')
plt.plot(S8, wt, 'g-', label='S8')
plt.legend()

# %% My S curve, S EXPONENTIAL

def S_exp(zeta, s1, s2):
    # in terms of zeta = h - dem
    return s1*np.exp(s2*zeta)

# Exponential regression
# S = s1*e**(s2*wt)
# So ln(S) = ln(s1) + s2*wt
sto_curve = S8
s2, log_s1 = np.polyfit(x=wt, y=np.log(sto_curve), deg=1)
s1 = np.exp(log_s1)

plt.figure()
plt.plot(S_exp(wt, s1, s2), wt, label='parameterized S')
plt.plot(sto_curve, wt, '.', label='from data')
plt.ylabel('wt(m)')
plt.xlabel('storage coeff ')
plt.legend()


# %% My S curve, S LINEAR

def S_linear(zeta, s1, s2):
    return s1 - s2*zeta

# Linear regression
# S = s1 - s2 * wt
sto_curve = S8
s2_lin, s1_lin = np.polyfit(x=wt, y=sto_curve, deg=1)
s2_lin = -s2_lin

plt.figure()
plt.plot(S_linear(wt, s1_lin, s2_lin), wt, label='parameterized S')
plt.plot(sto_curve, wt, '.', label='from data')
plt.ylabel('wt(m)')
plt.xlabel('storage coeff ')
plt.legend()


# %%
