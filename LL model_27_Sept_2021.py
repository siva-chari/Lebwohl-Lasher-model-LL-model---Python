# -*- coding: utf-8 -*-
"""
Created on Thu Sep  9 19:21:14 2021

@author: Nived
"""
import numpy as np
import matplotlib.pyplot as plt
from numba import jit


#initialising configuration


@jit
def initialize_config(lx,ly,lz):
    config= np.zeros((lx,ly,lz))
    
    for i in range(lx):
        for j in range(ly):
            for k in range(lz):
                config[i,j,k]=np.random.uniform(0,np.pi)
    return config


#def initialize_all_alligned(lx,ly,lz):
#    config=np.zeros((lx,ly.lz))
    
#    for i in range(lx):
#        for j in range(ly):
#            for k in range(lz):
#                config[i,j,k]= np.pi
#    return config


def initialize_all_one_rad(lx,ly,lz):
    config=np.ones((lx,ly,lz))
    return config


#energy of configuration


@jit
def energy_config(config,eps,lx,ly,lz):
    
    esum=0.0
    
    
    for i in range(lx):
        for j in range(ly):
            for k in range(lz):
                #angle between neighbouring molecules along x direction
                
                iup=config[(i+1)%lx,j,k] - config[i,j,k]
                idown = config[(i-1)%lx,j,k] - config[i,j,k]
                
                #finding cos^2 of angles and subtracting 1 as in summation
                
                t1= (3.0*(np.cos(iup))**2 - 1.0) + (3.0*(np.cos(idown))**2 - 1.0)
                
                
                #angle between neighbouring molecules along y direction
                
                jup = config[i,(j+1)%ly,k] - config[i,j,k]
                jdown = config[i, (j-1)%ly, k] - config[i,j,k]
                
                t2= (3.0*(np.cos(jup))**2 - 1.0) + (3.0*(np.cos(jdown))**2 - 1.0)
                
                #angle between neighbouring molecules along z direction
                
                kup = config[i, j, (k+1)%lz] - config[i,j,k]
                kdown = config[i, j, (k-1)%lz] - config[i,j,k]
                
                t3=(3.0*(np.cos(kup))**2 - 1.0) + (3.0*(np.cos(kdown))**2 - 1.0)
                
                #sum of terms
                nnsum = t1 + t2 + t3
                
                #energy= 1/2 * [3cos^2(theta)-1]
                
                esum = esum - eps * 0.5 * nnsum * 0.5
                
                
    return esum

@jit
def get_order_parameter(config,lx,ly,lz):
    p2sum=0.0
    
    for i in range(lx):
        for j in range(ly):
            for k in range(lz):
                
                p2cost=1.5*(np.cos(config[i,j,k]))**2 - 0.5
                
                p2sum=p2sum+p2cost
                
    return p2sum
                

def one_mcmove(config,eps,lx,ly,lz,temperature):
    i=np.random.randint(0,lx)
    j=np.random.randint(0,ly)
    k=np.random.randint(0,lz)
    
    #energy before move
    
    E0=energy_config(config,eps,lx,ly,lz)
    
    #making the move
    
    rot=np.random.uniform(0,np.pi/2.0)
    config[i,j,k] = config[i,j,k] + rot
    
    #energy after move
    
    E1=energy_config(config,eps,lx,ly,lz)
    
    dE= E1-E0
    kB=1.0
    beta=1.0/(kB*temperature)
    
    if (dE < 0):
        E0=E1
        energy=E1
    
    else:
        if(np.random.uniform()<np.exp(-beta*dE)):
            
            E0=E1
            energy=E1
        else:
            
            config[i,j,k]=config[i,j,k]-rot
            energy=E0
            
            
    return config

#making one mc sweep

@jit
def one_MC_sweep(config,eps,lx,ly,lz,temperature):
    N=lx*ly*lz
    
    for istep in range(N):
        config=one_mcmove(config,eps,lx,ly,lz,temperature)
        
        
    energy=energy_config(config,eps,lx,ly,lz)
    
    return [config,energy]

#To store the datas in file

def open_out_files():
    
    global LL_model_file
    global out_data
    
    LL_model_file="LL_model.dat"
    
    out_data=open(LL_model_file, 'w+')
    return

def close_all_files():
    global LL_model_file
    out_data.close()
    
    return

#============= MAIN PROGRAM ===============================#

[lx, ly, lz] = 4, 4, 4
Tinit = 0.1
Tfinal = 4.0
dT = 0.1
kB = 1.0
eps=1.0
   
N = lx*ly*lz 
      
nMCsweeps = 15000
nEquil_Sweeps = 10000
nTempSteps = int(np.abs((Tfinal - Tinit)/dT))


open_out_files()

config = np.zeros((lx, ly, lz))

#config=np.ones((lx,ly,lz))


for i in range(100):

    [config,E0]=one_MC_sweep(config,eps,lx,ly,lz,Tinit)



for istep in range(nTempSteps+1):
    Tcurr = Tinit + istep*dT
    beta = 1.0/(kB*Tcurr)
    
    
    eavg=0.0
    opavg=0.0
    
    
    for isweep in range(nMCsweeps):
        [config,E]=one_MC_sweep(config,eps,lx,ly,lz,Tcurr)
        
        if (isweep > nEquil_Sweeps):
            
            op = get_order_parameter(config,lx,ly,lz)
             
            eavg=eavg+E
            
            opavg=opavg + np.abs(op)
            
    avgOver = (nMCsweeps - nEquil_Sweeps)
    eavg = eavg/avgOver
    eavg_per_molecule=eavg/N*1.0
    opavg=opavg/avgOver
    opavg_per_molecule = opavg/N*1.0
    
    out_data.write("{:.8f} {:.8f} {:.8f}\n".format(Tcurr,eavg_per_molecule,opavg_per_molecule ))
    out_data.flush()
    
    
close_all_files()
                
                                
