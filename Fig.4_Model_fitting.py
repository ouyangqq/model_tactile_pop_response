# -*- coding: utf-8 -*-
"""
Created on Sun Apr  7 12:49:24 2019

@author: qiangqiang ouyang
"""
import ultils as alt
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import Receptors as rslib
import simset as mysim 

tsensors=[]  
Ttype_buf=["SA1","RA1","PC"]
for ch in range(len(Ttype_buf)):
    tsensor=rslib.tactile_receptors(Ttype=Ttype_buf[ch])
    tsensors.append(tsensor) 
    
    
def func(x, a, b, c):
    return a * np.exp(-b * x) + c


def fr(vs,kf,frmax):
    tmp=kf*vs/0.015
    res=tmp*(tmp<frmax)+(tmp>=frmax)*frmax
    return res
    

def Th_func(f,KN,Kb1,Kb2,Ku,fB,fL,Q):
    S=1j*2*np.pi*f
    WB=2*np.pi*fB
    WL=2*np.pi*fL
    summ=Kb1*S+Kb2*S**2 
    return 1e6*0.015/np.abs(summ*((WB/Q/(S**2+WB*S/Q+WB**2))**(1))+Ku*WL/(S+WL))/KN

def single_PC_func(f,KN,Kb2,fB,Q):
    S=1j*2*np.pi*f
    WB=2*np.pi*fB
    summ=Kb2*S**2 
    return 1e6*0.015/np.abs(summ*(1/(S**2+WB*S/Q+WB**2))*(WB/(S+WB)))/KN

def single_RA1_func(f,KN,Kb1,Kb2,fB,Q):
    S=1j*2*np.pi*f
    WB=2*np.pi*fB
    summ=WB/Q*Kb1*S+Kb2*S**2 
    return 1e6*0.015/np.abs(summ*(1/(S**2+WB*S/Q+WB**2))*(WB/(S+WB)))/KN

def single_SA1_func(f,KN,Kb1,Kb2,Ku,fB,fL,Q):
    S=1j*2*np.pi*f
    WB=2*np.pi*fB
    WL=2*np.pi*fL
    summ=WB/Q*Kb1*S+Kb2*S**2 
    return 1e6*0.015/np.abs(summ*((1/(S**2+WB*S/Q+WB**2))*(WB/(S+WB)))+Ku*WL/(S+WL))/KN


Th_data_Set=[np.loadtxt('data/txtdata/fitting_NTH_SA1.txt'),
             np.loadtxt('data/txtdata/fitting_NTH_RA1.txt'),
             np.loadtxt('data/txtdata/fitting_NTH_PC.txt')]


maxfr=[tsensors[0].maxfr,tsensors[1].maxfr,tsensors[2].maxfr]
f=[20,50,100,300,600]
conditions=np.vstack([f[0]*np.ones([15,1]),f[1]*np.ones([15,1]),f[2]*np.ones([15,1]),f[3]*np.ones([15,1]),f[4]*np.ones([15,1])])
ob_fr_data=np.load('data/ob_vibro_firing_rate.npy')
data_s_SA1=np.hstack([conditions,ob_fr_data[0][0]])
data_s_RA1=np.hstack([conditions,ob_fr_data[0][1]])
data_s_PC=np.hstack([conditions,ob_fr_data[0][2]])


obfir=[data_s_SA1,data_s_RA1,data_s_PC]


    
svobfir=[]
sfreq=[50,100,300]
for ch in range(3):
    res=obfir[ch][obfir[ch][:,0]==sfreq[ch],1:3]
    svobfir.append(res)
    
           
rate_to_dm_dt=[np.array(svobfir[0]),np.array(svobfir[1]),np.array(svobfir[2])]


single_paras=[]
paras_dict=[]
def single_paras_fitting(ch):
    lowbounds=[[0,0,0,0,0,0,0],
               [0,0,0,0,0],
               [0,0,0,0]]
    '''
    upbounds=[[20000,10,10,10,1000,100,1],
              [500,10,10,1000,1],
              [40000,10,1000,1]]
    '''
    upbounds=[[10000,10,10,10,1000,100,1],
              [10000,10,10,1000,1],
              [1000000,10,1000,1]]
    xdata = Th_data_Set[ch][:,0]
    ydata = Th_data_Set[ch][:,1]

    plt.scatter(xdata, ydata,color='gray', marker='+') 
    x=np.arange(np.min(xdata),np.max(xdata),0.1)

    if(ch==0):
        popt, pcov = curve_fit(single_SA1_func, xdata, ydata, bounds=(lowbounds[ch], upbounds[ch]))
        observed=ydata
        predicted=single_SA1_func(xdata, *popt)
        R2="{0:.3f}".format(alt.R2(observed,predicted))
        plt.plot(x, single_SA1_func(x, *popt),mysim.colors[ch],linewidth=1.5,
        label=' KN=%5.3f\n Kb1=%5.3f\n Kb2=%5.3e\n Ku=%5.3e\n fb=%5.3f\n fl=%5.3f\n Q=%5.3f\n' % tuple(popt)+'\n $\mathrm{R}^{2}$='+str(R2))   
        single_paras.append(popt)
        paras_dict_SA1 = {'Kb1':popt[1], 'Kb2':popt[2],  'Ku': popt[3],'fb': popt[4],'fl': popt[5], 'Q': popt[6]} 
        paras_dict.append(paras_dict_SA1)
    
    if(ch==1):
        popt, pcov = curve_fit(single_RA1_func, xdata, ydata, bounds=(lowbounds[ch], upbounds[ch]))
        observed=ydata
        predicted=single_RA1_func(xdata, *popt)
        R2="{0:.3f}".format(alt.R2(observed,predicted))
        plt.plot(x, single_RA1_func(x, *popt),mysim.colors[ch],linewidth=1.5,
        label=' KN=%5.3f\n Kb1=%5.3f\n Kb2=%5.3e\n fb=%5.3f\n Q=%5.3f\n' % tuple(popt)+'\n $\mathrm{R}^{2}$='+str(R2)) 
        single_paras.append(popt)
        paras_dict_RA1 = {'Kb1':popt[1],'Kb2':popt[2], 'fb': popt[3], 'Q': popt[4]} 
        paras_dict.append(paras_dict_RA1)
    elif(ch==2):
        popt, pcov = curve_fit(single_PC_func, xdata, ydata, bounds=(lowbounds[ch], upbounds[ch]))
        observed=ydata
        predicted=single_PC_func(xdata, *popt)
        R2="{0:.3f}".format(alt.R2(observed,predicted))
        plt.plot(x, single_PC_func(x, *popt),mysim.colors[ch],linewidth=1.5,
        label=' KN=%5.3e\n Kb2=%5.3f\n fb=%5.3e\n Q=%5.3f\n' % tuple(popt)+'\n $\mathrm{R}^{2}$='+str(R2)) 
        single_paras.append(popt)
        paras_dict_PC= {'Kb2':popt[1], 'fb': popt[2], 'Q': popt[3]} 
        paras_dict.append(paras_dict_PC)
     
    plt.xscale('log')
    plt.yscale('log')
    
    plt.xticks([1,10,100,1000],fontsize=8) 
    plt.yticks([0.01,0.1,1,10,100,1000],fontsize=8)  
    
    plt.xlabel('Frequency [Hz]')
    if(ch==0):plt.ylabel('Depth [um]')
    plt.legend(loc=3,ncol=1,fontsize=8)
    np.save('data/single_fitting_paras.npy',single_paras)

rp=0.4
Densitys=[72.2,143.5,24.8]
std_SA1=np.sqrt((100/Densitys[0]-np.pi*(rp)**2)*np.pi*(rp)**2/(100/Densitys[0])**2)
std_RA1=np.sqrt((100/Densitys[1]-np.pi*(rp)**2)*np.pi*(rp)**2/(100/Densitys[1])**2)
std_PC=np.sqrt((100/Densitys[2]-np.pi*(rp)**2)*np.pi*(rp)**2/(100/Densitys[2])**2)



def Pop_SA1_func(x,Cs,Cd,Ce,RN,Kf):
    Kn=single_paras[0][0]
    
    tmp=fr(0.25*Cd*RN*(Cs+Ce*std_SA1)*x*(1/np.pi)/single_SA1_func(sfreq[0],*single_paras[0])/Kn*0.015,Kf,tsensors[0].maxfr)
    return tmp
    
def Pop_RA1_func(x,Cs,Cd,Ce,RN,Kf,w):
    Kn=single_paras[1][0]
    tmp=fr(0.25*Cd*RN*(Cs+Ce*std_RA1)*x*((1+w)/np.pi)/single_RA1_func(sfreq[1],*single_paras[1])/Kn*0.015,Kf,tsensors[1].maxfr)
    return tmp

def Pop_PC_func(x,Cs,Cd,Ce,RN,Kf,w):
    Kn=single_paras[2][0]
    tmp=fr(0.25*Cd*RN*(Cs+Ce*std_PC)*x*((1+w)/np.pi)/single_PC_func(sfreq[2],*single_paras[2])/Kn*0.015,Kf,tsensors[2].maxfr)
    return tmp


pop_paras=[]
def pop_paras_fitting(ch):

    lowbounds=[[0,0,0,0,0],
               [0,0,0,0,0,0],
               [0,0,0,0,0,0]]
    
    '''
    upbounds=[[5000,2,5000,100,100],
              [500,2,100,100,100,0.2],
              [2000,2,1000,100,100,1]]
    '''
    
    upbounds=[[1000,2,1000,100,100],
              [5000,2,100,100,100,1],
              [5000,2,100,100,100,1]]
    #np.load('data/ob_vibro_firing_rate.npy')
    dataset=rate_to_dm_dt[ch]
    xdata = dataset[:,0]
    ydata = dataset[:,1]
    
    
    plt.scatter(xdata, ydata,color='gray', marker='+')#, label='Original data')
    x=np.arange(np.min(xdata),np.max(xdata),1)
    if(ch==0):
        popt, pcov = curve_fit(Pop_SA1_func, xdata, ydata, bounds=(lowbounds[ch], upbounds[ch]))
        observed=ydata
        predicted=Pop_SA1_func(xdata, *popt)      
        R2="{0:.3f}".format(alt.R2(observed,predicted))
        plt.plot(x, Pop_SA1_func(x, *popt),mysim.colors[ch],linewidth=1.5,
        label='Cs=%5.3f\n Cd=%5.3f\n Ce=%5.3f\n Rc=%5.3f\n 1/τ=%5.3f' % tuple(popt)+'\n\n $\mathrm{R}^{2}$='+str(R2))   
        pop_paras.append(popt)
        paras_dict[0]['Cs'],paras_dict[0]['Cd'],paras_dict[0]['Ce'],paras_dict[0]['Rc'],paras_dict[0]['Kf']=popt[0],popt[1],popt[2],popt[3],popt[4]
    if(ch==1):
        popt, pcov = curve_fit(Pop_RA1_func, xdata, ydata, bounds=(lowbounds[ch], upbounds[ch]))
        observed=ydata
        predicted=Pop_RA1_func(xdata, *popt)
        R2="{0:.3f}".format(alt.R2(observed,predicted))
        plt.plot(x, Pop_RA1_func(x, *popt),mysim.colors[ch],linewidth=1.5,
        label='Cs=%5.3f\n  Cd=%5.3f\n Ce=%5.3f\n Rc=%5.3f\n 1/τ=%5.3f\n w=%5.3f'  % tuple(popt)+'\n\n $\mathrm{R}^{2}$='+str(R2)) 
        pop_paras.append(popt)
        paras_dict[1]['Cs'],paras_dict[1]['Cd'],paras_dict[1]['Ce'],paras_dict[1]['Rc'],paras_dict[1]['Kf'],paras_dict[1]['w']=popt[0],popt[1],popt[2],popt[3],popt[4],popt[5]
    elif(ch==2):
        popt, pcov = curve_fit(Pop_PC_func, xdata, ydata, bounds=(lowbounds[ch], upbounds[ch]))
        observed=ydata
        predicted=Pop_PC_func(xdata, *popt)
        R2="{0:.3f}".format(alt.R2(observed,predicted))
        plt.plot(x, Pop_PC_func(x, *popt),mysim.colors[ch],linewidth=1.5,
        label='Cs=%5.3f\n Cd=%5.3f\n Ce=%5.3f\n Rc=%5.3f\n 1/τ=%5.3f\n w=%5.3f'  % tuple(popt)+'\n\n $\mathrm{R}^{2}$='+str(R2)) 
        pop_paras.append(popt)
        paras_dict[2]['Cs'],paras_dict[2]['Cd'],paras_dict[2]['Ce'],paras_dict[2]['Rc'],paras_dict[2]['Kf'],paras_dict[2]['w']=popt[0],popt[1],popt[2],popt[3],popt[4],popt[5]
 

    plt.xscale('log')
    plt.xticks([10,100,1000],fontsize=8) 
    plt.xlabel('Depth [um]')
    if(ch==0):plt.ylabel('Firing rate [Spikes/s]')
    plt.legend(ncol=1,fontsize=8)
    np.save('data/pop_fitting_paras.npy',pop_paras)


 
plt.figure(figsize=(10,7))
plt.subplots_adjust(hspace=0.4)
plt.subplots_adjust(wspace=0.3)
plt.subplot(2,3,1)
plt.title("SA1")
single_paras_fitting(0)
plt.subplot(2,3,2)
plt.title("RA1")
single_paras_fitting(1)
plt.subplot(2,3,3)
plt.title("PC")
single_paras_fitting(2)

plt.subplot(2,3,4)
plt.title("SA1")
pop_paras_fitting(0)
plt.subplot(2,3,5)
plt.title("RA1")
pop_paras_fitting(1)
plt.subplot(2,3,6)
plt.title("PC")
pop_paras_fitting(2)
tmp= {'SA1':paras_dict[0],'RA1':paras_dict[1],'PC':paras_dict[2]}
np.save('data/fitting_paras.npy',tmp)

plt.savefig('saved_figs/submitting/model_fitting.png',bbox_inches='tight', dpi=300) 