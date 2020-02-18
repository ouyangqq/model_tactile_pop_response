# -*- coding: utf-8 -*-
"""
Created on Sat Nov 18 17:11:16 2017
@author: Administrator
"""
import os 
import sys
import ultils as alt
import Receptors as rslib
import numpy as np
import matplotlib.pyplot as plt
import simset as mysim 
from PIL import Image
import img_to_eqstimuli as imeqst
from scipy import stats

#matplotlib.use('TkAgg')
# -*- coding: utf-8 -*-

width=27.08#mm
#width=8#mm

height=12#mm

shift=0.2

speed=0

depth=1# mm 
#During  each  stimulus  the  bar  or  bars  depressed  the  skin  by  1,000  um  (rise  time  50  ms). 
doth=3

simT=0.5#0.1
simdt=0.001

Ttype_buf=['SA1','RA1','PC']
tsensors=[]
pbuf=np.load('data/loc_pos_buf_fingertip.npy')    
for tp in range(len(Ttype_buf)):
    tsensor=rslib.tactile_receptors(Ttype=Ttype_buf[tp])
    tsensor.set_population(pbuf[tp][0],pbuf[tp][1],simTime=simT,sample_rate=1/simdt,Density=pbuf[tp][2],roi=mysim.fingertiproi) 
    tsensors.append(tsensor)
  
barlocs=np.loadtxt('Data/txtdata/loc_gratings.txt')[:,0] 
#barlocs=np.loadtxt('Data/txtdata/one_bar_res.txt')[288:292,0] 

'''
pimg=np.zeros([int(height/rslib.Dbp),int(width/rslib.Dbp)])
for i in range(len(barlocs)-1):
    if(i%2==1): pimg[:,int(barlocs[i]/rslib.Dbp):int(barlocs[i+1]/rslib.Dbp)]=8
    elif(i%2==0): pimg[:,int(barlocs[i]/rslib.Dbp):int(barlocs[i+1]/rslib.Dbp)]=0
pimg[:,0:int(barlocs[0]/rslib.Dbp)]=0
pimg[:,int(barlocs[-1]/rslib.Dbp):]=0
buf1=imeqst.constructing_equivalent_probe_stimuli_from_pimage(pimg,width,height,mysim.fingertiproi)
'''
img2 =Image.open('saved_figs/gratings_24-15.jpg')
buf1=imeqst.constructing_equivalent_probe_stimuli_from_image(img2,width,height,mysim.fingertiproi)
np.save('data/forms_gratings.npy',buf1)


#--------------------------------
sp_for_shift=0
#Dp=depth*np.ones(int(simT/simdt))*1e-3 #m
Dp=rslib.step_wave(tsensor.t,0*simT,0.9*simT,20,-20,depth*0.001).reshape([tsensor.t.size,1]) #rise time 50 ms  rate=1000/50
stimuli_buf=imeqst.img_stimuli_static_pressing(simdt,simT,Dp,buf1[1],mysim.fingertiproi,0,height/2,shift,width)



#------------------
simulation_res=[]
Aeeps=np.load('data/forms_gratings.npy')[2]
Aeeps[1]=Aeeps[1]*doth
for tp in range(1):
    tmp=[]
    for row in range(len(stimuli_buf)):
        ips=stimuli_buf[row]
        tsensors[tp].population_simulate(EEQS=Aeeps,Ips=[ips,'Depth'])
        tmp.append(np.array(tsensors[tp].Va))  
    simulation_res.append(tmp)
np.save('data/gratings_simulation_res.npy',simulation_res)   
#------------------


outputd=[]
def print_gratings_spiking_trians():
    ftiproi=mysim.fingertiproi#*rslib.rtm(-np.pi/2)
    ftiproi=np.vstack([ftiproi,ftiproi[0,:]])
    #colr=['g','b','c']
    plt.figure(figsize=(2,6))
    plt.subplots_adjust(hspace=0.35)
    
    buf=np.load('data/forms_gratings.npy')[1]
    sres=np.load('data/gratings_simulation_res.npy') 
    obdata=np.loadtxt('data/txtdata/fr_gratings.txt')
    #obdata=np.loadtxt('Data/txtdata/one_bar_res.txt')[0:-4,:] 
    tbuf=np.zeros([len(sres[0]),int(simT/simdt)])
    
    ax=plt.subplot(3,1,1)
    plt.text(-5,18,"(d)",fontsize=14)#
    ax.spines['top'].set_color('None')
    ax.spines['right'].set_color('None')
    plt.scatter(buf[:,0],buf[:,1],s=0.05,c=1e3*buf[:,5],cmap=plt.cm.Greys,vmin=0,vmax=1)
    
    '''
    tx,ty=np.linspace(0,width,100),height/2*np.ones(100)
    plt.plot(tx,ty,'k--')
    
    plt.plot(ftiproi[:,0]+width/2,ftiproi[:,1]+height/2,'y-',linewidth=1)
    plt.fill_between(ftiproi[:,0]+width/2,ftiproi[:,1]+height/2,facecolor='y',alpha=0.5) 
    plt.annotate('', xy=(6*width/6,height/2), xytext=(2.5*width/4,height/2),
                 arrowprops=dict(color='c',headwidth = 5,width = 0.05,shrink=0.00))
    '''   
    plt.xticks([0,4,8,12,16,20,24,28],fontsize=8)
    plt.yticks([0,height/2,height],fontsize=8)
    plt.ylabel("y [mm]",fontsize=8)
    
    #recorded sites
    #num=10
    #sel_points=np.vstack([0*np.ones(num),np.linspace(-height/2,height/2,num)]).T #ms
    #sel_points=np.array([[0,0]]) #mm
    sel_points=np.array([[0,0]])
    #ax.plot(sel_points[:,0]+width/2,sel_points[:,1]+height/2,c='k',linewidth=0,
            #marker='o',markersize=5,markerfacecolor='w')
    ax=plt.subplot(3,1,2)
    ax.spines['top'].set_color('None')
    ax.spines['right'].set_color('None')
    Aobdata=np.zeros([len(tbuf),2])
    xt=np.linspace(0,width,len(tbuf))
    for i in range(len(tbuf)):
        sel=np.where(np.abs(xt[i]-obdata[:,0])==np.min(np.abs(xt[i]-obdata[:,0])))[0][0]
        Aobdata[i,0]=obdata[sel,0]
        Aobdata[i,1]=obdata[sel,1]
    
    res=np.zeros(len(sres[0]))
    for ch in range(1):
        sel_entry=tsensors[ch].points_mapping_entrys(sel_points)
        for sp in range(len(tbuf)):
            #buf1[sp]=len(sres[ch][sp][1][sel_entry])/tsensors[ch].T
            A=(sres[ch][sp][sel_entry,:]==0.04)
            A=np.sum(A)
            #A=A[A>0]
            res[sp]=np.average(A)/tsensors[ch].T
            #tbuf[sp,:]=sres[ch][sp][0][sel_entry,:] # sel Vf signal
        #res[13:35]=15
        #res[40:43]=15
        #res[50:53]=15
        #res=np.average(tbuf,1)
        plt.plot(Aobdata[:,0],res,mysim.colors[ch],label='Simulated',
                 marker='o',markerfacecolor='none',markersize=3)
        plt.plot(Aobdata[:,0],Aobdata[:,1],'gray',label='Recorded',
                 marker='o',markerfacecolor='none',markersize=3)
        plt.yticks([0,50,100,150],fontsize=8)
        plt.xticks([0,4,8,12,16,20,24,28],fontsize=8)
    
    plt.legend(loc=1,prop={'family':'simSun','size':7}) 
    plt.xlabel("x [mm]",fontsize=8)
    plt.ylabel("Firing rate",fontsize=8)
    plt.savefig('saved_figs/gratings_firing.png',bbox_inches='tight', dpi=300)     
    
    ax=plt.subplot(3,1,3)   
    ax.spines['top'].set_color('None')
    ax.spines['right'].set_color('None')
    for ch in [0]:
        x=res
        y=Aobdata[:,1]
        #pvalue=round(stats.ranksums(x, y)[1],2)
        [R21,pval]=stats.pearsonr(y,x)
        print(pval)

        fit=alt.curve_fit(x,y)
        R2="{0:.3f}".format(alt.R2(fit[2],y))
        plt.scatter(x,y,color='w',edgecolors=mysim.colors[ch],marker=mysim.markers[ch],
                       s=15,label=' $\mathrm{R}^{2}$='+str(R2))   #+', P='+str(pvalue)   
        outputd.append(np.array([y,x]).T)
        plt.plot(fit[0],fit[1],'--',color=mysim.colors[ch])  
        
        [R21,pval]=stats.pearsonr(y,x)
        print('Pvalue='+str(pval))
        
    plt.yticks([0,20,40,60,80],fontsize=8)
    plt.xticks([0,20,40,60,80],fontsize=8)    
    plt.legend(loc=1,prop={'family':'simSun','size':8}) 
    plt.xlabel("observed Firing rate",fontsize=8)
    plt.ylabel("Simulated Firing rate",fontsize=8)

#print_fig(0)
print_gratings_spiking_trians()
plt.savefig('saved_figs/submitting/gratings_firing.png',bbox_inches='tight', dpi=300)

A_3mm_bar=outputd

