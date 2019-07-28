# -*- coding: utf-8 -*-
"""
Created on Sun Dec 30 17:35:43 2018
@author: qiangqiang ouyang
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from mpl_toolkits.mplot3d import Axes3D
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 30 17:35:43 2018
@author: qiangqiang ouyang
"""

import Receptors as rslib
import simset as mysim 
import ultils as alt
import img_to_eqstimuli as imeqst
from scipy.optimize import curve_fit
#simulation
#sel_points=np.array([0,0]).reshape(1,2) #ms  

def fr_func(x, a,b):
    tmp=a*np.log(x)-b
    return (tmp>0)*tmp+0*(tmp<=0)

def fp_func(x, u,a):   
    tmp=1/(1+np.e**(-x/a+u/a))
    return tmp
  
Ttype_buf=['SA1','RA1','PC']
tsensors=[]
pbuf=np.load('data/loc_pos_buf_fingertip.npy')    
for ch in range(len(Ttype_buf)):
    tsensor=rslib.tactile_receptors(Ttype=Ttype_buf[ch])
    tsensor.set_population(pbuf[ch][0],pbuf[ch][1],simTime=1,sample_rate=1000,Density=pbuf[ch][2],roi=mysim.fingertiproi) 
    tsensors.append(tsensor)

'----Vibration stimuli construction from reference [8]---'
ob_fr_data=np.load('data/ob_vibro_firing_rate.npy')

st1=[[20],[50],[100],[300],[600]]
st2=[[10,50],[10,100],[50,250],[50,500],[100,1000]]
st3=[[10,25],[5,100],[25,250],[25,500],[50,500]]
sa1=ob_fr_data[0][0][:,0]*1e-6
sa2=ob_fr_data[1][0][:,0]*1e-6
sa3=ob_fr_data[2][0][:,0]*1e-6
sa_buf=[sa1,sa2,sa3]
simT=[1,1,1]  # 1 Simulation time for different signal
dt=0.00098
t=[np.arange(0,simT[0],dt),np.arange(0,simT[1],dt),np.arange(0,simT[2],dt)] 
stimuli_buf=[]


tmp_SA1=[]
for j in range(len(st1)):
    setd=sa1[j*15:(j+1)*15]
    for i in range(len(setd)):      
        stimulus=rslib.sin_wave(t[0],2*np.pi*st1[j][0],setd[i])
        tmp_SA1.append([stimulus,[st1[j],setd[i]]])

tmp_RA1=[]
for j in range(len(st2)):
    setd=sa2[j*5:(j+1)*5]
    for i in range(len(setd)):  
        stimulus=rslib.sin_wave(t[1],2*np.pi*st2[j][0],setd[i])\
        +rslib.sin_wave(t[1],2*np.pi*st2[j][1],setd[i])
        tmp_RA1.append([stimulus,[st2[j],setd[i]]])
        
tmp_PC=[]  
for j in range(len(st3)):
    setd=sa3[j*5:(j+1)*5]
    for i in range(len(setd)):   
        a=np.random.uniform(-setd[i],setd[i],t[2].size)
        stimulus=rslib.butterworth_filter(1,2*np.sqrt(2)*a,st3[j],'band',1/dt)
        tmp_PC.append([stimulus,[st3[j],setd[i]]])
        
stimuli=[tmp_SA1,tmp_RA1,tmp_PC,sa_buf]


ch_sl=[0,1,2]
sg_sl=[0,1,2]
sfr=0
spr=1

'''
'delete the '...' above and below if want run the simulation again'
testing_bf=[]
rad=0.5
#pimage=np.ones([3,3])
[pimage,eqs]=mysim.constructing_probe_stimuli(np.array([[0,0,rad,1]]))  #Is
Aeeps=imeqst.constructing_equivalent_probe_stimuli_from_pimage(pimage[0],pimage[1],pimage[2],mysim.fingertiproi)
np.save('data/vibro_probe_1mm_eeqs.npy',Aeeps)

simdata=[]
for ch in ch_sl:  
    tmp1=[]
    for sigt in sg_sl:  # signal type
            tsensors[ch].set_population(pbuf[ch][0],pbuf[ch][1],
                    simTime=simT[sigt],sample_rate=1/dt,
                    Density=pbuf[ch][2],roi=mysim.fingertiproi) 
            tmp2=[]
            for st in range(len(stimuli[sigt])):  # signal set
                setstimuli=[]
                DP=stimuli[sigt][st][0][:]
                DP=DP.reshape(len(DP),1)
                ips=[np.hstack([pimage[1]/2*np.ones([len(DP),1]),pimage[2]/2*np.ones([len(DP),1]),DP]),'Depth']
                tsensors[ch].population_simulate(EEQS=Aeeps,Ips=ips,noise=0)
                sel=tsensors[ch].points_mapping_entrys(np.array([[0,0]]))[0]
                tmp2.append(np.array(tsensors[ch].Va[sel,:]))     
            tmp1.append(tmp2)
    simdata.append(tmp1)
np.save('data/sim_vibro_fring_data.npy',simdata)
#----------------------------------------------
'''

colors=['g','b','r']
marker_buf=['^','s','o','p','d'] 
simd=np.load('data/sim_vibro_fring_data.npy')

# computing firing rate
simdata=[]
for ch in ch_sl: 
    tmp1=[]
    for sigt in sg_sl:  # signal type
            tmp2=[]
            for st in range(len(stimuli[sigt])):
                res=np.sum(simd[ch][sigt][st][:]==0.04)/simT[sigt] # Va
                tmp2.append(res)
            tmp1.append(np.array(tmp2))
    simdata.append(tmp1)

titles=['(a)','(b)','(c)','(d)']
titles1=['Sinusoidal','Diharmonic','Noise']
ylabels=['Firing rate [Hz]','Probability']
fyticks=[[0,25,50],[0,100,200],[0,100,200,300]]
fxticks=[[1,10,100],[1,10,100],[1,10,100]]
ylimts=[500,200,100]
def plot_mean_fring_rate():
    plt.figure(figsize=(8,7))
    for ch in ch_sl: 
        for sigt in sg_sl: 
            plt.subplot(3,3,3*sigt+ch+1)
            if(sigt==1)&(ch==0):plt.ylabel('Firing rate [spikes/s]',fontsize=12)
            if(sigt==2)&(ch==1):plt.xlabel('Depth [um]',fontsize=12)
            if(ch==2):plt.text(ylimts[sigt],2*fyticks[2][-1]/3,titles1[sigt],rotation=90,fontsize=10)
            if(sigt==0):plt.title(Ttype_buf[ch],fontsize=8)
            if((ch==0)&(sigt==0)):plt.text(0.01,55,titles[0],fontsize=14)
            for st in range(5):
                space=int(len(stimuli[sigt])/5)
                x=np.array(simdata[ch][sigt])[st*space:(st+1)*space]
                y=ob_fr_data[sigt][ch][st*space:(st+1)*space,1]
                fit=alt.curve_fit(x,y)
                R2="{0:.3f}".format(alt.R2(fit[2],y))
                plt.plot(1e6*stimuli[3][sigt][st*space:(st+1)*space],np.array(simdata[ch][sigt])[st*space:(st+1)*space],
                         linewidth=0.5,label=str(stimuli[sigt][st*space][1][0])+' $\mathrm{R}^{2}$='+str(R2),c=mysim.color_bf[ch][st],
                        marker=marker_buf[st],markerfacecolor='none',markersize=4)
                

                plt.plot(ob_fr_data[sigt][ch][st*space:(st+1)*space,0],ob_fr_data[sigt][ch][st*space:(st+1)*space,1],
                        '--',linewidth=0.4, c='gray',marker=marker_buf[st],markerfacecolor='none',markersize=3)
                
                print(stats.ranksums(x, y))
            plt.xscale('log')
            plt.xticks(fontsize=6)
            if(ch==0):plt.yticks(fyticks[0],fontsize=6)
            if(ch==1):plt.yticks(fyticks[1],fontsize=6)
            if(ch==2):plt.yticks(fyticks[2],fontsize=6)
            plt.legend(loc=2,prop={'family':'simSun','size':6}) 


Tsigts=['Sine','Di','Noise']
markers=['s','d','o']


def plot_prediction_relevance():
    plt.figure(figsize=(7,6))
    for ch in ch_sl: 
        for sigt in sg_sl: 
            plt.subplot(3,3,3*ch+sigt+1)
            if(sigt==0)&(ch==1):plt.ylabel('Observed Firing rate [Hz]',fontsize=10)
            if(sigt==1)&(ch==2):plt.xlabel('Predicted Firing rate [Hz]',fontsize=10)
            if(sigt==2):plt.text(fyticks[ch][-1]*1.05,fyticks[ch][-1]/2,Ttype_buf[ch],fontsize=10)
            if(ch==0):plt.title(titles1[sigt],fontsize=8)
            if((ch==0)&(sigt==0)):plt.text(0.01,60,titles[0],fontsize=14)
            for st in range(5):
                space=int(len(stimuli[sigt])/5)
                x=np.array(simdata[ch][sigt])[st*space:(st+1)*space]
               
                y=ob_fr_data[sigt][ch][st*space:(st+1)*space,1]
                pvalue=pvalue=round(stats.ranksums(x, y)[1],2)
                fit=alt.curve_fit(x,y)
                R2=round(alt.R2(fit[2],y),2)
                
                plt.scatter(x,y,c=mysim.color_bf[ch][st],marker=marker_buf[st],s=10,
                            label=str(stimuli[sigt][st*space][1][0])+' '+' $\mathrm{R}^{2}$='+str(R2)+', P='+str(pvalue))
                print(pvalue)
            if(ch==0):
                plt.xticks(fyticks[0],fontsize=6)
                plt.yticks(fyticks[0],fontsize=6)
            if(ch==1):
                plt.xticks(fyticks[1],fontsize=6)
                plt.yticks(fyticks[1],fontsize=6)
            if(ch==2):
                plt.xticks(fyticks[2],fontsize=6)
                plt.yticks(fyticks[2],fontsize=6)
            plt.legend(fontsize=6,edgecolor='gray')


outputd=[]
def plot_prediction_relevance_sine_20hz():
   ch_sl=[0,1,2]
   sg_sl=[0]
   plt.figure(figsize=(8.1,2.3))
   plt.subplots_adjust(wspace=0.4)
   for ch in ch_sl:
       ax=plt.subplot(1,3,ch+1)
       plt.title(Ttype_buf[ch],fontsize=8)
       if(ch==0): plt.text(-4,30,'(b)',fontsize=14)
       ax.spines['top'].set_color('None')
       ax.spines['right'].set_color('None')
       plt.xlabel("Predicted rate [spikes/s]",fontsize=8)
       plt.ylabel("Measured rate [spikes/s]",fontsize=8)

       for sigt in sg_sl:
           obser,preds=ob_fr_data[sigt][ch][:,1],np.array(simdata[ch][sigt])
           outputd.append(np.array([obser,preds]).T)
           
           pvalue=round(stats.ranksums(np.array(simdata[ch][sigt]),
                                       ob_fr_data[sigt][ch][:,1])[1],3)
            
           print(pvalue)
           space=int(len(stimuli[sigt])/5)
           for st in range(1): 
               x=np.array(simdata[ch][sigt])[st*space:(st+1)*space]
               y=ob_fr_data[sigt][ch][:,1][st*space:(st+1)*space]
               fit=alt.curve_fit(x,y)
               R2="{0:.3f}".format(alt.R2(fit[2],y))
               plt.plot(fit[0],fit[1],'--',linewidth=1,color=mysim.colors[ch])
               plt.scatter(x,y,marker=markers[ch],color=mysim.colors[ch],s=10,label=' $\mathrm{R}^{2}$='+str(R2))    
   
       plt.xticks(fontsize=8)
       plt.yticks(fontsize=8)
       plt.legend(fontsize=8,edgecolor='gray')

outputd=[]
def plot_prediction_relevance1():
   plt.figure(figsize=(7.2,2))
   plt.subplots_adjust(wspace=0.4)
   for sigt in sg_sl:
       ax=plt.subplot(1,3,sigt+1)
       plt.title(titles1[sigt],fontsize=8)
       if(sigt==0): plt.text(-50,450,titles[1],fontsize=14)
       ax.spines['top'].set_color('None')
       ax.spines['right'].set_color('None')
       plt.xlabel("Predicted rate[Hz]",fontsize=8)
       plt.ylabel("Measured rate[Hz]",fontsize=8)
       R2s=np.zeros(len(sg_sl))
       plt.plot(np.arange(0,400,1),np.arange(0,400,1),'-',color='k')
       for ch in ch_sl:
           obser,preds=ob_fr_data[sigt][ch][:,1],np.array(simdata[ch][sigt])
           outputd.append(np.array([obser,preds]).T)
           
           pvalue=round(stats.ranksums(np.array(simdata[ch][sigt]),
                                       ob_fr_data[sigt][ch][:,1])[1],3)
           print(pvalue)
           space=int(len(stimuli[sigt])/5)
           R2s=np.zeros(5)
           for st in range(5): 
               x=np.array(simdata[ch][sigt])[st*space:(st+1)*space]
               y=ob_fr_data[sigt][ch][:,1][st*space:(st+1)*space]
               fit=alt.curve_fit(x,y)
               R2s[st]=alt.R2(fit[2],y)
               plt.plot(fit[0],fit[1],'--',linewidth=1,color=mysim.color_bf[ch][st])
               plt.scatter(x,y,marker=markers[ch],color=mysim.color_bf[ch][st],s=10)    
           avgR2=round(np.average(R2s),2)
           plt.scatter(0,0,marker=markers[ch],color=mysim.colors[ch],s=10,label=Ttype_buf[ch]+' '+' $\mathrm{R}^{2}$='+str(avgR2)+', P='+str(pvalue))    
           plt.legend(loc=1,fontsize=6,edgecolor='k')
       if(ch==0):
            plt.xticks(fyticks[0],fontsize=6)
            plt.yticks(fyticks[0],fontsize=6)
       if(ch==1):
            plt.xticks(fyticks[1],fontsize=6)
            plt.yticks(fyticks[1],fontsize=6)
       if(ch==2):
            plt.xticks(fyticks[2],fontsize=6)
            plt.yticks(fyticks[2],fontsize=6)

       plt.legend(loc=1,fontsize=6,edgecolor='gray')

         
'plot_Mean_fring_rate_across_evoked_neurons'
plot_mean_fring_rate()   
plt.savefig('saved_figs/Mean_fring_rate_across_evoked_neurons.png',bbox_inches='tight', dpi=300) 



'plot_plot_prediction_relevance with neurophysiological counterparts'
plot_prediction_relevance1()   
plt.savefig('saved_figs/prediction_relevance1.png',bbox_inches='tight', dpi=300) 
A1=outputd[0]
for i in range(2):A1=np.hstack([A1,outputd[i+1]])
A2=outputd[3]
for i in range(5):A2=np.hstack([A2,outputd[i+4]])

'plot_plot_prediction_relevance with neurophysiological counterparts'
plot_prediction_relevance_sine_20hz()   
plt.savefig('saved_figs/prediction_relevance_sine_20.png',bbox_inches='tight', dpi=300) 


from PIL import Image
img1=Image.open('saved_figs/Mean_fring_rate_across_evoked_neurons.png')
img2=Image.open('saved_figs/prediction_relevance_sine_20.png')
img=Image.new(img1.mode,(img1.size[0],img1.size[1]+img2.size[1]))
img.paste(img1,(0,0))
img.paste(img2,(30,img1.size[1]))

img.save('saved_figs/submitting/res_vibro_fring.png')
img.show()
