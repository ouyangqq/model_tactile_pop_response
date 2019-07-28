# -*- coding: utf-8 -*-
"""
Created on Sun Dec 30 17:35:43 2018
@author: qiangqiang ouyang
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
import matplotlib.pyplot as plt
from scipy import stats
#matplotlib.use('TkAgg')
# -*- coding: utf-8 -*-
shift=0.2
speed=60
pf=0.6#pressing force (N)
rad=0.5
doth=0.5  # dot height

dotspos=np.loadtxt('data/txtdata/texture_dots.txt')   
S=np.hstack([dotspos,rad*np.ones([len(dotspos),1]),doth*np.ones([len(dotspos),1])]) 
[pimg,eqs]=mysim.constructing_probe_stimuli(S)
width,height=pimg[1],pimg[2]
Aeeps=imeqst.constructing_equivalent_probe_stimuli_from_pimage(pimg[0],width,height,mysim.fingertiproi)

simT=width/speed
simdt=0.001
Ttype_buf=['SA1','RA1','PC']
tsensors=[]
pbuf=np.load('data/loc_pos_buf_fingertip.npy')    
for tp in range(len(Ttype_buf)):
    tsensor=rslib.tactile_receptors(Ttype=Ttype_buf[tp])
    tsensor.set_population(pbuf[tp][0],pbuf[tp][1],simTime=simT,sample_rate=1/simdt,
                           Density=pbuf[tp][2],roi=mysim.fingertiproi) 
    tsensors.append(tsensor)

'''
'delete the "..." above and below if want to run the simulation again'
PF=pf*np.ones(int(simT/simdt))
ips=imeqst.img_stimuli_scaning_with_uniformal_speed(simdt,simT,PF,speed,0,width,0,height,shift)
simulation_res=[]
for tp in range(len(Ttype_buf)):
    tmp1=[]
    for i in range(len(ips)):
        A=tsensors[tp].population_simulate(EEQS=Aeeps,Ips=[ips[i],'Pressure'],noise=3) 
        sel=tsensors[tp].points_mapping_entrys(np.array([[0,0]]))[0]
        #tmp=[tsensors[tp].Uc[sel,:],tsensors[tp].Vnf[sel,:],tsensors[tp].Va[sel,:]]
        tmp1.append(np.array(tsensors[tp].Va[sel,:]))
    simulation_res.append(tmp1)
np.save('data/dots_single_repeat_simulation_res.npy',simulation_res) 
#------------------------
'''

def print_dots_spiking_trians():
    ftiproi=np.loadtxt('data/txtdata/fingertip_roi.txt')
    ftiproi=np.vstack([ftiproi,ftiproi[0,:]])
    plt.figure(figsize=(7,4*0.7))
    buf=Aeeps[2]
    sres=np.load('data/dots_single_repeat_simulation_res.npy') 
    
    ax=plt.subplot(4,1,1)
    plt.text(-10,height+6,'(b)',fontsize=14)
    ax.scatter(buf[:,0],buf[:,1],s=0.02,c=1e3*buf[:,5],cmap=plt.cm.Greys,vmin=0,vmax=1)
    plt.yticks([0,height/2,height],fontsize=6)
    plt.xticks(np.arange(0,width+width/10,width/10),fontsize=8)
    
    ax2 = ax.twinx() 
    plt.yticks([])
    plt.ylabel('EPS',fontsize=8)
    rsites=np.arange(-height/2,height/2,shift)
    num=len(rsites)
    sel_points=np.vstack([0*np.ones(num),rsites]).T #ms
    
    for ch in range(3):
        ax1=plt.subplot(4,1,ch+2,sharex=ax)
        plt.subplots_adjust(hspace=0.1)
        for i in range(num):
            #res=np.array(tbuf[i][:])
            res=simdt*np.where(sres[ch][i]==0.04)[0]
            plt.scatter(res*speed,height/2-sel_points[i,1]*np.ones(len(res)),c=mysim.colors[ch],s=0.02)
            plt.xticks(np.arange(0,width+width/10,width/10),fontsize=8)
            plt.yticks([0,height/2,height],fontsize=6)
        
        plt.xlabel('Postion [mm]',fontsize=8) 
        if(ch==1):plt.ylabel('Distance [mm]',fontsize=7) 
        ax2 = ax1.twinx() 
        plt.yticks([])
        plt.ylabel(Ttype_buf[ch],fontsize=8)
        
    plt.xlabel('Postion [mm]')
    plt.savefig('saved_figs/dots_spking_repeat_single.png',bbox_inches='tight', dpi=300)  
    
def plot_observed_texturedots_spiking_trians():
    plt.figure(figsize=(6,3*0.65))
    plt.subplots_adjust(hspace=0.3)
    obtrains_SA1=np.array(Image.open('saved_figs/observed_texturedots_SA1_230.jpg'))
    obtrains_RA1=np.array(Image.open('saved_figs/observed_texturedots_RA1_230.jpg'))
    obtrains_PC=np.array(Image.open('saved_figs/observed_texturedots_PC_230.jpg'))
    ax=plt.subplot(3,1,1)  
    plt.text(-50,-40,'(a)',fontsize=14)
    plt.imshow(obtrains_SA1,aspect='auto')
    plt.xticks([],color='None')
    plt.yticks(np.linspace(0,obtrains_SA1.shape[0],4), np.int16(np.linspace(height,0,4)),fontsize=6)
    ax2 = ax.twinx() 
    plt.yticks([])
    plt.ylabel('SA1',fontsize=8)
    
    ax=plt.subplot(3,1,2,sharex=ax,sharey=ax)  
    plt.imshow(obtrains_RA1,aspect='auto')
    plt.xticks([],color='None')
    plt.yticks(np.linspace(0,obtrains_RA1.shape[0],4), np.int16(np.linspace(height,0,4)),fontsize=6)
    plt.ylabel('Distance [mm]',fontsize=8) 
    ax2 = ax.twinx() 
    plt.yticks([])
    plt.ylabel('RA1',fontsize=8)
    
    ax=plt.subplot(3,1,3) 
    plt.imshow(obtrains_PC,aspect='auto')
    plt.xticks(np.linspace(0,obtrains_PC.shape[1],11), np.int16(np.linspace(0,width,11)),fontsize=8)
    plt.yticks(np.linspace(0,obtrains_PC.shape[0],4), np.int16(np.linspace(height,0,4)),fontsize=6)
    plt.xlabel('Position [mm]',fontsize=8)   
    ax2 = ax.twinx() 
    plt.yticks([])
    plt.ylabel('PC',fontsize=8)
    plt.savefig('saved_figs/ob_dots_spking.png',bbox_inches='tight', dpi=340) 


wd=[0,20]  #mm define a moving window according to work of ref.[43] 

def print_PulsePS_and_PulsePdots():
    sres=np.load('data/dots_single_repeat_simulation_res.npy')
    buf1=np.loadtxt('data/txtdata/ob_Frate_Tdots_RA1_SA1.txt')
    buf2=np.loadtxt('data/txtdata/ob_MIPD_Tdots_RA1_SA1.txt')
    obMIPS=[buf1[12:24,1],buf1[0:12,1],
            np.loadtxt('data/txtdata/ob_Frate_Tdots_PC.txt')[:,1]]
    obMIPD=[buf2[12:24,1],buf2[0:12,1],
            np.loadtxt('data/txtdata/ob_MIPD_Tdots_PC.txt')[:,1]]

    np.save('data/ob_Frate_Tdots.npy',obMIPS)
    np.save('data/ob_MIPD_Tdots.npy',obMIPD)
    
    spaces=buf1[0:12,0]
    N=len(spaces)
    positions=np.linspace(202,15,N) 
    ptimes=(positions/width)*simT
    
    rsites=np.arange(-wd[1]/2,wd[1]/2,shift)
    num=len(rsites)
    
    sim_Frate_Tdots=[]
    sim_MIPD_Tdots=[]
    for ch in range(3):      
        FPS=np.zeros(len(ptimes))
        FPD=np.zeros(len(ptimes))
        for m in range(len(ptimes)):
            wd[0]=spaces[m]*3# Three times the local dot spacing
            duration=wd[0]/width*simT
            st=int(ptimes[m]/tsensors[ch].dt)
            sn=int(duration/tsensors[ch].dt)
            dotsn=(wd[0]*wd[1])/spaces[m]**2
            tmp=np.array(sres[ch])[:,st:st+sn]
            FPS[m]=np.sum(tmp==0.04)/duration/num
            FPD[m]=np.sum(tmp==0.04)/dotsn
        sim_Frate_Tdots.append(FPS)  
        sim_MIPD_Tdots.append(FPD) 
    
    simbuf=[sim_Frate_Tdots,sim_MIPD_Tdots]
    obbuf=[obMIPS,obMIPD]
    suptitles=['MIPS','MIDP']
    for sigt in range(2):  
        ax=plt.subplot(2,2,sigt+1)
        ax.spines['top'].set_color('None')
        ax.spines['right'].set_color('None')
       
        if(sigt==0):plt.text(0.5,220,'(c)',fontsize=14)
        plt.title(suptitles[sigt],fontsize=8)
        for ch in range(3):  
            plt.plot(spaces,simbuf[sigt][ch],mysim.colors[ch],
                     marker=mysim.markers[ch],markerfacecolor='none',markersize=6,label=Ttype_buf[ch])
            plt.plot(spaces,obbuf[sigt][ch],'gray',
                     marker=mysim.markers[ch],markerfacecolor='none',markersize=6)
        if(sigt==0) :    
            plt.xticks([1,2,3,4,5,6],fontsize=6)
            plt.yticks([0,50,100,150,200],fontsize=6)
            plt.ylabel('Mean impulses per second',fontsize=8)
            plt.xlabel('Dot spacing [mm]',fontsize=8)
        
        if(sigt==1): 
            plt.xticks([1,2,3,4,5,6],fontsize=6)
            plt.yticks([0,50,100,150,200,250,300],fontsize=6)
            plt.ylabel('Mean impulses per dots',fontsize=8)
            plt.xlabel('Dot spacing [mm]',fontsize=8)
            
        plt.legend(fontsize=8,edgecolor='w') 
    
    np.save('data/sim_Frate_Tdots.npy',sim_Frate_Tdots)
    np.save('data/sim_MIPD_Tdots.npy',sim_MIPD_Tdots)

outputd=[]    
def plot_prediction_relevance():
   suptitles=['MIPS','MIDP']
   plotlabels=['SA1, ','RA1, ','PC, ']
   ticks=[[0,50,100,150],[0,50,100,150,200,250]]
   obdata=[np.load('data/ob_Frate_Tdots.npy'),np.load('data/ob_MIPD_Tdots.npy')] 
   simdata=[np.load('data/sim_Frate_Tdots.npy'),np.load('data/sim_MIPD_Tdots.npy')]
   
   for sigt in [0,1]: 
       ax=plt.subplot(2,2,sigt+3)
       if(sigt==0): plt.text(-2,160,'(d)',fontsize=14)
       ax.spines['top'].set_color('None')
       ax.spines['right'].set_color('None')
       plt.title(suptitles[sigt],fontsize=8)
       plt.xlabel('Predicted '+suptitles[sigt],fontsize=8)
       plt.ylabel('Observed '+suptitles[sigt],fontsize=8)
       R2s=np.zeros(3)
       for ch in [0,1,2]:
           x=simdata[sigt][ch][:]
           y=obdata[sigt][ch][:]
           pvalue=round(stats.ranksums(x, y)[1],2)
           print(pvalue)

           fit=alt.curve_fit(x,y)
           R2="{0:.3f}".format(alt.R2(fit[2],y))
           R2s[ch]=R2
           plt.scatter(x,y,color='w',edgecolors=mysim.colors[ch],marker=mysim.markers[ch],
                       s=15,label=plotlabels[ch]+' $\mathrm{R}^{2}$='+str(R2))   #+', P='+str(pvalue)   
           outputd.append(np.array([y,x]).T)
           plt.plot(fit[0],fit[1],'--',color=mysim.colors[ch])

       plt.xticks(ticks[sigt],fontsize=6)
       plt.yticks(ticks[sigt],fontsize=6)

       plt.legend(loc=2,fontsize=6,edgecolor='gray')

plot_observed_texturedots_spiking_trians()
print_dots_spiking_trians()


plt.figure(figsize=(5.2,6))
plt.subplots_adjust(wspace=0.3)
plt.subplots_adjust(hspace=0.5)
print_PulsePS_and_PulsePdots()
plot_prediction_relevance()
A=outputd[0]
for i in range(5):A=np.hstack([A,outputd[i+1]])

plt.savefig('saved_figs/Tdots_relevant.png',bbox_inches='tight', dpi=300) 

sim_img1=Image.open('saved_figs/dots_spking_repeat_single.png')
ob_img1=Image.open('saved_figs/ob_dots_spking.png')
sim_img2=Image.open('saved_figs/Tdots_relevant.png')

img=Image.new(sim_img1.mode,(ob_img1.size[0]+sim_img2.size[0]+50,
                         ob_img1.size[1]+sim_img1.size[1]))
img.paste(ob_img1,(10,20))
img.paste(sim_img1,(0,ob_img1.size[1]))
img.paste(sim_img2,(sim_img1.size[0],40))
img.show()
img.save("saved_figs/submitting/Dot_texture.png")
