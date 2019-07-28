# -*- coding: utf-8 -*-
"""
Created on Sun Dec 30 17:35:43 2018
@author: qiangqiang ouyang
"""

import numpy as np
from scipy import stats
import ultils as alt
import matplotlib.pyplot as plt
from scipy import optimize
import matplotlib.cbook as cbook 
import Receptors as receptorlib
import simset as mysim
from PIL import Image
import img_to_eqstimuli as imeqst


bensimia_RFsize_buf=np.hstack([alt.read_data('data/txtdata/bensimia_RFSIZE.txt',[1,2]),np.loadtxt('data/txtdata/bensimia_RFSIZE.txt')])
observed_RFsize_buf=np.hstack([alt.read_data('data/txtdata/observed_rF_size.txt',[1,2]),np.loadtxt('data/txtdata/observed_rF_size.txt')])
indent_buf=np.array([50,100,200,350,500])*1e-6  #um

Ttype_buf=["SA1","RA1","PC"]
tsensors=[]  
pbuf=np.load('data/loc_pos_buf_fingertip.npy')    
simT=0.4
simdt=0.001

for ch in range(len(Ttype_buf)):
    tsensor=receptorlib.tactile_receptors(Ttype=Ttype_buf[ch])
    tsensor.set_population(pbuf[ch][0],pbuf[ch][1],simTime=simT,sample_rate=1/simdt,Density=pbuf[ch][2],roi=mysim.fingertiproi) 
    tsensors.append(tsensor) 

stimuli=np.zeros([tsensor.t.size,len(indent_buf)])
for m in range(len(indent_buf)):
    rate1=indent_buf[m]*1e6/200
    rate2=-indent_buf[m]*1e6/200
    stimuli[:,m]=receptorlib.step_wave(tsensor.t,0,0.2,rate1,rate2,indent_buf[m])

points=[]
count=0
dist=0
x0=np.arange(-15,15,1)
y0=np.zeros([len(x0),1])
points=np.hstack([x0.reshape([len(x0),1]),y0])
while(dist<14):
    count=count+1
    dist=count*(np.sqrt(3)/2)
    y=y0+dist
    if(count%2==1): x=x0+0.5
    else:x=x0
    A1=np.hstack([x.reshape([len(x),1]),y])
    A2=np.hstack([x.reshape([len(x),1]),-y])
    points=np.vstack([points,A1,A2])

sel=(points[:,0]**2+points[:,1]**2)<=6.1**2
sdots=points[sel,:]

'''
'delete the '...' above and below if want run the simulation again'
simulation_res=[]
for repeat in range(5):
    print('repeat time: ',repeat)
    rad=0.25 #mm
    #pimage=np.ones([3,3])
    tmp_res=[]
    for k in range(len(sdots)):
        probe_loc=sdots[k,:]
        H=np.zeros([len(sdots),1])
        H[k,0]=0.85
        [pimage,eqs]=mysim.constructing_probe_stimuli(np.hstack([sdots,np.ones([len(sdots),1])*rad,H]))
        Aeeps=imeqst.constructing_equivalent_probe_stimuli_from_pimage(pimage[0],pimage[1],pimage[2],mysim.fingertiproi)
        tmp=[]
        for ch in range(2):  #3 afferent types
            tmp1=[]
            for j in range(len(indent_buf)): 
                stimulus=stimuli[:,j]
                DP=stimulus.reshape(len(stimulus),1)
                w=pimage[1]
                h=pimage[2]
                ips=[np.hstack([w/2*np.ones([len(DP),1]),h/2*np.ones([len(DP),1]),DP]),'Depth']
                tsensors[ch].population_simulate(EEQS=Aeeps,Ips=ips,noise=0)
                sel=tsensors[ch].points_mapping_entrys(np.array([[0,0]]))[0]
                tmp1.append(np.array(tsensors[ch].Va[sel,:]))  
            tmp.append(tmp1)
        tmp_res.append(tmp)
    simulation_res.append(tmp_res)
np.save('data/rf_probe_simres.npy',simulation_res)
#-----------------------------------------------
'''

outputd=[]
def plot_sim_rfsize_indents():
    ax=plt.subplot(2,2,1)
    plt.text(-1,8,'(a)',fontsize=14)
    #plt.title("(b)",fontsize=14)
    ax.spines['top'].set_color('None')
    ax.spines['right'].set_color('None')
    buf=sdots
    plt.scatter(buf[:,0],buf[:,1],s=0.2,c='k',cmap=plt.cm.Greys,vmin=0,vmax=3)
    plt.yticks([-6,-3,0,3,6],fontsize=7)
    plt.xticks([-6,-3,0,3,6],fontsize=7)
    plt.xlabel('x [mm]',fontsize=7) 
    plt.ylabel('y [mm]',fontsize=7) 
    
    ax=plt.subplot(2,2,2)
    plt.text(250,43,'(b)',fontsize=14)
    bensimia_RFsize_buf=np.hstack([alt.read_data('data/txtdata/bensimia_RFSIZE.txt',[2,1]),np.loadtxt('data/txtdata/bensimia_RFSIZE.txt')])
    observed_RFsize_buf=np.hstack([alt.read_data('data/txtdata/observed_rF_size.txt',[1,2]),np.loadtxt('data/txtdata/observed_rF_size.txt')])
    prfz=np.load('data/rfsize_probe_indent.npy') 
    
    indent_buf=np.array([50,100,200,350,500])*1e-6  #um
    ax.spines['top'].set_color('None')
    ax.spines['right'].set_color('None')
    marker_buf=['^','s','o','p','d']
    
    for ch in range(2): 
       #measured rf size
       x=indent_buf*1e6
       y1=observed_RFsize_buf[observed_RFsize_buf[:,0]==ch+1,2] 
       plt.plot(x,y1,'--',color='gray',label=Ttype_buf[ch],
                marker=marker_buf[ch],markerfacecolor='none',markersize=4)
    
       #simulated rf size with bensimia model
       y2=bensimia_RFsize_buf[bensimia_RFsize_buf[:,0]==ch+1,2] 
       plt.plot(x,y2,'k',label=Ttype_buf[ch],
                marker=marker_buf[2+ch],markerfacecolor='none',markersize=4)
    
       #simulated rf size with currrent model  
       x=indent_buf*1e6
       y3=prfz[ch]
       plt.plot(x,y3,mysim.colors[ch],label=Ttype_buf[ch],
                marker=marker_buf[ch],markerfacecolor='none',markersize=4)  
       
       alt.wilcoxon_signed_rank_test(y1, y2)
       alt.wilcoxon_signed_rank_test(y1, y3)

       outputd.append(np.array([y1,y2,y3]).T)
       
    plt.yticks([0,10,20,30,40],fontsize=6)  
    plt.xticks([0,100,200,300,400,500,600],fontsize=6)
    plt.xlabel('Indentation depth [um]',fontsize=8)  
    plt.ylabel('RF size [$\mathrm{mm}^{2}$]',fontsize=8)   
    plt.legend(ncol=2,loc=1,fontsize=6,edgecolor='gray')


def plot_prediction_relevance():
   suptitles=['Touchsim','Our work']
   plotlabels=['SA1','RA1','PC']
   
   ticks=[[0,10,20,30],[0,10,20,30]]
   bensimia_rfz=np.hstack([alt.read_data('data/txtdata/bensimia_RFSIZE.txt',[2,1]),np.loadtxt('data/txtdata/bensimia_RFSIZE.txt')])
   observed_rfz=np.hstack([alt.read_data('data/txtdata/observed_rF_size.txt',[1,2]),np.loadtxt('data/txtdata/observed_rF_size.txt')])
   prfz=np.load('data/rfsize_probe_indent.npy') 
   
   y1=[bensimia_rfz[bensimia_rfz[:,0]==1,2],bensimia_rfz[bensimia_rfz[:,0]==2,2]] 
   y2=[observed_rfz[observed_rfz[:,0]==1,2],observed_rfz[observed_rfz[:,0]==2,2]] 
   
   simdata=[y1,prfz] 
   obdata=[y2,y2]

   for model in [0,1]: 
       ax=plt.subplot(2,2,model+3)
       if(model==0): plt.text(33,37,'(c)',fontsize=14)
       ax.spines['top'].set_color('None')
       ax.spines['right'].set_color('None')
       plt.title(suptitles[model],fontsize=8)
       plt.xlabel('Predicted '+'RF size',fontsize=8)
       plt.ylabel('Observed '+'RF size',fontsize=8)
       R2s=np.zeros(3)
       for ch in [0,1]:
           x=simdata[model][ch][:]
           y=obdata[model][ch][:]
           pvalue=round(stats.ranksums(x, y)[1],3)
           print(pvalue)
           fit=alt.curve_fit(x,y)
           R2="{0:.3f}".format(alt.R2(fit[2],y))
           R2s[ch]=R2
           plt.scatter(x,y,marker=mysim.markers[ch],
                       color='w',edgecolors=mysim.colors[ch],s=20,label=plotlabels[ch]+', $\mathrm{R}^{2}$='+str(R2))      #+', P='+str(pvalue)
           
           plt.plot(fit[0],fit[1],'--',color=mysim.colors[ch])
       plt.xticks(ticks[model],fontsize=6)
       plt.yticks(ticks[model],fontsize=6)
       plt.legend(fontsize=6,edgecolor='gray')



def predicted_data_get(): 
    pred_fzize_buf=[]
    sim_res=np.load('data/rf_probe_simres.npy')
    rf_t1=np.zeros([len(sim_res),len(indent_buf)])
    for ch in range(2):
        for repeat in range(5): 
            for indent in range(len(indent_buf)): 
                buf=np.zeros(len(sdots))
                for loc in range(len(sdots)): 
                    tmp1=sim_res[repeat][loc][ch][indent]==0.04  # Va
                    tmp1=np.sum(tmp1)/tsensors[ch].T
                    buf[loc]=tmp1 
                rf_t1[repeat,indent]=np.sum(buf>0.01*tsensors[ch].maxfr)/1.15  
        pred_fzize_buf.append(np.average(rf_t1,0))
    np.save('data/rfsize_probe_indent.npy',pred_fzize_buf)


predicted_data_get()
plt.figure(figsize=(4.6,4.5))
plt.subplots_adjust(hspace=0.7)
plt.subplots_adjust(wspace=0.45)   
plot_sim_rfsize_indents()
plot_prediction_relevance()
A=outputd
plt.savefig('saved_figs/submitting/rf_size.png',bbox_inches='tight', dpi=300)


