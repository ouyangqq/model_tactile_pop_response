# -*- coding: utf-8 -*-
"""
Created on Sun Dec 30 17:35:43 2018
@author: qiangqiang ouyang
"""

import numpy as np
import ultils as alt
import matplotlib.pyplot as plt
from scipy import optimize
import Receptors as rslib
import simset as mysim
import img_to_eqstimuli as imeqst
from scipy import stats
path='L:\PHD_Research\papers\Manuscript_TBioCAS'

Ttype_buf=["SA1","RA1","PC"]
tsensors=[]  
pbuf=np.load('data/loc_pos_buf_fingertip.npy')    
simT=1
simdt=0.001

for ch in range(len(Ttype_buf)):
    tsensor=rslib.tactile_receptors(Ttype=Ttype_buf[ch])
    tsensor.set_population(pbuf[ch][0],pbuf[ch][1],simTime=simT,sample_rate=1/simdt,Density=pbuf[ch][2],roi=mysim.fingertiproi) 
    tsensors.append(tsensor) 

curves=np.array([0,80.6,172,256,340,521,694])  #m
PFs=np.array([10,15,20])*9.8*1e-3
rads=1000/curves #mm

distances=np.arange(-1,4.5,0.5) #mm

'''
'delete the '...' above and below if want run the simulation again'
#-----------simulating reponse change as a function of curvature  --------------
simulation_res=[]
eps_buf=[]
for ch in range(2):  #3 afferent types
    tmp=[]
    for k in range(len(PFs)): 
        tmp1=[]
        Pd=np.ones([tsensor.t.size,1])*PFs[k]
        Pd[0],Pd[1],Pd[-1]=0,0,0
        #Pd=rslib.step_wave(tsensor.t,0*simT,simT,4000,-4000,PFs[k]).reshape([tsensor.t.size,1])
        for j in range(len(rads)): 

            if(curves[j]==0):[pimage,eps]=mysim.constructing_probe_stimuli(np.array([[0,0,20,1]]))
            else:[pimage,eps]=mysim.constructing_spherically_curved_surfaces_probes(np.array([[0,0,rads[j],2]]))
        
            if(ch==0)&(k==0):eps_buf.append(eps)
            Aeeps=imeqst.constructing_equivalent_probe_stimuli_from_pimage(pimage[0],pimage[1],pimage[2],mysim.fingertiproi)
            w=pimage[1]
            h=pimage[2]
            ips=[np.hstack([w/2*np.ones([len(Pd),1]),h/2*np.ones([len(Pd),1]),Pd]),'Pressure']
            tsensors[ch].population_simulate(EEQS=Aeeps,Ips=ips,noise=0)
            sel=tsensors[ch].points_mapping_entrys(np.array([[0,0]]))[0]
            tmp1.append([np.array(tsensors[ch].Vnf[sel,:]),np.array(tsensors[ch].Va[sel,:])])  
        tmp.append(tmp1)
    simulation_res.append(tmp)
np.save('data/surface_curve_simres.npy',simulation_res)
np.save('data/surface_curve_eps.npy',eps_buf)
#-------------------------------------------------------------------

#-----------simulating reponse change as a function of distance under different curvature--------------
simulation_res=[]
eps_buf=[]
for ch in range(1):  #3 afferent types
    tmp=[]
    for k in range(len(rads)): 
        tmp1=[]
        Pd=np.ones([tsensor.t.size,1])*PFs[1]
        Pd[0],Pd[1],Pd[-1]=0,0,0
        #Pd=rslib.step_wave(tsensor.t,0*simT,0.9*simT,5000,-5000,PFs[1]).reshape([tsensor.t.size,1])
        #15g
        for j in range(len(distances)): 
            if(curves[k]==0):[pimage,eps]=mysim.constructing_probe_stimuli(np.array([[0,0,20,2]])) # construt a plane
            # if curvature =0, set a flat plane  
            else:[pimage,eps]=mysim.constructing_spherically_curved_surfaces_probes(np.array([[0,0,rads[k],1]]))
            Aeeps=imeqst.constructing_equivalent_probe_stimuli_from_pimage(pimage[0],pimage[1],pimage[2],mysim.fingertiproi)
            #stimulus=stimuli[:,j]
            w=pimage[1]
            h=pimage[2]
            ycenter=h/2
            y=distances[j]+ycenter
            ips=[np.hstack([w/2*np.ones([len(Pd),1]),y*np.ones([len(Pd),1]),Pd]),'Pressure']
            tsensors[ch].population_simulate(EEQS=Aeeps,Ips=ips,noise=0)
            sel=tsensors[ch].points_mapping_entrys(np.array([[0,0]]))[0]
            tmp1.append([np.array(tsensors[ch].Vnf[sel,:]),np.array(tsensors[ch].Va[sel,:])])  
        tmp.append(tmp1)
    simulation_res.append(tmp)
np.save('data/surface_curve_Rf_dis_simres.npy',simulation_res)
#-----------------------------------------------
'''

def plot_eps_under_differnt_curvature():
    eps_data=np.load('data/surface_curve_eps.npy')    
    for i in range(1,len(rads)):
        ax=plt.subplot(7,6,6+i)
        if(i==1):plt.text(-2.5,38,"(a)",fontsize=14)#
        if(i==3):plt.text(0,40,"Contructed EPS",fontsize=10)#
        label=str(curves[i])+' $\mathrm{m}^{-1}$'
        plt.title(label,fontsize=8)#loc="left"
        ax.spines['top'].set_color('None')
        ax.spines['right'].set_color('None')
        plt.scatter(eps_data[i][:,0],eps_data[i][:,1],c=1e3*eps_data[i][:,5],s=0.01,
                    cmap=plt.cm.Greys,vmin=0,vmax=rads[i])  
    
        plt.yticks([0,13,26],fontsize=6)  
        plt.xticks([0,13,26],fontsize=6)
        plt.xlabel('x [mm] ',fontsize=8)  
        if(i==1):plt.ylabel('y [mm]',fontsize=8)   


outputd=[]
def plot_impulses_response_as_func_of_curvature_for_SA1():
    obdata=np.loadtxt('data/txtdata/fring_diff_cav.txt')
    ax=plt.subplot(323)
    plt.text(0,154,"(b)",fontsize=14)#
    plt.title('SA1',fontsize=8)
    ax.spines['top'].set_color('None')
    ax.spines['right'].set_color('None')
    marker_buf=['^','s','o','p','d']
    sim_data=np.load('data/surface_curve_simres.npy')
    sim_fr_cav=[]
    for ch in [0]: 
       for k in range(len(PFs)): 
  
           preds=np.zeros(len(rads))
           for i in range(len(rads)):preds[i]=np.sum(sim_data[ch][k][i][1][:]==0.04)
           
           plt.plot(curves,preds,mysim.colors[ch],label=str(PFs[k])+' N',
                    marker=marker_buf[k],markerfacecolor='none',markersize=6)  
           obser=obdata[k*len(rads):(k+1)*len(rads),ch+1]
           plt.plot(curves,obser,'gray',
                    marker=marker_buf[k],markerfacecolor='none',markersize=6)  
           outputd.append(np.array([obser,preds]).T)
           sim_fr_cav.append(preds)
    np.save('data/sim_fr_cav_SA1.npy',sim_fr_cav)  
    plt.yticks([0,50,100,150],fontsize=6)  
    plt.xticks([0,100,200,300,400,500,600,700],fontsize=6)
    plt.xlabel('Curvature [$\mathrm{m}^{-1}$] ',fontsize=8)  
    plt.ylabel('Response impulses in 1s',fontsize=8)   
    plt.legend(loc=0,prop={'family':'simSun','size':7}) 
    
    
 
def plot_impulses_response_as_func_of_curvature_for_RA1():
    
    A=np.hstack([alt.read_data('data/txtdata/fring_diff_cav_RA1.txt',[1,2,3]),
                        np.loadtxt('data/txtdata/fring_diff_cav_RA1.txt')])
    buf=A[A[:,0]==1,2]
    gwt1=np.zeros([len(curves),2])
    gwt1[0,:]=[np.average(buf[0:5]),np.std(buf[0:5])]
    for i in range(6): gwt1[i+1,:]=np.average(buf[5+6*i:5+i*6+6]),np.std(buf[5+6*i:5+i*6+6])
    
    buf=A[A[:,0]==2,2]
    gwt2=np.zeros([len(curves),2])
    gwt2[0,:]=[np.average(buf[0:4]),np.std(buf[0:4])]
    for i in range(6): gwt2[i+1,:]=np.average(buf[4+5*i:4+i*5+5]),np.std(buf[5+6*i:5+i*6+6])
   
    buf=A[A[:,0]==3,2]
    gwt3=np.zeros([len(curves),2])
    gwt3[0,:]=[np.average(buf[0:5]),np.std(buf[0:6])]
    for i in range(6): gwt3[i+1,:]=np.average(buf[6+7*i:6+i*7+7]),np.std(buf[5+6*i:5+i*6+6])
   
    obdata=np.vstack([gwt1,gwt2,gwt3])
    np.savetxt('data/txtdata/fring_diff_cav_RA1.txt',obdata)
    
    ax=plt.subplot(335)
    plt.text(0,41,"(c)",fontsize=14)#
    plt.title('RA1',fontsize=8)
    ax.spines['top'].set_color('None')
    ax.spines['right'].set_color('None')
    marker_buf=['^','s','o','p','d']
    #measured res
    sim_data=np.load('data/surface_curve_simres.npy')
    sim_fr_cav=[]
    for ch in [1]: 
       for k in range(len(PFs)): 
           
           preds=np.zeros(len(rads))
           for i in range(len(rads)):preds[i]=np.sum(sim_data[ch][k][i][1][:]==0.04)
           
           plt.plot(curves,preds,mysim.colors[ch],label=str(PFs[k])+' N',
                    marker=marker_buf[k],markerfacecolor='none',markersize=6)  
           obser=obdata[k*len(rads):(k+1)*len(rads),0]
           plt.plot(curves,obser,'gray',
                    marker=marker_buf[k],markerfacecolor='none',markersize=6)  
           outputd.append(np.array([obser,preds]).T)
           stats.ranksums(obser,preds)
           sim_fr_cav.append(preds)

    np.save('data/sim_fr_cav_RA1.npy',sim_fr_cav)
    plt.yticks([0,10,20,30,40],fontsize=6)  
    plt.xticks([0,100,200,300,400,500,600,700],fontsize=6)
    plt.xlabel('Curvature [$\mathrm{m}^{-1}$] ',fontsize=8)  
    plt.ylabel('Response impulses in 1s',fontsize=8)   
    plt.legend(loc=0,prop={'family':'simSun','size':7}) 

def plot_impulses_response_as_func_of_distance_SA1():
    obdata=np.loadtxt('data/txtdata/fring_diff_distances.txt')
    ax=plt.subplot(324)
    plt.text(-1,122,"(c)",fontsize=14)#
    plt.title('SA1',fontsize=8)
    ax.spines['top'].set_color('None')
    ax.spines['right'].set_color('None')
    marker_buf=['^','s','o','p','d','>','<']
    sim_data=np.load('data/surface_curve_Rf_dis_simres.npy') 
    sim_fr_dis=[]
    for ch in range(1): 
       for k in range(1,len(rads)): 
           preds=np.zeros(len(distances))
           for i in range(len(distances)):preds[i]=np.sum(sim_data[ch][k][i][1][:]==0.04)
           plt.plot(distances,preds,mysim.colors[ch],label=str(curves[k])+' $\mathrm{m}^{-1}$',
                    marker=marker_buf[k],markerfacecolor='none',markersize=6) 
           obser=obdata[k*len(distances):(k+1)*len(distances),1]
           plt.plot(distances,obdata[k*len(distances):(k+1)*len(distances),1],'gray',
                    marker=marker_buf[k],markerfacecolor='none',markersize=6)  
    
           stats.ranksums(obser,preds)
           sim_fr_dis.append(preds)
           outputd.append(np.array([obser,preds]).T)
           
    np.save('data/sim_fr_dis.npy',sim_fr_dis)
    plt.yticks([0,40,80,120],fontsize=6)  
    plt.xticks([-1,0,1,2,3,4],fontsize=6)
    plt.xlabel('Y Distance [$\mathrm{mm}$] ',fontsize=8)  
    plt.ylabel('Response impulses in 1s',fontsize=8)   
    plt.legend(loc=0,prop={'family':'simSun','size':6}) 
        
def plot_prediction_relevance():
   plotlabels1=[str(PFs[0]),str(PFs[1]),str(PFs[2])]
   plotlabels2=[str(curves[0]),
                str(curves[1]),
                str(curves[2]),
                str(curves[3]),
                str(curves[4]),
                str(curves[5]),
                str(curves[6]),
                ]
   
   ticks=[[0,40,80,120],[0,10,20,30],[0,40,80,120]]
  
   obdata=[np.loadtxt('data/txtdata/fring_diff_cav.txt')[:,1],
           np.loadtxt('data/txtdata/fring_diff_cav_RA1.txt')[:,0],
           np.loadtxt('data/txtdata/fring_diff_distances.txt')[:,1]] 
   
   simdata=[np.load('data/sim_fr_cav_SA1.npy'),
            np.load('data/sim_fr_cav_RA1.npy'),
            np.load('data/sim_fr_dis.npy')]
   for ch in range(1): 
       if(ch==0):ax=plt.subplot(3,2,5)
       else:ax=plt.subplot(3,2,6)
       #if(ch==0): plt.text(-2,160,'(e)',fontsize=14)
       ax.spines['top'].set_color('None')
       ax.spines['right'].set_color('None')
       
       plt.xlabel('Predicted IPS',fontsize=8)
       plt.ylabel('Observed IPS',fontsize=8)
       R2s=np.zeros(len(PFs))
       for sigt in range(len(PFs)):
           x=simdata[ch][sigt][:]
           y=obdata[ch][sigt*len(rads):(sigt+1)*len(rads)]
           pvalue=round(stats.ranksums(x, y)[1],3)
           print(pvalue)
           fit=alt.curve_fit(x,y)
           R2="{0:.3f}".format(alt.R2(fit[2],y))
           R2s[sigt]=R2
           plt.scatter(x,y,marker=mysim.markers[sigt],
                       color='w',edgecolors=mysim.otc1[sigt],s=20,label=plotlabels1[sigt]+'N, $\mathrm{R}^{2}$='+str(R2)) #+' P='+str(pvalue)     
           plt.plot(fit[0],fit[1],'--',color=mysim.otc1[sigt])
           plt.legend(loc=1,fontsize=6,edgecolor='k')

       plt.xticks(ticks[ch],fontsize=6)
       plt.yticks(ticks[ch],fontsize=6)
       plt.legend(fontsize=6,edgecolor='gray')
       
   '-----Fr changing with distance under diff cavature'
   for ch in [2]: 
       ax=plt.subplot(3,2,6)
       ax.spines['top'].set_color('None')
       ax.spines['right'].set_color('None')

       plt.xlabel('Predicted IPS',fontsize=8)
       plt.ylabel('Observed IPS',fontsize=8)

       R2s=np.zeros(len(rads))
       for sigt in range(1,len(rads)):
           x=simdata[ch][sigt-1][:]
           y=obdata[ch][sigt*len(distances):(sigt+1)*len(distances)]
           pvalue=round(stats.ranksums(x, y)[1],3)
           print(pvalue)

           fit=alt.curve_fit(x,y)
           R2="{0:.3f}".format(alt.R2(fit[2],y))
           R2s[sigt]=R2
           plt.scatter(x,y,marker=mysim.markers[sigt],
                       color='w',edgecolors=mysim.otc1[sigt],s=20,label=plotlabels2[sigt]+'$\mathrm{m}^{-1}$, $\mathrm{R}^{2}$='+str(R2)) #+' P='+str(pvalue)      
           plt.plot(fit[0],fit[1],'--',color=mysim.otc1[sigt])
           plt.legend(loc=1,fontsize=6,edgecolor='k')

       plt.xticks(ticks[2],fontsize=6)
       plt.yticks(ticks[2],fontsize=6)

       plt.legend(fontsize=6,edgecolor='gray')

    
fig=plt.figure(figsize=(8,8.5)) 
plt.subplots_adjust(hspace=0.4)
plt.subplots_adjust(wspace=0.6)
plot_eps_under_differnt_curvature()
plot_impulses_response_as_func_of_curvature_for_SA1()
A1=outputd[0]
for i in range(2):A1=np.hstack([A1,outputd[i+1]])

outputd=[]
plot_impulses_response_as_func_of_distance_SA1()
A2=outputd[0]
for i in range(5):A2=np.hstack([A2,outputd[i+1]])

plot_prediction_relevance()
plt.savefig('saved_figs/submitting/res_curve_surface.png',bbox_inches='tight', dpi=300) 
