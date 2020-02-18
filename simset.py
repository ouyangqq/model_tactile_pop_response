# -*- coding: utf-8 -*-
"""
Created on Sat Nov 18 17:11:16 2017
@author: Administrator
"""
import os 
import sys
#from tactile_receptors import Receptors as receptorlib
import numpy as np
import ultils as alt
import Receptors as rslib
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.cm as cm
import random
#matplot set
#colors=['g','olive','orange','c']
colors=['g','b','m','c'] # color set for simulatinig each afferent type and stimulus

otcolors=['c','darkblue','brown'] # color set for simulatinig each afferent type and stimulus with other models

otc1=['olive','deepskyblue','brown','y','b','g','k']

markers=['^','s','o','p','d','>','<'] 

color_bf=[['palegreen','lawngreen','aquamarine','springgreen','g'],
          ['cornflowerblue','royalblue','b','mediumblue','darkblue'],
          ['mediumorchid','violet','magenta','m','purple']]



#Simulation setup
prope_d=rslib.Dbp #mm
# 自定义colormap
def colormap():
    return mpl.colors.LinearSegmentedColormap.from_list('cmap', ['#FFFFFF', '#98F5FF', '#00FF00', '#FFFF00','#FF0000', '#8B0000'], 256)
                                                             

fingertiproi=np.loadtxt('data/txtdata/fingertip_roi.txt')
oo=np.array([(fingertiproi[:,0].max()+fingertiproi[:,0].min())/2,
            (fingertiproi[:,1].max()+fingertiproi[:,1].min())/2])
fingertiproi=fingertiproi-oo

bfingerroi=np.loadtxt('data/txtdata/bfinger_roi.txt')
plamroi=np.loadtxt('data/txtdata/plam_roi.txt')
skinroi=[fingertiproi,bfingerroi,plamroi]


Ttype_buf=['SA1','RA1','PC']



def print_intermediate_signals(tsensor): 
    sel_point=[0,0]
    sel= tsensor.points_mapping_entrys(sel_point)
    sel_t=int(3.5/tsensor.dt)
    plt.figure(figsize=(8,8)) 
    plt.subplot(911)  
    plt.plot(tsensor.X0[sel,:sel_t])
    plt.subplot(912)  
    plt.plot(tsensor.X1[sel,:sel_t])
    plt.subplot(913)  
    plt.plot(tsensor.X2[sel,:sel_t])
    plt.subplot(914)  
    plt.plot(tsensor.Sm[sel,:sel_t])
    plt.subplot(915)  
    plt.plot(tsensor.Qg[sel,:sel_t])
    plt.subplot(916)  
    plt.plot(tsensor.Vg[sel,:sel_t])
    plt.subplot(917)  
    plt.plot(tsensor.Vc[sel,:sel_t])
    plt.subplot(918)  
    plt.plot(tsensor.Vf[sel,:sel_t])
    plt.subplot(919)  
    plt.plot(tsensor.Va[sel,:sel_t])
    
#A=loc_rt(fingertiproi,pending)
def Visualizing_covering_areas_of_tactile_units(tsensors):
    plt.figure(figsize=(12,9)) 
    plt.subplots_adjust(hspace=0)
    buf=[]
    buf.append(np.load('data/loc_pos_buf_fingertip.npy'))
    buf.append(np.load('data/loc_pos_buf_bfinger.npy') )
    buf.append(np.load('data/loc_pos_buf_plam.npy') )
    for area in range(1):
        for ch in range(3):
            tsensors[ch].set_population(buf[area][ch][0],buf[area][ch][1],
                    simTime=1,sample_rate=1000,
                    Density=buf[area][ch][2],roi=skinroi[area]) 
            #punit_buf=buf[area][ch][0]
            if(area==2): ax=plt.subplot(2,3,ch+4)
            else: ax=plt.subplot(3,6,area*3+ch+1)
            ax.spines['top'].set_color('None')
            ax.spines['right'].set_color('None')
            plt.title(Ttype_buf[ch],fontsize=10)
            
            SC=np.ones([tsensors[ch].Nrr,tsensors[ch].Nrc,3])
            ES1=tsensors[ch].Es[0]
            ES2=tsensors[ch].Es[1]
            rands=np.random.choice(200,[tsensors[ch].Rm,3])
            for st in range(tsensors[ch].Rm):
                SC[SC.shape[0]-1-ES1[st,:],ES2[st,:],:]=rands[st,:]
                
            #SC[SC.shape[0]-tsensors[ch].pdots[:,0]-1,tsensors[ch].pdots[:,1]]=100    
            plt.imshow(SC,aspect='auto')#cmap=cm.gray,
            
            sroi=np.array(skinroi[area])#*rslib.rtm(-np.pi)
            sroi=np.vstack([skinroi[area],skinroi[area][0,:]])
            DD=sroi[:,0].max()-sroi[:,0].min()
            x=(sroi[:,0]-sroi[:,0].min())/DD*SC.shape[1]
            DD=sroi[:,1].max()-sroi[:,1].min()
            y=SC.shape[0]-(sroi[:,1]-sroi[:,1].min())/DD*SC.shape[0]
            plt.plot(x,y-1,'yellow',linewidth=2.5)
            plt.scatter(tsensors[ch].OEs[:,1],SC.shape[0]-1-tsensors[ch].OEs[:,0],s=2,
                       label='$\mathrm{D}_{a}$='+str(round(buf[area][ch][2],1)),marker='.',c='k')
        
            labelsy=np.int16(np.linspace(skinroi[area][:,1].max(),skinroi[area][:,1].min(),6))
            labelsx=np.int16(np.linspace(skinroi[area][:,0].min(),skinroi[area][:,0].max(),6))
            plt.xticks(np.linspace(0,SC.shape[1],6),labelsx,fontsize=8)
            plt.yticks(np.linspace(0,SC.shape[0],6),labelsy,fontsize=8)

            plt.legend(prop={'family':'simSun','size':8},loc = 1) 
    plt.savefig('saved_figs/sampled_areas.png',bbox_inches='tight', dpi=300) 


def Visualizing_large_covering_areas_of_tactile_units(tsensors, buf):
    plt.figure(figsize=(5.5,10)) 
    plt.subplots_adjust(hspace=0)
    for area in range(1):
        for ch in range(1):
            tsensors[ch].set_population(buf[area][ch][0],buf[area][ch][1],
                    simTime=1,sample_rate=1000,
                    Density=buf[area][ch][2],roi=skinroi[area]) 
            ax=plt.subplot(1,1,ch+1)
            ax.spines['top'].set_color('None')
            ax.spines['right'].set_color('None')
            ax.spines['bottom'].set_color('None')
            ax.spines['left'].set_color('None')
            plt.title(Ttype_buf[ch],fontsize=10)
            
            SC=np.ones([tsensors[ch].Nrr,tsensors[ch].Nrc,3])
            ES1=tsensors[ch].Es[0]
            ES2=tsensors[ch].Es[1]
            rands=np.random.choice(200,[tsensors[ch].Rm,3])
            
            #rands1=random.sample(range(0,tsensors[ch].Rm),tsensors[ch].Rm)*2
            #rands2=random.sample(range(0,tsensors[ch].Rm),tsensors[ch].Rm)*2
            #rands3=random.sample(range(0,tsensors[ch].Rm),tsensors[ch].Rm)*2
            #rands=np.vstack([rands1,rands2,rands3]).T

            for st in range(tsensors[ch].Rm):
                SC[SC.shape[0]-1-ES1[st,:],ES2[st,:],:]=rands[st,:]
                
            #SC[SC.shape[0]-tsensors[ch].pdots[:,0]-1,tsensors[ch].pdots[:,1]]=100    
            plt.imshow(SC,aspect='auto')#cmap=cm.gray,
            
            sroi=np.array(skinroi[area])#*rslib.rtm(-np.pi)
            sroi=np.vstack([skinroi[area],skinroi[area][0,:]])
            DD=sroi[:,0].max()-sroi[:,0].min()
            x=(sroi[:,0]-sroi[:,0].min())/DD*SC.shape[1]
            DD=sroi[:,1].max()-sroi[:,1].min()
            y=SC.shape[0]-(sroi[:,1]-sroi[:,1].min())/DD*SC.shape[0]
            plt.plot(x,y,'y-',linewidth=6,c='y')
            plt.scatter(tsensors[ch].OEs[:,1],SC.shape[0]-1-tsensors[ch].OEs[:,0],s=50,
                       label='D='+str(round(buf[area][ch][2],1)),marker='.',c='k')
        
            labelsy=np.int16(np.linspace(skinroi[area][:,1].max(),skinroi[area][:,1].min(),6))
            labelsx=np.int16(np.linspace(skinroi[area][:,0].min(),skinroi[area][:,0].max(),6))
            plt.xticks(np.linspace(0,SC.shape[1],6),labelsx,fontsize=8)
            plt.yticks(np.linspace(0,SC.shape[0],6),labelsy,fontsize=8)

            #plt.legend(prop={'family':'simSun','size':8},loc = 1) 
    plt.savefig('saved_figs/large_sampled_areas.png',bbox_inches='tight', dpi=300) 

def visualizating_resistance_matrix(dt):
    plt.figure(figsize=(7.5,6)) 
    ax=plt.subplot(1,1,1)
    img=dt[0][0][3]
    md=ax.imshow(img,aspect='auto',cmap=plt.cm.bwr,vmin=-2,vmax=2)
    cbar=plt.colorbar(md)
    plt.rcParams['font.size'] = 12
    plt.savefig('saved_figs/resistance_matrix_img.png',bbox_inches='tight', dpi=300) 
   

# stimulus shape construction based on probe dots
def constructing_probe_stimuli(Sps):
    rad=Sps[:,2].max()
    Wc=Sps[:,0].max()-Sps[:,0].min()+2*rad
    Hc=Sps[:,1].max()-Sps[:,1].min()+2*rad
    Nc=int(Wc/prope_d)+1
    Nr=int(Hc/prope_d)+1
    pimage=np.zeros([Nr,Nc])
    
    OEs=np.hstack([np.uint16((Sps[:,1:2]-Sps[:,1].min()+rad)/Hc*Nr),
                   np.uint16((Sps[:,0:1]-Sps[:,0].min()+rad)/Wc*Nc)])
    
    for i in range(len(Sps)):
        #for j in range(Nrcp):
        RSN=int(Sps[i,2]/prope_d)
        row=np.arange(OEs[i,0]-RSN,OEs[i,0]+RSN+1,1)
        col=np.arange(OEs[i,1]-RSN,OEs[i,1]+RSN+1,1)
        pendings=np.meshgrid(row,col)
        sel=(pendings[0]-OEs[i,0])**2+(pendings[1]-OEs[i,1])**2<=RSN**2
        entrys=[pendings[0][sel],pendings[1][sel]]
        
        pimage[entrys[0],entrys[1]]=Sps[i,3]
         
    'equvilent stimuli dots'
    w=Wc
    h=Hc
    selimg=pimage
    dots=np.meshgrid(np.linspace(0,w,selimg.shape[1]),np.linspace(h,0,selimg.shape[0]))
    x=dots[0].reshape(dots[0].size,1)
    y=dots[1].reshape(dots[1].size,1)
    th=selimg.reshape(selimg.size,1)
    tmp=np.hstack([x,y,th])
    pins=tmp[tmp[:,2]>0.1,:]
    x,y,th=pins[:,0:1],pins[:,1:2],pins[:,2:3]
    eq_stimuli=np.hstack([x,y,x*0,x*0,np.ones([x.size,1])*prope_d*1e-3,th*1e-3])
    return [[pimage,Wc,Hc],eq_stimuli]

# stimulus shape construction based on probe dots
def constructing_spherically_curved_surfaces_probes(Sps):
    rad=Sps[:,2].max()
    Wc=Sps[:,0].max()-Sps[:,0].min()+2*rad
    Hc=Sps[:,1].max()-Sps[:,1].min()+2*rad
    Nc=int(Wc/prope_d)
    Nr=int(Hc/prope_d)
    pimage=np.zeros([Nr,Nc])
    
    OEs=np.hstack([np.uint16((Sps[:,1:2]-Sps[:,1].min()+rad)/Hc*Nr),
                   np.uint16((Sps[:,0:1]-Sps[:,0].min()+rad)/Wc*Nc)])
    
    for i in range(len(Sps)):
        #for j in range(Nrcp):
        RSN=int(Sps[i,2]/prope_d)
        row=np.arange(OEs[i,0]-RSN,OEs[i,0]+RSN,1)
        col=np.arange(OEs[i,1]-RSN,OEs[i,1]+RSN,1)
        pendings=np.meshgrid(row,col)
        sel=(pendings[0]-OEs[i,0])**2+(pendings[1]-OEs[i,1])**2<RSN**2
        
        entrys=[pendings[0][sel],pendings[1][sel]] # select basic dot locating in circle area
        
        dist=np.sqrt((pendings[0][sel]-OEs[i,0])**2+(pendings[1][sel]-OEs[i,1])**2)
        
        
        hvals=np.sqrt(np.abs(RSN**2-dist**2))*prope_d# calulate the height values
        
        pimage[entrys[0],entrys[1]]=hvals#Sps[i,3]
         
    'equvilent stimuli dots'
    selimg=pimage
    dots=np.meshgrid(np.linspace(0,Wc,selimg.shape[1]),np.linspace(Hc,0,selimg.shape[0]))
    x=dots[0].reshape(dots[0].size,1)
    y=dots[1].reshape(dots[1].size,1)
    th=selimg.reshape(selimg.size,1)
    tmp=np.hstack([x,y,th])
    pins=tmp[tmp[:,2]>0,:]
    x,y,th=pins[:,0:1],pins[:,1:2],pins[:,2:3]
    eq_stimuli=np.hstack([x,y,x*0,x*0,np.ones([x.size,1])*prope_d*1e-3,th*1e-3])
    return [[pimage,Wc,Hc],eq_stimuli]
# stimulus shape construction based on probe dots
def get_probe_stimuli_dots(sites,rad):
    prope_d=0.4   
    tmp=[]
    for lr in np.arange(np.min(sites[:,1])-2*rad,np.max(sites[:,1])+2*rad,prope_d):
        for lc in np.arange(np.min(sites[:,0])-2*rad,np.max(sites[:,0])+2*rad,prope_d):
            for i,pin in enumerate(sites):
                if(((lc-pin[0])**2+(lr-pin[1])**2)<rad**2):
                    tmp.append(np.array([lc,lr,0,i]))
    shape_equal_dots=np.zeros([len(tmp),4])          
    for j in range(len(tmp)): 
        shape_equal_dots[j,:]=tmp[j]
    unitdots=np.hstack([shape_equal_dots,
                        prope_d/2*1e-3*np.ones([len(shape_equal_dots),1])])
    return unitdots


def receptors_build_in_a_skin_area(arearoi,sT=1,sr=1000,thea=0,Densitys=[165,210,40]):
    save_loc_data=[]
    for tp in range(len(Ttype_buf)):
        Mt=np.mat([[np.cos(-thea),np.sin(-thea)],[-np.sin(-thea),np.cos(-thea)]]);
        Ds=10/np.sqrt(Densitys[tp])
        row=0  #row plus
        num=100
        xmin,xmax,ymin,ymax=arearoi[:,0].min(),arearoi[:,0].max(),arearoi[:,1].min(),arearoi[:,1].max()
        buf=np.meshgrid(np.arange(int(xmin/Ds),int(xmax/Ds)+1,1),
                        np.arange(int(ymin/Ds),int(ymax/Ds)+1,1))
        num=buf[0].size
        locs=np.hstack([buf[0].reshape(num,1)*Ds,buf[1].reshape(num,1)*Ds])
        locs=np.array((locs+np.random.uniform(-1,1,locs.shape)/10)*Mt)
        entrys=np.hstack([buf[0].reshape(num,1),buf[1].reshape(num,1)])
        sroi=np.vstack([arearoi,arearoi[0,:]])
        sel=rslib.isPoisWithinPoly(sroi,locs)
  
        punit_buf=locs[sel,:]
        grid_buf=entrys[sel,:]
        
        save_loc_data.append([punit_buf,grid_buf])
    return save_loc_data 

def sim_all_setup(tsensor_buf,arearoi,sT=1,sr=1000,Densitys=[165,210,40]):
    save_loc_data=[]
    for tp in range(len(Ttype_buf)):
        #--------------SA1--------------#
        Mt=tsensor_buf[tp].Mt
        Ds=10/np.sqrt(Densitys[tp])
        num=100
        dmax=np.sqrt(arearoi[:,0]**2+arearoi[:,1]**2).max()
        buf=np.meshgrid(np.arange(-int(dmax/Ds),int(dmax/Ds)+1,1),
                        np.arange(-int(dmax/Ds),int(dmax/Ds)+1,1))
        num=buf[0].size
        locs=np.hstack([buf[0].reshape(num,1)*Ds,buf[1].reshape(num,1)*Ds])
        locs=np.array((locs+np.random.uniform(0,0,locs.shape)/5)*Mt)
        entrys=np.hstack([buf[0].reshape(num,1),buf[1].reshape(num,1)])

        sroi=np.vstack([arearoi,arearoi[0,:]])
        sel=rslib.isPoisWithinPoly(sroi,locs)
        punit_buf=locs[sel,:]
        grid_buf=entrys[sel,:]

        tsensor_buf[tp].set_population(punit_buf,grid_buf,simTime=sT,sample_rate=sr,Density=Densitys[tp],roi=arearoi) 

        save_loc_data.append([punit_buf,grid_buf,Densitys[tp],tsensor_buf[tp].G])
    return save_loc_data 
    
def sim_onetype_setup(tsensor,arearoi,sT=1,sr=1000,Density=100):
    #--------------SA1--------------#
    Mt=tsensor.Mt
    Ds=10/np.sqrt(Density)
    dmax=np.sqrt(arearoi[:,0]**2+arearoi[:,1]**2).max()
    buf=np.meshgrid(np.arange(-int(dmax/Ds),int(dmax/Ds)+1,1),
                        np.arange(-int(dmax/Ds),int(dmax/Ds)+1,1))
    num=buf[0].size
    locs=np.hstack([buf[0].reshape(num,1)*Ds,buf[1].reshape(num,1)*Ds])
    locs=np.array((locs+np.random.uniform(0,0,locs.shape)/5)*Mt)
    entrys=np.hstack([buf[0].reshape(num,1),buf[1].reshape(num,1)])

    sroi=np.vstack([arearoi,arearoi[0,:]])
    sel=rslib.isPoisWithinPoly(sroi,locs)
    punit_buf=locs[sel,:]
    grid_buf=entrys[sel,:]
    
    tsensor.set_population(punit_buf,grid_buf,simTime=sT,sample_rate=sr,roi=arearoi) 





'''
Ttype_buf=['SA1','RA1','PC']
tsensors=[]
pbuf=np.load('data/loc_pos_buf_fingertip.npy')    
for tp in range(len(Ttype_buf)):
    tsensor=rslib.tactile_receptors(Ttype=Ttype_buf[tp])
    tsensor.set_population(pbuf[tp][0],pbuf[tp][1],simTime=1,sample_rate=1000,Density=pbuf[tp][2],roi=fingertiproi) 
    print('Number of '+Ttype_buf[tp]+' in fingertip is',tsensor.Rm)
    tsensors.append(tsensor)    


Visualizing_covering_areas_of_tactile_units(tsensors)
'''
'''
df=[np.load('data/loc_pos_buf_fingertip.npy'),[],[]]
visualizating_resistance_matrix(df)

'------------------------'
Densitys1=np.array([[72.2,143.5,24.8],[32.5,40.9,11.3],[9.5,26.3,11.1]])
#Densitys2=np.array(Densitys1)*1.5
Densitys3=[[16,24,8],[15,20,6],[10,16,5]]

tsensors=[]
for tp in range(len(Ttype_buf)):
    tsensor=rslib.tactile_receptors(Ttype=Ttype_buf[tp])
    tsensors.append(tsensor)

save_loc_data=sim_all_setup(tsensors,fingertiproi,sT=1,sr=1000,Densitys=Densitys1[0])
np.save('data/loc_pos_buf_fingertip.npy',save_loc_data)   
save_loc_data=sim_all_setup(tsensors,bfingerroi,sT=1,sr=1000,Densitys=Densitys1[1])   
np.save('data/loc_pos_buf_bfinger.npy',save_loc_data)  
save_loc_data=sim_all_setup(tsensors,plamroi,sT=1,sr=1000,Densitys=Densitys1[2]) 
np.save('data/loc_pos_buf_plam.npy',save_loc_data)  


dt1=sim_all_setup(tsensors,fingertiproi,sT=1,sr=1000,Densitys=Densitys3[0])
dt2=sim_all_setup(tsensors,bfingerroi,sT=1,sr=1000,Densitys=Densitys3[1])    
dt3=sim_all_setup(tsensors,plamroi,sT=1,sr=1000,Densitys=Densitys3[2])  
np.save('data/loc_pos_low_density.npy',[dt1,dt2,dt3])  
Visualizing_large_covering_areas_of_tactile_units(tsensors,[dt1,dt2,dt3])

'''
