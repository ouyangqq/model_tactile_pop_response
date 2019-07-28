# -*- coding: utf-8 -*-
"""
Created on Sat Nov 18 17:11:16 2017
@author: Ouyang qiangqiang
"""
import Receptors as rslib
import numpy as np
import matplotlib.pyplot as plt
import simset as mysim 
from PIL import Image
import img_to_eqstimuli as imeqst

width=120#mm
height=12#mm

shift=0.05
speed=20
pf=0.35#pressing force (N)
simT=width/speed
simdt=0.001
doth=6

Ttype_buf=['SA1','RA1','PC']
tsensors=[]
pbuf=np.load('data/loc_pos_buf_fingertip.npy')    
for tp in range(len(Ttype_buf)):
    tsensor=rslib.tactile_receptors(Ttype=Ttype_buf[tp])
    tsensor.set_population(pbuf[tp][0],pbuf[tp][1],simTime=simT,sample_rate=1/simdt,Density=pbuf[tp][2],roi=mysim.fingertiproi) 
    tsensors.append(tsensor)

img1 =Image.open('saved_figs/letters_120-12.jpg')


'delete the "..." above and below if want to run the simulation again'
buf1=imeqst.constructing_equivalent_probe_stimuli_from_image(img1,width,height,mysim.fingertiproi)
np.save('data/forms_letters.npy',buf1)
PF=pf*np.ones(int(simT/simdt))
ips=imeqst.img_stimuli_scaning_with_uniformal_speed(simdt,simT,PF,speed,0,width,0,height,shift)
simulation_res=[]
Aeeps=np.load('data/forms_letters.npy')[2]
Aeeps[1]=Aeeps[1]*doth # height of embossed letter is 6 mm according to ref. [35]
for tp in range(len(Ttype_buf)):
    tmp=[]
    for row in range(len(ips)):#(len(stimuli_buf1)):
        tsensors[tp].population_simulate(EEQS=Aeeps,Ips=[ips[row],'Pressure'],noise=0)
        sel=tsensors[tp].points_mapping_entrys(np.array([[0,0]]))[0]
        tmp.append(np.array(tsensors[tp].Va[sel,:]))
    simulation_res.append(tmp)
np.save('data/letters_simulation_res_single_repeat.npy',simulation_res)   
#----------------------

num=int(height/shift)
sel_points=np.vstack([0*np.ones(num),np.linspace(-height/2,height/2,num)]).T #ms

def print_ob_letter_spiking_trians():
    plt.figure(figsize=(8,1*0.8))
    ax=plt.subplot(1,1,1)
    plt.text(-0,-30,"(a)",fontsize=12)#
    labelsx=np.round(np.linspace(0,width,7))
    labelsy=np.round(np.linspace(height,0,7))
    obimg=np.array(Image.open('saved_figs/ob_letters_80-10.jpg').convert('L'))
    ax.spines['top'].set_color('None')
    ax.spines['right'].set_color('None')
    ax.imshow(obimg,cmap=plt.cm.gray,aspect='equal')
    plt.xlabel('Postion [mm]',fontsize=8) 
    plt.ylabel('Distance [mm]',fontsize=8) 
    plt.xticks(np.linspace(0,obimg.shape[1],7),labelsx,fontsize=6)
    plt.yticks(np.linspace(0,obimg.shape[0],7),labelsy,fontsize=6)
    plt.savefig('saved_figs/ob_letter_spking.png',bbox_inches='tight', dpi=400)


def print_letter_spiking_trians():
    ftiproi=np.loadtxt('data/txtdata/fingertip_roi.txt')
    ftiproi=np.vstack([ftiproi,ftiproi[0,:]])
    plt.figure(figsize=(8,4*0.8))
    buf=np.load('data/forms_letters.npy')[1]
    sres=np.load('data/letters_simulation_res_single_repeat.npy') 

    ax=plt.subplot(4,1,1)
    plt.text(-0,14,"(b)",fontsize=14)#
    #plt.scatter(buf[:,0],buf[:,1],c=mysim.colors[3],s=0.2)
    ax.scatter(buf[:,0],buf[:,1],s=0.5,c=1e3*buf[:,5],cmap=plt.cm.Greys,vmin=0,vmax=3)
    plt.xticks(np.arange(0,width+width/10,width/10),fontsize=6)
    plt.yticks([0,height/2,height],fontsize=7)
    ax.twinx() 
    plt.yticks([])
    plt.ylabel('EPS',fontsize=8)

    for ch in range(3):
        ax1=plt.subplot(4,1,ch+2,sharex=ax)
        plt.subplots_adjust(hspace=0.1)
        for i in range(num):
            #res=np.array(tbuf[i][:])
            res=simdt*np.where(sres[ch][i]==0.04)[0]
            plt.scatter(res*speed,sel_points[-1-i,1]*np.ones(len(res)),c=mysim.colors[ch],marker='.',s=0.005)
            plt.xticks(np.arange(0,width+width/10,width/10),fontsize=7)
            plt.yticks(fontsize=7)
        if(ch==2):plt.xlabel('Postion [mm]',fontsize=10) 
        if(ch==1):plt.ylabel('Distance [mm]',fontsize=10)   
        ax1.twinx() 
        plt.yticks([])
        plt.ylabel(Ttype_buf[ch],fontsize=8)
    plt.savefig('saved_figs/letter_spking.png',bbox_inches='tight', dpi=300)     
  
       
print_ob_letter_spiking_trians()
print_letter_spiking_trians()

img1=Image.open('saved_figs/ob_letter_spking.png')
img2=Image.open('saved_figs/letter_spking.png')

img=Image.new(img1.mode,(img2.size[0],img1.size[1]+img2.size[1]))

img.paste(img1,(80,0))
img.paste(img2,(0,img1.size[1]))
img.save("saved_figs/submitting/Form_all.png")
img.show()

