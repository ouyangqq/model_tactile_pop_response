import Receptors as rslib
import numpy as np
import matplotlib.pyplot as plt
import simset as mysim 
import img_to_eqstimuli as imeqst

filepath='saved_figs/'
pos_buf=np.load('data/pos_buf.npy')    
vpos_buf=np.load('data/vpos_buf.npy') 
  
srates=np.array([0.3,1,2,3])

ttype=['SA1','RA1','PC']
pins=[9,100,400]
np.meshgrid()
densities=np.array([100,200,300,400,500,600])
repeats=3
Tc_buf=np.zeros([len(ttype),len(pins),len(srates),len(densities),repeats])


'''
'................run the simulation................'
for ch in range(len(ttype)):
    for pin_c in range(len(pins)):
        for srate_c in range(len(srates)):
            for den_c in range(len(densities)):   
                for repeat_c in range(repeats):
                    setstimuli=[]
                    tsensor=rslib.tactile_receptors(Ttype=ttype[ch])
                    mysim.sim_onetype_setup(tsensor,mysim.skinroi[0],sT=1,sr=srates[srate_c]*1000,Density=densities[den_c])          
                    pimage=np.random.uniform(0.5,1,[int(np.sqrt(pins[pin_c])),int(np.sqrt(pins[pin_c]))])
                    Aeeps=imeqst.constructing_equivalent_probe_stimuli_from_pimage(pimage,1,1,mysim.fingertiproi)
                    DP=np.ones([tsensor.t.size,1])*30*9.8*1e-3
                    ips=[np.hstack([0.5*np.ones([tsensor.t.size,1]),
                                    0.5*np.ones([tsensor.t.size,1]),
                                    DP]),'Depth']
                    Tc_buf[ch,pin_c,srate_c,den_c,repeat_c]=tsensor.population_simulate(EEQS=Aeeps,Ips=ips,acquire_spikes=False)[3]
                    del tsensor
np.save('data/timeliness_buf_new.npy',Tc_buf) 
#-------------------------------------------------------------------
'''

Tc_buf=np.load("data/timeliness_buf_new.npy")
Tc_avg=np.zeros(Tc_buf[:,:,:,:,0].shape)  
Tc_std=np.zeros(Tc_buf[:,:,:,:,0].shape)  


color_bf=[['aquamarine','cornflowerblue','lightcoral'],
          ['springgreen','b','deeppink'],
          ['g','darkblue','r']]
marker_buf=['.-','^-','s-','o-','p-','d-'] 
marker_buf1=['.','^','s','o','p','d'] 
type_buf=['SA1','RA1','PC']
titilebuf=['(a) ','(b) ','(c) ','(d) Overall']


#---convert density into setting number-----#
units_numbuf=np.zeros(len(densities))
for j in range(len(units_numbuf)):
    tsensor=rslib.tactile_receptors(Ttype='SA1')
    mysim.sim_onetype_setup(tsensor,mysim.skinroi[0],sT=0.1,sr=1000,Density=densities[j])
    units_numbuf[j]=tsensor.Rm
    del tsensor

Tn_avg=np.zeros(Tc_buf[:,:,:,0,0].shape)  
nrange=np.arange(10,3000,1)
 
def plot_f1(clolor_buf,marker_buf):
    for j in range(len(ttype)):
        plt.subplot(2,2,j+1) 
        plt.subplots_adjust(hspace=0.3)
        plt.title(titilebuf[j]+type_buf[j], fontsize=14)
        plt.plot(units_numbuf,np.ones(len(units_numbuf)),'k--',linewidth=1)
        for k in range(len(pins)):
            for i in range(srates.size):
                Tc_avg[j,k,i,:]=np.mean(Tc_buf[j,k,i,:,:],axis=1)
                Tc_std[j,k,i,:]=np.std(Tc_buf[j,k,i,:,:],axis=1)
                plt.errorbar(np.uint16(units_numbuf),Tc_avg[j,k,i,:],yerr=Tc_std[j,k,i,:],
                             label='['+str(pins[k])+','+str(srates[i])+']',
                             color=clolor_buf[k][j],marker=marker_buf[i],
                             linewidth=1,markersize=5)     
        plt.xticks(np.uint16(units_numbuf),fontsize=8)  
        plt.yticks(np.arange(0,7,1))
        plt.xlabel(u"Number of tactile units ", fontsize=10)
        plt.ylabel(u"Simulation time (s)", fontsize=10)
        plt.legend(loc=0,ncol=3,prop={'family':'simSun','size':8}) 
  
def plot_f2(clolor_buf,marker_buf):
    for j in range(len(ttype)):
        for k in range(len(pins)):
            for i in range(srates.size):
                #---computing quadratic fitting curve to evaluate the MNT 
                coef2=np.polyfit(units_numbuf,Tc_avg[j,k,i,:],1) 
                yp=np.abs(np.poly1d(coef2)(nrange)-1)
                Tn_avg[j,k,i]=nrange[np.where(yp==np.min(yp))[0]]
            plt.plot(srates,Tn_avg[j,k],label=str(ttype[j])+','+str(pins[k]),color=clolor_buf[k][j],marker=marker_buf[i],linewidth=1,markersize=5)     
    plt.xticks(srates)
    plt.yticks([0,500,1000,1500,2000,2500,3000])
    avg_all=[]
    for i in range(srates.size):
       avg_all.append(np.average(Tn_avg[:,:,i]))
    
    plt.xlabel(u"Simpling rate (kHz)", fontsize=10)
    plt.ylabel(u"MNTARS", fontsize=12)
    plt.legend(loc=0,ncol=3,prop={'family':'simSun','size':8}) 
  

plt.figure(figsize=(10.5,7.5))
plot_f1(color_bf,marker_buf1)

plt.subplot(2,2,4)
plt.title(titilebuf[3], fontsize=14)
plot_f2(color_bf,marker_buf1)

plt.savefig('saved_figs/submitting/computation_efficiency.png',bbox_inches='tight', dpi=300)

