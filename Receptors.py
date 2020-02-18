#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
Simulation of a tactile receptors in skin
'''        
import numpy as np
import time as timec
from scipy import signal
from skimage import transform
from PIL import Image
sp_T=0#ms
fire_rate_max=200 #spikes/s #ms
actionp_plus=(30-(-70)) # -70 reset_p
actionp_minus=((-70)-(-100))
Dbp=0.2#mm  diameter of basic probe

class tactile_receptors():
    def __init__(self,Ttype=' ',Pls=0): #whether load fitting parameters 1:Yes
        self.Ttype=Ttype
        '-----skin mechanics-----'
        self.poisson_v=0.4
        self.E_ym=50*1e3  #pa,Young modulus 
        '-----neuron electronic-----'
        self.v_reset=-65*1e-3 #v
        self.VL=15*1e-3  #v
        paras_dict=np.load('data/fitting_paras.npy').item()
        if(self.Ttype=="SA1"):
            self.thea=3*np.pi/4 #3*np.pi/4 
            self.Cd=paras_dict['SA1']['Cd']
            self.Rwv=1
            self.Rc=paras_dict['SA1']['Rc']       
            self.Ku=paras_dict['SA1']['Ku']
            self.Kb1=paras_dict['SA1']['Kb1']
            self.Kb2=paras_dict['SA1']['Kb2']
            self.Q=paras_dict['SA1']['Q']
            self.wb=2*np.pi*paras_dict['SA1']['fb']
            self.wl=2*np.pi*paras_dict['SA1']['fl']
            self.Cs=paras_dict['SA1']['Cs']
            self.Cp=0
            self.Ce=paras_dict['SA1']['Ce']
            self.Kf=paras_dict['SA1']['Kf']
            self.Nds=1
            self.maxfr=120
        elif (self.Ttype=="RA1"):
            self.thea=3*np.pi/4  
            self.Cd=paras_dict['RA1']['Cd']
            self.Rwv=1
            self.Rc=paras_dict['RA1']['Rc']
            self.Ku=0
            self.Kb1=paras_dict['RA1']['Kb1']
            self.Kb2=paras_dict['RA1']['Kb2']
            self.Q=paras_dict['RA1']['Q']
            self.wb=2*np.pi*paras_dict['RA1']['fb']
            self.wl=2*np.pi*0
            self.Cs=paras_dict['RA1']['Cs']
            self.Cp=paras_dict['RA1']['w']
            self.Kf=paras_dict['RA1']['Kf']
            self.Ce=paras_dict['RA1']['Ce']
            self.Nds=1
            self.maxfr=200
            
        elif (self.Ttype=="PC"):
            self.thea=3*np.pi/4 
            self.Cd=paras_dict['RA1']['Cd']
            self.Rwv=1
            self.Rc=paras_dict['PC']['Rc']
            self.Ku=0
            self.Kb1=0
            self.Kb2=paras_dict['PC']['Kb2']
            self.Q=paras_dict['PC']['Q']
            self.wb=2*np.pi*paras_dict['PC']['fb']
            self.wl=2*np.pi*0
            self.Cs=paras_dict['PC']['Cs']
            self.Cp=paras_dict['PC']['w']
            self.Ce=paras_dict['PC']['Ce']
            self.Nds=2
            #self.Ta=0.004
            self.Kf=paras_dict['PC']['Kf']#37
            self.maxfr=300
        self.Mt=np.mat([[np.cos(-self.thea),np.sin(-self.thea)],[-np.sin(-self.thea),np.cos(-self.thea)]]);

    def set_population(self,punit_buf,grid_buf,simTime=1,sample_rate=1000,Density=150,youngs=50*1e3,roi=[]):  
        self.E_ym=youngs
        self.T=float (simTime) 
        self.dt = float (1/sample_rate)   #Timestep
        self.t= np.linspace(0,self.T,int( self.T/self.dt))  
        self.Rn=1
        self.Rm=len(grid_buf)
        self.stp=int(self.Rm/2)
        self.Density=Density
        self.V1=np.random.uniform(-0.000,0.000,(self.Rm*self.Rn,self.t.size))
        self.V2=np.random.uniform(-0.000,0.000,(self.Rm*self.Rn,self.t.size))
        self.G=np.mat(np.zeros((self.Rm*self.Rn,self.Rm*self.Rn)))
        self.Gi=np.mat(np.zeros((self.Rm*self.Rn,self.Rm*self.Rn))) 
        self.Va=np.random.uniform(-70,-70,(self.Rm*self.Rn,self.t.size))
        self.Vg=np.random.uniform(-0.000,0.000,(self.Rm*self.Rn,self.t.size))
        self.Is=np.random.uniform(-0.000,0.000,(self.Rm*self.Rn,self.t.size))
        self.Vs=np.zeros([self.Rm*self.Rn,self.t.size])
        self.Vnf=np.random.uniform(-0.000,0.000,(self.Rm*self.Rn,self.t.size))
        self.Vf=np.random.uniform(-0.000,0.000,(self.Rm*self.Rn,self.t.size))
        self.X0=np.zeros((self.Rm*self.Rn,self.t.size))
        self.VN=np.random.uniform(-10,10,(self.Rm*self.Rn,self.t.size)) 
        self.SN=np.random.uniform(-0,0,(self.Rm*self.Rn,self.t.size))   
        self.Uc=np.array(np.zeros((self.Rm*self.Rn,self.t.size)));   
        self.Dt=np.random.uniform(-0.000,0.000,(self.Rm*self.Rn,self.t.size))
        self.spike_trains=[]
        
        #-----Locations of receptors-----#
        self.r_pos=np.array(punit_buf);  
        tmp=grid_buf+100
        entry_buf=np.int16(tmp[:,0]+tmp[:,1]*(100*2))
        self.v_pos=entry_buf;
        for i in range(self.Rm):
            [w,v]=grid_buf[i,:]
            
            tmp=grid_buf==[w-1,v]
            sel=tmp[:,0]&tmp[:,1]
            if(np.sum(sel)):
                #loc=sel.nonzero()[0][0]
                self.G[sel,i]=-1
        
            tmp=grid_buf==[w+1,v]
            sel=tmp[:,0]&tmp[:,1]
            if(np.sum(sel)):
                #loc=sel.nonzero()[0][0]
                self.G[sel,i]=-1
                
            tmp=grid_buf==[w,v-1]
            sel=tmp[:,0]&tmp[:,1]
            if(np.sum(sel)):
                self.G[sel,i]=-1/self.Rwv
                
            tmp=grid_buf==[w,v+1]
            sel=tmp[:,0]&tmp[:,1]
            if(np.sum(sel)):
                self.G[sel,i]=-1/self.Rwv
            self.G[i,i]=2*(1+self.Rwv)/(self.Rwv)/self.Cd
        self.Gi=self.Rc*self.G.I
        self.GiFit=self.G.I  # inverse G for fitting
        
        '---Presetup---' 
        #size of skin contact scaning image  
        self.rentrys=[]
        Wc=roi[:,0].max()-roi[:,0].min()
        Hc=roi[:,1].max()-roi[:,1].min()
        
        self.Wc,self.Hc=Wc,Hc
        #self.Wc1,self.Wc2=roi[:,0].max(),Wc-roi[:,0].max()
        #self.Hc1,self.Hc2=roi[:,1].max(),Hc-roi[:,1].max()
        
        
        self.Nc=int(Wc/Dbp)
        self.Nr=int(Hc/Dbp)
        
        self.Nrc=int(Wc/Dbp/self.Nds)
        self.Nrr=int(Hc/Dbp/self.Nds)
        
        OEs=np.hstack([np.uint16((self.r_pos[:,1:2]-roi[:,1].min())/Hc*(self.Nrr)),
                       np.uint16((self.r_pos[:,0:1]-roi[:,0].min())/Wc*(self.Nrc))])
        self.OEs=OEs
        
        Esc=np.meshgrid(np.arange(0,self.Nrr,1),np.arange(0,self.Nrc,1))
        Esc=np.hstack([Esc[0].reshape(Esc[0].size,1),Esc[1].reshape(Esc[0].size,1)])
        
        locs=np.hstack([Esc[:,1:2]*Wc/self.Nrc+roi[:,0].min(),Esc[:,0:1]*Hc/self.Nrr+roi[:,1].min()])
        
        sroi=np.vstack([roi,roi[0,:]])
        
        sel=isPoisWithinPoly(sroi,locs)
        pdots=Esc[sel,:]  #Pixels in skin area
        self.pdots=pdots
        Esbuf=[]
        for i in range(self.Rm):
            Esbuf.append([]) 
            
        # select closet pxiels in SC image that attribute to each tactile unit
        for i in range(len(pdots)):
            tmp=np.float32(pdots[i,:])-np.float32(OEs)
            tmp=tmp[:,0]**2+tmp[:,1]**2
            sel=int(np.where(tmp==tmp.min())[0][0])
            Esbuf[sel].append(pdots[i,:])
            
        Nbuf=np.zeros(self.Rm)   
        for i in range(self.Rm):
            Nbuf[i]=len(Esbuf[i])
        Nrcp=int(np.max(Nbuf))    
        rows=np.zeros([self.Rm,Nrcp])
        cols=np.zeros([self.Rm,Nrcp])
        
        for i in range(self.Rm):
            'supply up to Nrcp pixel with center pixel for each tactile unit'
            num=len(Esbuf[i])
            if(num>0):
                eadots=np.array(Esbuf[i]).T
                supplys=np.ones([2,Nrcp-num])  
            else: 
                eadots=OEs[i,:].reshape(2,1)
                supplys=np.ones([2,Nrcp-1]) 
        
            supplys[0,:]=supplys[0,:]*OEs[i,0]
            supplys[1,:]=supplys[1,:]*OEs[i,1]
            rows[i,:]=np.hstack([eadots,supplys])[0,:]
            cols[i,:]=np.hstack([eadots,supplys])[1,:]
        self.Es=[np.uint16(rows),np.uint16(cols)]    
    

    def location_map(self):
        pan=[self.Rm/2,self.Rn/2]
        self.r_pos[:,:]=10/np.sqrt(self.Density)*(self.v_pos-pan)*self.Mt+np.random.uniform(-1,1,(self.Rm*self.Rn,2))*(10/np.sqrt(self.Density))

    def f_sr(self,r,rp,p):
        res=0
        k=2*rp*self.E_ym/(1-self.poisson_v)
        res= ((r<=rp)*(1/k)+(r>rp)*(2/(np.pi*k)*np.arcsin((r>rp)*rp/(r+0.000000001))))*p
        return res

    def points_mapping_entrys(self,stimuli):
        entrys=np.zeros(len(stimuli))
        for i in range(0,len(stimuli)):
            dif=self.r_pos-stimuli[i:i+1,:] 
            dists=dif[:,0]**2+dif[:,1]**2
            entrys[i]=np.where(dists==dists.min())[0]
        return np.uint16(entrys)


    def population_simulate(self,EEQS=[],Ips=[[],'Pressure'],acquire_spikes=True,disinf=True,noise=10): #Ips: Inputing parameters
        El   =  -0.065        # restint membrane potential [V]
        thresh = -0.050         # spiking threshold [V]
        gl =   0.16
        Cm =   0.0049
        tau=Cm/gl
        V1=np.random.uniform(-0.000,0.000,(self.Rm*self.Rn,self.t.size))
        Deriv=np.random.uniform(-0.000,0.000,(self.Rm*self.Rn,self.t.size))
        self.VN[:,:]=butterworth_filter(1,np.random.uniform(-0.01,0.01,(self.t.size)),1000,'low',10e3)+self.v_reset
        self.SN[:,:]=1e-6*butterworth_filter(1,np.random.uniform(-noise,noise,(self.t.size)),1000,'low',10e3)
        self.Va[:,:]=self.VN[:,:]
        Cm=(1-self.poisson_v**2)/(2*self.E_ym)
        tau=1/self.Kf
        VH=self.maxfr*self.VL/self.Kf#v
        C1=(2+self.wb/self.Q*self.dt)/(1+self.wb/self.Q*self.dt+self.wb**2*self.dt**2)
        C2=-1/(1+self.wb/self.Q*self.dt+self.wb**2*self.dt**2)
        C3=self.dt**2/(1+self.wb/self.Q*self.dt+self.wb**2*self.dt**2)
        tc1=timec.time()
        Dp=0
        for pt in range(2,self.t.size):  #时间 t.size
            # Simplified skin contact model
            se1,se2=int((Ips[0][pt,1]+self.Hc/2)/Dbp),int((Ips[0][pt,0]+self.Wc/2)/Dbp)
            SC=EEQS[1][se1:se1+self.Nr,se2:se2+self.Nc]
            # Resistance network model
            RI=SC[0:self.Nr:self.Nds,0:self.Nc:self.Nds]*1e-3
            SCT=RI[self.Es[0],self.Es[1]]
            if(Ips[1]=='Pressure'):Dp=Cm*Ips[0][pt,2]/(self.Nds*Dbp*1e-3*np.sqrt(RI[RI>0].size/4))
            elif(Ips[1]=='Depth'):Dp=Ips[0][pt,2]
            Ht=np.max(SCT,axis=1)
            self.Dt[:,pt]=Ht-np.max(Ht)+Dp    
            self.Dt[self.Dt[:,pt]<=0,pt]=0
            self.Is[:,pt]=self.Cs*(self.Dt[:,pt]+self.SN[:,pt])+self.Ce*self.Dt[:,pt]*np.var(1e3*SCT,axis=1)
            self.Uc[:,pt:pt+1]=self.Gi*np.mat(self.Is[:,pt:pt+1])  
            # Single-unit model
            tmp=0
            tmp=tmp+self.wb/self.Q*self.Kb1*(self.Uc[:,pt]-self.Uc[:,pt-1])/(self.dt)
            tmp=tmp+self.Kb2*(self.Uc[:,pt]-2*self.Uc[:,pt-1]+self.Uc[:,pt-2])/(self.dt)**2
            Deriv=tmp
            V1[:,pt]=C1*V1[:,pt-1]+C2*V1[:,pt-2]+C3*Deriv
            self.V1[:,pt] =(self.V1[:,pt-1]+self.wb*V1[:,pt]*self.dt)/(1+self.wb*self.dt)
            if(self.Ttype=="SA1"): self.V2[:,pt] =(self.V2[:,pt-1]+ self.Ku*self.wl*self.Uc[:,pt]*self.dt)/(1+self.wl*self.dt)
            Sum=self.V1[:,pt]+self.V2[:,pt]
            self.Vg[:,pt]=self.Cp*np.abs(Sum)*(Sum<0)+Sum*(Sum>=0)
            self.Vs[:,pt]=self.Vg[:,pt]*(self.Vg[:,pt]<VH)+VH*(self.Vg[:,pt]>=VH);
            self.Va[:,pt] = self.Va[:,pt-1]+(self.Vs[:,pt]+El-self.Va[:,pt])*self.dt/tau
            # In case we exceed threshold
            self.Va[self.Va[:,pt]>thresh,pt-1:pt+1]=[0.04,El] 
            # set the last step and current step to spike value and resting membrane potential
        
        if(acquire_spikes==True):
            self.spike_trains=[]
            for i in range(0,self.Rm): 
                self.spike_trains.append(self.dt*np.where(self.Va[i,:]==0.04)[0])    
        tc2=timec.time()
        if(disinf==True):print(self.Ttype,(tc2-tc1)/self.T)
        return self.Va[:,:],(tc2-tc1)/self.T, self.spike_trains,tc2-tc1    

def isPoisWithinPoly(poly,pois):
    #res=np.bool8(np.ones(len(pois)))
    #输入：点，多边形三维数组
    #poly=[[[x1,y1],[x2,y2],……,[xn,yn],[x1,y1]],[[w1,t1],……[wk,tk]]] 三维数组
    #可以先判断点是否在外包矩形内 
    #if not isPoiWithinBox(poi,mbr=[[0,0],[180,90]]): return False
    #但算最小外包矩形本身需要循环边，会造成开销，本处略去
     #交点个数
    #for epoly in poly: #循环每条边的曲线->each polygon 是二维数组[[x1,y1],…[xn,yn]]
    #for ch in range(len(pois)): #[0,len-1]
    sinsc=np.zeros(len(pois))
    
    for i in range(len(poly)-1): #[0,len-1]
        out=np.bool8(np.zeros(len(pois)))
        s_poi=poly[i,:]
        e_poi=poly[i+1,:]
        #输入：判断点，边起点，边终点，都是[lng,lat]格式数组
        out=out|(s_poi[1]*np.ones(len(pois))==e_poi[1]*np.ones(len(pois)))#排除与射线平行、重合，线段首尾端点重合的情况
        out=out|((s_poi[1]>pois[:,1])&(e_poi[1]>pois[:,1]))
        out=out|((s_poi[1]<pois[:,1])&(e_poi[1]<pois[:,1]))  #线段在射线上边
        #线段在射线下边
        #if s_poi[1]==poi[1] and e_poi[1]>poi[1]: #交点为下端点，对应spoint
        #    return False
        #if e_poi[1]==poi[1] and s_poi[1]>poi[1]: #交点为下端点，对应epoint
        #    return False
        #if s_poi[0]<poi[0] and e_poi[1]<poi[1]: #线段在射线左边
        #    return False
        xseg=e_poi[0]-(e_poi[0]-s_poi[0])*(e_poi[1]-pois[:,1])/(e_poi[1]-s_poi[1]+0.00000001) #求交
        out=out|(xseg<pois[:,0]) #交点在射线起点的左侧
        sinsc=sinsc+(~out)
        #sinsc=isRayIntersectsSegment(pois,s_poi,e_poi)
    res=sinsc%2==1
    #res=res&tmp
    return res

def loc_rt(roi,locs):  #select dots_in area of restriction  unit:mm
    res=np.bool8(np.ones(len(locs)))
    for i in range(-1,len(roi)-1):  #0.00000001 was added to avoid overfiting
        s=(0-roi[i,0])/(roi[i+1,0]-roi[i,0]+0.00000001)-(0-roi[i,1])/(roi[i+1,1]-roi[i,1]+0.00000001)
        if(s>0):
            jdg=(locs[:,0]-roi[i,0])/(roi[i+1,0]-roi[i,0]+0.00000001)-(locs[:,1]-roi[i,1])/(roi[i+1,1]-roi[i,1]+0.00000001)>0
        else:
            jdg=(locs[:,0]-roi[i,0])/(roi[i+1,0]-roi[i,0]+0.00000001)-(locs[:,1]-roi[i,1])/(roi[i+1,1]-roi[i,1]+0.00000001)<0
        res=res&jdg
    return res

def points_map_to_entrys(stimuli,thea,grid_buf,Density):
    tmp=grid_buf+100
    entry_buf=np.int16(tmp[:,0]+tmp[:,1]*(100*2))
    v_pos=entry_buf;
    Mt=np.mat([[np.cos(thea),np.sin(thea)],[-np.sin(thea),np.cos(thea)]]);
    c2=np.round(stimuli[:,[0,1]]*Mt.T*np.sqrt(Density)/10)+100
    AA=np.uint16(c2[:,0]+200*c2[:,1])
    Es=(v_pos[:]==AA[:]).T
    return np.where(Es==True)[0]


def Mt(thea):
    return np.mat([[np.cos(thea),np.sin(thea)],[-np.sin(thea),np.cos(thea)]])


def sin_wave(X,w,intentation):
    y=intentation*np.sin(w*X)
    return y

def sin_wave1(X,wst,intentation):
    y=intentation*wst*np.cos(wst*X)
    return y

def triangular_wave(X,rate,intentation):
    T=2*intentation/(rate*(1e-3))
    y1=2*intentation*1e6*((2*X/T)-0.5)
    y2=2*intentation*1e6*np.floor(2*X/T)
    y3=(y1-y2)*(y1>y2)+(y2-y1)*(y1<y2)
    return y3*1e-6

def square_wave(X,rate,intentation):
    T=2*intentation/(rate*(1e-3))
    y1=intentation*((2*X/T)-0.5)
    y2=intentation*np.floor(2*X/T)
    y3=(rate*(1e-3))*(y1>y2)+(-rate*(1e-3))*(y1<y2)
    return y3

def trapezoidal(X,rate,intentation):
    T=2*intentation/(rate*(1e-3))
    y1=2*intentation*1e6*((2*X/T))
    y2=2*intentation*1e6*np.floor(2*X/T)
    y3=(y1-y2)*(y1>y2)+(y2-y1)*(y1<y2)
    return y3*1e-6
def step_wave(X,Tstart,Tend,rate,rate1,intentation):
    state1=rate*(1e-3)*(X-Tstart)*(X>Tstart)
    state2=state1+(intentation-state1)*(state1>intentation)
    state3=state2+(intentation+rate1*1e-3*(X-Tend)-state2)*(X>Tend)
    state4=state3+(0-state3)*(state3<0)
    return state4


def band_filter(dt,t,fl,fh,signal):
    #tmp=0
    s0=np.zeros((t.size))
    s1=np.zeros((t.size))
    wbl,wbh=2*np.pi*fl,2*np.pi*fh
    #s0+=signal[:]
    for time in range(0,t.size-1):  #时间 t.size
        s0[time+1] =(1-wbh*dt)*s0[time]+wbh*(signal[time])*dt   
        s1[time+1] =(1-wbl*dt)*s1[time]+ (s0[time+1]-s0[time])
    return s1 

def butterworth_filter(order,x,f,typ,fs):
    if(typ=='low'):
        w1=2*f/fs
        b, a = signal.butter(order, w1, typ)
    elif(typ=='high'):
        w1=2*f/fs
        b, a = signal.butter(order, w1, typ)
    elif(typ=='band'):
        w1=2*f[0]/fs
        w2=2*f[1]/fs
        b, a = signal.butter(order,[w1,w2], typ)
    sf = signal.filtfilt(b,a,x)  
    return sf


def f_sr(r,rp,p,E_ym=50*1e3):
    res=0
    poisson_v=0.4
    k=2*rp*E_ym/(1-poisson_v)
    res= ((r<=rp)*(1/k)+(r>rp)*(2/(np.pi*k)*np.arcsin((r>rp)*rp/(r+0.000000001))))*p
    #else: return 2/(np.pi*k)*np.arcsin(rp/r)*p
    return res


def f_sp(x,Ta):
    tmp2=np.sin(2*np.pi*x/Ta)
    state1=(actionp_plus*tmp2*(tmp2>=0)+actionp_minus*tmp2*(tmp2<0))
    return state1*(x<=Ta)

def compare_sp(cmp,f,x,Ta):
    value=1-f*Ta
    return 1*(cmp>=value)+0*(cmp<value)

def st_step_wave(x,rate,Tstart):
    state1=rate*(1e-3)*(x-Tstart)*(x>Tstart)
    return state1

def basic_sp(f,x,Ta):
    value=np.abs(np.cos(np.pi*f*Ta/2))
    tmp1=np.abs(np.sin(np.pi*f*x))
    return 1*(tmp1>=value)+0*(tmp1<value)

def freq_modulate(f,x):
    #return np.abs(np.sin(np.pi*f*x))
    return -f*x+1

def triangular_wave1(X,rate,intentation):
    T=2*intentation/(rate*(1e-3))
    y1=2*intentation*1e6*((2*X/T)-0.5)
    y2=2*intentation*1e6*np.floor(2*X/T)
    y3=(rate*(1e-3))*(y1>y2)+(-rate*(1e-3))*(y1<y2)
    return y3*1e-6


def step_wave2(X,Tstart,Tend,rate,rate1,intentation):
    state1=rate*(1e-3)*(X-Tstart)*(X>Tstart)
    state2=state1+(intentation-state1)*(state1>intentation)
    state3=state2+(intentation+rate1*1e-3*(X-Tend)-state2)*(X>Tend)
    state4=state3+(0-state3)*(state3<0)
    return state4

def step_wave1(X,Tstart,Tend,rate,rate1,intentation):
    #return intentation*wst*np.cos(wst*X)
    state1=(rate*1e-3)*(X>Tstart)
    state2=state1+(0-state1)*(X>(Tstart+intentation*1e3/rate))
    state3=state2+((rate1*1e-3)-state2)*(X>Tend)
    state4=state3+(0-state3)*(X>(Tend+intentation*1e3/np.abs(rate1)))
    return state4


def single_NTH_func(f,Rc,Cs,G,Kb1,Kb2,Ku,WB,WL,Q):
    S=1j*2*np.pi*f
    summ=WB/Q*Kb1*S+Kb2*S**2 
    return 1e6*0.015/np.abs(summ*((1/(S**2+WB*S/Q+WB**2))*(WB/(S+WB)))+Ku*WL/(S+WL))/(Rc*Cs*G[0,0])

def transfer_func1(S,Kb,Ku,WB,Q,WL,Num):
    summ=0
    for i in range(2):
       summ+=Kb[i]*S**(i+1) 
    return summ*((WB/Q/(S**2+WB*S/Q+WB**2))**(1))+Ku*WL/(S+WL)

def transfer_func(S,KN,Kb,Ku,WB,Q,WL,Num):
    summ=0
    for i in range(2):
       summ+=Kb[i]*S**(i+1) 
    return (summ*((WB/Q/(S**2+WB*S/Q+WB**2))**(1))+Ku*WL/(S+WL))*KN

def probe_pi(presure,rad,E_ym=50*1e3):
    poisson_v=0.4
    return ((1-poisson_v**2)/(2*rad*E_ym))*presure

def probe_ip(indent,rad,E_ym=50*1e3):  #pa,Young modulus
    poisson_v=0.4  
    return ((2*rad*E_ym)/(1-poisson_v**2))*indent

def rtm(thea):            
    return np.mat([[np.cos(thea),np.sin(thea)],[-np.sin(thea),np.cos(-thea)]])