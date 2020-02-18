"""
# -*- coding: utf-8 -*-
Created on Sat Nov 18 17:11:16 2017
@author: Administrator
"""
'''
https://www.cnblogs.com/smallpi/p/4555854.html
'''
import os 
import sys
import ultils as alt
import Receptors as rslib
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import simset as mysim 
from skimage import transform
import matplotlib.cm as cm
import scipy.signal as signal


# a function that generates a Gaussian operator
def func(x,y,sigma=1):
    return 100*(1/(2*np.pi*sigma))*np.exp(-((x-2)**2+(y-2)**2)/(2.0*sigma**2))

def rgb2gray(rgb):
    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 *b 
    return gray
    
#image.reshape(int(h/mysim.prope_d),int(w/mysim.prope_d))
#image=transform.resize(image,(int(h/mysim.prope_d),int(w/mysim.prope_d))) #比例缩放
def constructing_equivalent_probe_stimuli_from_pimage(pimage,w,h,roi):
    'probe image'
    pimageinf=np.array([w,h,pimage.shape[1],pimage.shape[0]])
    
    bw=roi[:,0].max()-roi[:,0].min()
    bh=roi[:,1].max()-roi[:,1].min()
    cols=int(bw/rslib.Dbp)
    rows=int(bh/rslib.Dbp)
    '''
    Or=[int(0-roi[:,0].min()/bw*cols),
        int(0-roi[:,1].min()/bh*rows)]  
    #extend EQS
    EPS=np.array(pimage)
    A=np.array(pimage)
    A=np.hstack([np.zeros([A.shape[0],Or[0]]),A])
    A=np.hstack([A,np.zeros([A.shape[0],cols-Or[0]])])
    A=np.vstack([A,np.zeros([rows-Or[1],A.shape[1]])])
    EEPS=np.vstack([np.zeros([Or[1],A.shape[1]]),A])
    '''
    #extend EQS
    EPS=np.array(pimage)
    A=np.array(pimage)
    A=np.hstack([np.zeros([A.shape[0],cols]),A])
    A=np.hstack([A,np.zeros([A.shape[0],cols])])
    A=np.vstack([A,np.zeros([rows,A.shape[1]])])
    EEPS=np.vstack([np.zeros([rows,A.shape[1]]),A])
    
    'equvilent stimuli dots'
    selimg=pimage
    dots=np.meshgrid(np.linspace(0,w,selimg.shape[1]),np.linspace(h,0,selimg.shape[0]))
    x=dots[0].reshape(dots[0].size,1)
    y=h-dots[1].reshape(dots[1].size,1)
    th=selimg.reshape(selimg.size,1)
    tmp=np.hstack([x,y,th])
    pins=tmp[:,:]
    x,y,th=pins[:,0:1],pins[:,1:2],pins[:,2:3]
    eq_stimuli=np.hstack([x,y,x*0,x*0,np.ones([x.size,1])*rslib.Dbp*1e-3,th*1e-3])


    return [pimageinf,EEPS,eq_stimuli,EPS]


def constructing_equivalent_probe_stimuli_from_image(img,w,h,roi):
    # Generate a 5*5 Gaussian operator with a standard deviation of 5
    suanzi1 = np.fromfunction(func,(15,15),sigma=3)
    # Laplace extent operator
    suanzi2 = np.array([[1, 1, 1],
                        [1,-8, 1],
                        [1, 1, 1]])
    'Original image'
    res_buf=[]
    res_buf.append([w,h])
    res_buf.append(np.array(img))
    
    'Grayscale image'
    grayimg=np.array(rgb2gray(np.array(img)))
    grayimg =grayimg * (1 / 255)  #归一化
    #grayimg=grayimg/50 # color height
    res_buf.append(grayimg)
  
    'edge dection'
    image2=np.array(grayimg)
    
    image2[0:2,:]=1
    image2[image2.shape[0]-1:image2.shape[0],:]=1
    image2[:,0:1]=1
    image2[:,image2.shape[1]-1:image2.shape[1]]=1
    #grayimg =(grayimg - grayimg.min()) * (1 / (grayimg.max() - grayimg.min()))  #归一化
    
    # Using the generated Gaussian operator to convolve with the original image to smooth the image
    image2 = signal.convolve2d(image2, suanzi1, mode="same")
    
    # edge dection to image
    image2 = signal.convolve2d(image2, suanzi2, mode="same")
    # normalize the pixel value of image int 0-1
    image2 = (image2 - image2.min()) * (1 / (image2.max() - image2.min())) 
    #(image2/float(image2.max()))*1
    #Make the gray value larger than the gray average value 0 (white) for easy observation of the edge
    image2[image2>image2.mean()] = 0
    
    #fill zeros to pixels beside edge of image
    num=9
    image2[0:num,:]=0
    image2[image2.shape[0]-num:image2.shape[0],:]=0
    image2[:,0:num]=0
    image2[:,image2.shape[1]-num:image2.shape[1]]=0
    
    edgimage=image2
    res_buf.append(np.array(edgimage))
    
    'obtian height image with edge enhancement'
    edimage=(edgimage - edgimage.min()) * (1 / (edgimage.max() - edgimage.min()))
    imageh=grayimg#+edimage
    
    imageh=(imageh - imageh.min()) * (1 / (imageh.max() - imageh.min()))
    res_buf.append(imageh)
    
    'EPS image'
    pimage=transform.resize(imageh,(int(h/rslib.Dbp),int(w/rslib.Dbp)))
    pimage[pimage>=0.1]=1
    pimage[pimage<0.1]=0
    
    epsimg=pimage
    res_buf.append(pimage)
    pimageinf=np.array([w,h,pimage.shape[1],pimage.shape[0]])
    

    
    bw=roi[:,0].max()-roi[:,0].min()
    bh=roi[:,1].max()-roi[:,1].min()
    cols=int(bw/rslib.Dbp)
    rows=int(bh/rslib.Dbp)
    
    '''
    Or=[int(0-roi[:,0].min()/bw*cols),
        int(0-roi[:,1].min()/bh*rows)]  
    #extend EQS
    'extended EQS '
    A=np.array(pimage)
    A=np.hstack([np.zeros([A.shape[0],Or[0]]),A])
    A=np.hstack([A,np.zeros([A.shape[0],cols-Or[0]])])
    A=np.vstack([A,np.zeros([rows-Or[1],A.shape[1]])])
    A=np.vstack([np.zeros([Or[1],A.shape[1]]),A])
    '''
    A=np.array(pimage)
    A=np.hstack([np.zeros([A.shape[0],cols]),A])
    A=np.hstack([A,np.zeros([A.shape[0],cols])])
    A=np.vstack([A,np.zeros([rows,A.shape[1]])])
    A=np.vstack([np.zeros([rows,A.shape[1]]),A])
    
    eepsimg=A
    res_buf.append(eepsimg)
    
    
    
    'equvilent stimuli dots'
    selimg=pimage
    dots=np.meshgrid(np.linspace(0,w,selimg.shape[1]),np.linspace(h,0,selimg.shape[0]))
    x=dots[0].reshape(dots[0].size,1)
    y=dots[1].reshape(dots[1].size,1)
    th=selimg.reshape(selimg.size,1)
    tmp=np.hstack([x,y,th])
    pins=tmp[tmp[:,2]>0.1,:]
    res_buf.append(pins)
    
    x,y,th=pins[:,0:1],pins[:,1:2],pins[:,2:3]
    eq_stimuli=np.hstack([x,y,x*0,x*0,np.ones([x.size,1])*mysim.prope_d*1e-3,th*1e-3])
    #return EEQS
    # 显示图像
    '''
    plt.figure(figsize=(6,3*2))
    plt.subplot(3,1,1)
    plt.imshow(img)
    plt.axis("off")
    plt.subplot(3,1,2)
    plt.imshow(pimage,cmap=cm.Greys)
    plt.axis("off")
    
    plt.subplot(3,1,3)
    plt.scatter(pins[:,0],pins[:,1],s=0.5,c=pins[:,2],cmap=cm.Greys,vmin=0,vmax=3)

    plt.show()
    '''
    return res_buf,eq_stimuli,[pimageinf,eepsimg],[pimageinf,epsimg]



def img_stimuli_static_pressing(dt,T,pf,buf,roi,spx,spy,shift,w):
    slide=0
    stimuli_buf=[]
    while(1):   
        ips=np.zeros([int(T/dt),3])
        for i in range(int(T/dt)):
            [ox,oy]=[spx+slide,spy]
            ips[i,:]=[ox,oy,pf[i]]
        slide=slide+shift
        stimuli_buf.append(ips)
        if(slide+spx>w):break
    return stimuli_buf

def img_stimuli_scaning_with_uniformal_speed(dt,T,pf,speed,sp_scandir,end_scandir,sp_shiftdir,end_shiftdir,shift):
    vslide=0
    stimuli_buf=[]
    while(1): 
        ips=np.zeros([int(T/dt),3])
        for i in range(int(T/dt)):
            [ox,oy]=[sp_scandir+i*dt*speed,sp_shiftdir+vslide]
            ips[i,:]=[ox,oy,pf[i]]
            if(ox>end_scandir):break
        vslide=vslide+shift
        stimuli_buf.append(ips)
        if(vslide>end_shiftdir):break
    return stimuli_buf

def print_figs(s,buf,fig):
    labelsy=np.arange(0,buf[0][1]+buf[0][1]/5,buf[0][1]/5)
    labelsx=np.uint16(np.arange(0,buf[0][0]+buf[0][0]/5,buf[0][0]/5))
    
    ax = fig.add_subplot(8,2,2+s)
    plt.title('Original  image')
    img=buf[1]
    ax.imshow(img,aspect='auto')
    plt.xticks(np.linspace(0,img.shape[1],6),labelsx,fontsize=8)
    plt.yticks(np.linspace(0,img.shape[0],6),labelsy,fontsize=8)
    
    ax = fig.add_subplot(8,2,4+s,sharex=ax)
    plt.title('Grayscale image after normalization')
    img=buf[2]
    md=ax.imshow(img,cmap=cm.Greys,aspect='auto')
    plt.xticks(np.linspace(0,img.shape[1],6),labelsx,fontsize=8)
    plt.yticks(np.linspace(0,img.shape[0],6),labelsy,fontsize=8)
    if(s==-1):plt.colorbar(md,cax=fig.add_axes([0.48, 0.33, 0.01, 0.25]))
    '''
    ax = fig.add_subplot(6,2,6+s,sharex=ax)
    plt.title('Edge dection using Laplace operator')
    img=buf[3]
    ax.imshow(img,cmap=cm.Greys,aspect='auto')
    plt.xticks(np.linspace(0,img.shape[1],6),labelsx,fontsize=8)
    plt.yticks(np.linspace(0,img.shape[0],6),labelsy,fontsize=8)
    '''
    ax = fig.add_subplot(8,2,6+s,sharex=ax)
    plt.title('Height image with edge enhancement')
    img=buf[4]
    ax.imshow(img,cmap=cm.Greys,aspect='auto')
    plt.xticks(np.linspace(0,img.shape[1],6),labelsx,fontsize=8)
    plt.yticks(np.linspace(0,img.shape[0],6),labelsy,fontsize=8)
   
    ax = fig.add_subplot(8,2,8+s)
    plt.title('Equivalent probe stimuli (EPS) image')
    img=buf[5]
    ax.imshow(img,cmap=cm.Greys,aspect='auto',vmin=0,vmax=1)
    plt.xticks(np.linspace(0,img.shape[1],6),labelsx,fontsize=8)
    plt.yticks(np.linspace(0,img.shape[0],6),labelsy,fontsize=8)
    
    ax = fig.add_subplot(4,2,6+s)
    

    plt.title('Extended EPS (EEPS) image')
    img=buf[6]
    ax.imshow(img,cmap=cm.Greys,aspect='auto')
    ftiproi=np.loadtxt('data/txtdata/fingertip_roi.txt')
    ftiproi=img.shape[0]-np.vstack([ftiproi,ftiproi[0,:]])*9
    plt.plot(ftiproi[:,0]-ftiproi[:,0].min(),ftiproi[:,1]-ftiproi[:,1].min(),'y-',linewidth=1)
    
    plt.fill_between(ftiproi[:,0]-ftiproi[:,0].min(),ftiproi[:,1]-ftiproi[:,1].min(),facecolor='y',alpha=0.3) 
    plt.scatter([25],[50],s=60,c='k',marker='+')
    plt.xticks(np.linspace(0,img.shape[1],6),np.round(1.5*labelsx,1),fontsize=8)
    plt.yticks(np.linspace(0,img.shape[0],6),np.round(3.2*labelsy,1),fontsize=8)
    plt.xlabel('x [mm]',fontsize=8)
    plt.ylabel('y [mm]',fontsize=8)
   
    '''
    ax = fig.add_subplot(6,2,10+s)
    plt.title('Equivalent probe stimuli')
    pins=buf[7]
    ax.scatter(pins[:,0],pins[:,1],s=0.5,c=pins[:,2],cmap=cm.Greys,vmin=0,vmax=3)
    plt.xticks(labelsx,fontsize=8)
    plt.yticks(labelsy,fontsize=8)
    '''
'''
#nat_image =Image.open('saved_figs/letters_80-10.jpg')
#art_image =Image.open('saved_figs/gratings_24-15.jpg')
nat_image =Image.open('saved_figs/natimage.jpg')
art_image=Image.open('saved_figs/artimage.jpg')
#art_image =Image.open('saved_figs/letters_80-10.jpg')
art_buf=constructing_equivalent_probe_stimuli_from_image(art_image,30,10,mysim.fingertiproi)[0]
nat_buf=constructing_equivalent_probe_stimuli_from_image(nat_image,30,10,mysim.fingertiproi)[0]
fig =plt.figure(figsize=(10,9*2)) 
plt.subplots_adjust(hspace=0.5) 
plt.subplots_adjust(wspace=0.4) 
print_figs(-1,art_buf,fig)
print_figs(0,nat_buf,fig)
plt.savefig('saved_figs/image_to_stimuli.png', bbox_inches='tight',dpi=300) 
'''