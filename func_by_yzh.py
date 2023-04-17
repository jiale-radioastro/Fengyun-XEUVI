#!/usr/bin/env python
# coding: utf-8



import h5py

import matplotlib.pyplot as plt
import matplotlib.axes as ax
import numpy as np
import pandas

import os, fnmatch
def find(pattern, path):
    result = []
    for root, dirs, files in os.walk(path):
        for name in files:
            if fnmatch.fnmatch(name, pattern):
                result.append(os.path.join(root, name))
    return result


def makeflat(file_dir,savedir,ifsave=1):
    #file_dir='/home/houzy/XEUVI_FY3E/2022-04/08/'
    ff=(find("*V0.HDF",file_dir)) #find target HDF files from the current directory, usually all files within one day

    numf=len(ff)
    flagrad=0
    ii=0
    while flagrad != 1:
        f = h5py.File(ff[ii], 'r') #read every HDF file, each includes several frames of images
        dd=f['Data']  
        radtest=dd['Radiance'] #the variable named Data.Radiance is the observed EUV image
        ii=ii+1
        if(radtest.shape[1] == 1072 and radtest.shape[2] == 1032): #only choose those data with the size of 1072*1032, others don't contain EUV images
            newradd=radtest
            flagrad=1

    for i in range(ii,numf):
        print(ff[i])
        f = h5py.File(ff[i], 'r')
    #     print(f)
        dd=f['Data']
        rad=dd['Radiance']
        if(rad.shape[1] == 1072 and rad.shape[2] == 1032):
            newradd=np.concatenate((newradd,rad),axis=0) #store each frame of image into a data array named "newradd"

    newrad=newradd#[:,0:1063,5:1028]

    newaa=np.median(newrad,axis=0)  #caclulate the median value of newrad along time axis (time median of the image sequence)


    import math
    radius=1000
    theta=np.linspace(0,2*(math.pi),int(2*(math.pi)*radius))
    radii=np.linspace(0,radius,int(radius))

    img_pol=np.zeros((len(radii),len(theta)))

    xcp=1088./2 #the central x coordinate
    ycp=1035./2 #the central y coordinate

    for ii in range(0,len(radii)-1):  #convert to polar coordinate
        for jj in range(0,len(theta)-1):
            xp=radii[ii]*math.cos(theta[jj])+xcp
            yp=radii[ii]*math.sin(theta[jj])+ycp
            if xp <= 1071 and yp <= 1031:
                img_pol[ii,jj]=newaa[int(xp),int(yp)]
            
    index=np.median(img_pol,axis=1)


    flat_img=np.zeros((1072,1032)) 
    for i in range(0,1071):
        for j in range(0,1031):
            dis=math.sqrt((float(i-xcp))**2+(float(j-ycp))**2)
            if dis < (radius-1.):
                inc=index[round(dis)]
                flat_img[i,j]=inc


    flat=np.divide(newaa,flat_img,where=flat_img!=0)[24:-24, 4:-4] #calculate the flat field
    if ifsave:
        np.save(savedir+'flat',flat)
    return flat

def divide_flat(ims,flat):
    numimg=len(ims)
    for i in range(0,numimg):
        img=ims[i,:,:]
        img_after=np.divide(img,flat,where=flat!=0)
        ims[i,:,:] = img_after
    return ims
    




