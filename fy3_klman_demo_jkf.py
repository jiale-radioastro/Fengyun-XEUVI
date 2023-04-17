# -*- coding: utf-8 -*-
"""
Created on Thu Feb 23 17:37:07 2023

@author: jkf
"""

import numpy as np

from tqdm import tqdm

from func_by_jkf import *
D=disk(1024,1024,500)

'''
Mp4file='0420'
filelist='D:/ftpshare/jkf/bxy/xeuvi/aligned_xeuvi_from_jiale/network/0420/data.fits'
filelist=sorted(glob.glob(filelist))
IM=[]
for file in tqdm(filelist):
    im,h=fitsread(file)
    IM.append(im)
plt.close('all')    
'''
def rm1(ims):
    IM = ims.copy()
    IM=np.array(IM).squeeze()
    Dr=disk(1024,1024,500)^disk(1024,1024,420)

    tot=IM.shape[0]

    im=IM.copy()
    
    gain=0.97

    T=0.05

    ssa=[]
    for i in tqdm(range(1,tot)): 

        tmp=IM[i].copy()    
        IM[i]=IM[i-1]*gain+(1-gain)*IM[i]
        s=im[i]/IM[i]
        s=removenan(s)

        ss=s[D].std() 
        ssa.append(ss)
        print(ss)
        if ss>T:

            im[i]=IM[i]
        
    # #####################################################
    im=im*D[np.newaxis]
    return im