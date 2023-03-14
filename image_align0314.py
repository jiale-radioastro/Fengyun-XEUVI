# created on Mar. 14th

import h5py,os,glob
import numpy as np
import polarTransform as pT
from tqdm import tqdm
import sunpy.visualization.colormaps
from IPython import display
from func_by_jkf import *
from func_by_hzy import *
from func_by_zjl import *


work_dir='/Users/jiale/Desktop/fengyun_XEUVI/2022-04/21/' 
figs_dir=work_dir+'figures3/'

filelist=sorted(glob.glob(work_dir+'*_V0.HDF'))
radiusSize, angleSize = 1024, 1800
IM=np.zeros((1,1024,1024))
timelist=[]
for file in tqdm(filelist):
    f=h5py.File(file,'r')
    im=np.array(f['Data/Radiance'][:])
    msec=f['/Time/Msec_Count'][:]
    day=f['/Time/Day_Count'][:]
    if im.size != 0:
        IM=np.vstack((IM,im[:,24:-24, 4:-4])) 
        timelist.extend(read_timelist(msec,day))

IM=IM[1:,:,:]
flag=np.array(predict_day(IM,if1024=1))
IM=IM[np.where(flag==1)]
timelist=np.array(timelist)[np.where(flag==1)]

center=(MC,NC)
im1,re=firstalign(IM[:,:,:])
im0=np.zeros(im1.shape)

aia_reference_time=timelist[0]
dylist=[]
dlist=[]

for i in tqdm(range(im0.shape[0])):
    if i==0 or (Time(timelist[i])-aia_reference_time).value*24>1:
        aia_map=give_aiamap(timelist[i],aia_dir=aia_dir)
        aia_reference_time=aia_map.date
        aia_map=reduce_aiamap(aia_map,im1[i])
    
    #aia_map_drot=drot_map(aia_map,timelist[i])
    aia_map_drot=aia_map
    aia_img=removenan(aia_map_drot.data.T)
    imp0, Ptsetting = pT.convertToPolarImage(aia_img, center, radiusSize=radiusSize, angleSize=angleSize)
    imp0=np.log(np.maximum(imp0,0)+1)

    imp, Ptsetting = pT.convertToPolarImage(im1[i], center, radiusSize=radiusSize, angleSize=angleSize)
    tmp=np.log(np.maximum(imp,0)+1)
    dy, dx, cor = xcorrcenter(imp0[:,100:500].astype('float32'), tmp[:,100:500].astype('float32'))  # tanslation value by CC
    print(i,dy,dx,cor)
    dylist.append(dy)
    imt = immove(imp.astype('float32'),  dy,0).astype('float32')

    imd=Ptsetting.convertToCartesianImage(imt)
    d, cor, tform = all_align(aia_img.astype('float32'), imd.astype(
       'float32'), winsize=71, step=20, r_t=2, arrow=0)  # tform by Optical flow
    print(i,d,cor)
    if cor<0.8:
       imd = warp(imd.astype('float32'), tform)
       d, cor, tform = all_align(aia_img.astype('float32'), imd.astype(
             'float32'), winsize=31, step=20, r_t=2, arrow=0)  # tform by Optical flow
       print(i,d,cor)
    im0[i] = warp(imd.astype('float32'), tform)
    dlist.append(d)

    xeuvi_cmap=plt.get_cmap('sdoaia193')
    plt.figure('img', figsize=(8,8),dpi=200)
    plt.imshow(im0[i].T,origin='lower',vmin=1, vmax=2000,cmap=xeuvi_cmap)
    plt.xlabel('Solar-X [pixel]',fontsize=10)
    plt.ylabel('Solar-Y [pixel]',fontsize=10)
    plt.title(str(timelist[i]))
    fig1 = plt.gcf()
    #fig1.savefig("{} image{:0>3d}_{:0>2d}.jpeg".format(figs_dir,numi,t),format='jpeg')
    fig1.savefig(figs_dir+str(i)+".jpeg",format='jpeg')
    plt.close('all')
    display.clear_output(wait=True)
