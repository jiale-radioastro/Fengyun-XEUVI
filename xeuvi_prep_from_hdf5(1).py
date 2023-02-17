# Created by Jiale Zhang on Nov. 3rd
# Perform data calibration for XEVUI, including alignment and rotation
# Save figures in ".jpeg" format and save data in ".fits" format
# Tested on python 3.9.15

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from cv2 import warpPolar, Sobel, CV_64F
import os
from astropy.io import fits
from scipy.signal import medfilt, correlate
from scipy.optimize import curve_fit
from tqdm import tqdm
from sunpy.map import Map
from sunpy.image.transform import affine_transform
import sunpy.visualization.colormaps
from aiapy.calibrate import register, update_pointing
from IPython import display
from scipy.io import readsav
import h5py
import cv2

# The function of the limb, used for fitting
def limb_curve(theta,x_offset,y_offset,r_sun):
    deg2rad=np.pi/180
    return np.sqrt((r_sun*np.cos(theta*deg2rad)-x_offset)**2\
                   +(r_sun*np.sin(theta*deg2rad)-y_offset)**2)

# The ouput settings
save_figs=True  # True if you want to save figures
save_fits=False   # True if you want to save calibrated data in '.fits' format
save_npy=False   # True if you want to save the offset of the solar disk center and rotation angle




# File path and the output directory
# work_dir='/Users/jiale/Desktop/fengyun_XEUVI/XEUVI-optimizing/data/fist20211028_1500-1820/' # The directory where the input fits files exist
work_dir='/Users/gyh/Desktop/research/X-EUVI数据质量提升/ZhangJL/220419/'

hdf5names= sorted([i for i in os.listdir(work_dir) if (i.startswith('FY3E') and i.endswith('.HDF'))])
# fitsnames=fitsnames[0:2]
#The filenames should start with "xeuvi" and end with ".fits"
figs_dir=work_dir+'figures/'
fits_dir=work_dir+'calibrated_data/'
npy_path=work_dir+'calibrated_data/cal_parm.npy'
aia_fits=work_dir+'aia_data/'+[i for i in os.listdir(work_dir+'aia_data/') if i.endswith('.fits')][0]
# os.listdir(aia_fits)

if save_figs:# Create the directory if it does not exist
    if not os.path.exists(figs_dir):
        os.mkdir(figs_dir)
if save_fits:
    if not os.path.exists(fits_dir):
        os.mkdir(fits_dir)


# 读取HDF5数据
with h5py.File(work_dir+hdf5names[0],'r') as f:
    filedata=f['/Data/Radiance'][:]
    msec=f['/Time/Msec_Count'][:]
    day=f['/Time/Day_Count'][:]

sz_tyx=np.shape(filedata)
sz_tr=[2896, 18198]
maxRadius=np.round(max(sz_tyx[1],sz_tyx[2])/2**0.5).astype('int')
r_sun=1460.          # The estimated solar radius in pixel
# r_sun_range=[1520,1640]  # The possible range of the solar radius in pixel
r_sun_range=[1300,1720]
fit_parm=[0,0,395]      # The initial settings of the fitting parameters [x_offset,y_offset,r_sun]
tf_rad=[200,1200]
filter_kernel=1      # The kernel length for medium filter




#load reference aia images

aia_map=Map(aia_fits)
aia_map_prep=register(update_pointing(aia_map))
img_xy=aia_map_prep.data
sz_xy2=np.shape(img_xy)
maxRadius2=np.round(sz_xy2[0]/2**0.5).astype('int')
# cartesian coord. to polar coord.
center=(sz_xy2[0]/2-0.5,sz_xy2[1]/2-0.5)
flags=8  #WarpFillOutliers 8   WarpInverseMap 16
# sz_tr=[2896, 18198]
# tf_rad=[200,1200]
img_tr=warpPolar(img_xy,dsize=sz_tr,center=center,maxRadius=maxRadius2,flags=flags)
tf_curve=np.mean(img_tr[:,tf_rad[0]:tf_rad[1]],axis=1)
original_tf_curve=np.hstack([tf_curve,tf_curve])
# plt.figure('tf_curve')
# plt.plot(original_tf_curve)



parmlist=[]
for numi in tqdm(range(0,len(hdf5names))):
    with h5py.File(work_dir+hdf5names[numi],'r') as f:
        filedata=f['/Data/Radiance'][:]
        msec=f['/Time/Msec_Count'][:]
        day=f['/Time/Day_Count'][:]

    if filedata.size == 0:
        continue

    # header = file[0].header
    # filedata = file[0].data
    sz_tyx=np.shape(filedata)

    for t in range(sz_tyx[0]):
    # 一个hdf5文件中有多个时刻的数据，用t标记
        img_xy=np.transpose(filedata[t,:,:])
        img_xy=img_xy*(img_xy>0)
        sz_xy=(sz_tyx[2],sz_tyx[1])

        # cartesian coord. to polar coord.
        center=(sz_xy[0]/2-0.5,sz_xy[1]/2-0.5)
        flags=8  #WarpFillOutliers 8   WarpInverseMap 16
        img_tr=warpPolar(img_xy,dsize=sz_tr,center=center,maxRadius=maxRadius,flags=flags)
        #0 dim t axis. 1 dim r axis

        img_tr_sharp = Sobel(img_tr,CV_64F,1,0,ksize=5)
        upper=r_sun_range[1]
        lower=r_sun_range[0]
        limb_list=[]
        t_list=np.arange(0,360,360/sz_tr[1])
        r_list=np.arange(0,maxRadius,maxRadius/sz_tr[0])
        for i in range(len(t_list)):
            tmp=img_tr_sharp[i,:]
            max_tmp=np.max(tmp[lower:upper])
            index=[j for j in range(lower,upper) if (tmp[j] == max_tmp)]
            limb_list.append(float(np.mean(r_list[index])))
            #limb_list.append(float(r_list[index[0]]))
        limb_list_smooth=medfilt(limb_list,kernel_size=filter_kernel)
        parm, cov = curve_fit(limb_curve, t_list[:], limb_list_smooth[:],p0=fit_parm,method='trf',loss='cauchy')
        parm=list(parm)

        center=(sz_xy[0]/2-0.5-parm[0],sz_xy[1]/2-0.5-parm[1])
        flags=8  #WarpFillOutliers 8   WarpInverseMap 16
        img_tr=warpPolar(img_xy,dsize=sz_tr,center=center,maxRadius=maxRadius,flags=flags)
        #0 dim t axis. 1 dim r axis

        ## ===========初始的旋转改正方法===================
        # tf_curve=np.mean(img_tr[:,tf_rad[0]:tf_rad[1]],axis=1)
        # rot_angle=0
        # if (numi ==0 and t==0):
        #     #original_tf_curve=np.hstack([tf_curve,tf_curve])
        #     rot_angle=(np.argmax(correlate(tf_curve, original_tf_curve)) - (len(tf_curve)-1)) * 360/sz_tr[1]
        # else:
        #     # rot_angle=(np.argmax(correlate(tf_curve, original_tf_curve)) - (len(tf_curve)-1)) * 360/sz_tr[1]
        #     rot_angle=0
        # parm.append(rot_angle)
        # parmlist.append(parm)

        c = np.cos(np.deg2rad(rot_angle))
        s = np.sin(np.deg2rad(rot_angle))
        rmatrix = np.array([[c, -s],[s, c]])
        img_xy2=affine_transform(img_xy,rmatrix,image_center=(sz_xy[0]/2-0.5-parm[0],sz_xy[1]/2-0.5-parm[1]), recenter=True)

        if (numi ==0 and t==0):
            img_xy_original=img_xy2
        # cartesian coord. to polar coord.
        center=(sz_xy[0]/2-0.5,sz_xy[1]/2-0.5)
        flags=8  #WarpFillOutliers 8   WarpInverseMap 16
        img_tr=warpPolar(img_xy2,dsize=sz_tr,center=center,maxRadius=maxRadius,flags=flags)
        tf_curve=np.mean(img_tr[:,tf_rad[0]:tf_rad[1]],axis=1)
        original_tf_curve=np.hstack([tf_curve,tf_curve])



        # =====================新的旋转改正方法=======================
        # 获取 img_xy2 的形状
        h, w = img_xy2.shape[:2]

        # 旋转 img_xy2 并查找最大相关系数
        best_angle = None
        best_corr = -np.inf
        for angle in np.arange(-180, 180, 10):
        # for angle in np.arange(-180, 180, 0.1):
            M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1)
            rotated = cv2.warpAffine(img_xy2, M, (w, h))
            img1 = cv2.normalize(img_xy_original, None, 0, 10000, cv2.NORM_MINMAX, cv2.CV_32F)
            img2 = cv2.normalize(rotated, None, 0, 10000, cv2.NORM_MINMAX, cv2.CV_32F)

            result = cv2.matchTemplate(img1, img2, cv2.TM_CCORR_NORMED)
            (_, maxVal, _, _) = cv2.minMaxLoc(result)
            if maxVal > best_corr:
                best_angle = angle
                best_corr = maxVal

        for angle in np.arange(best_angle-10, best_angle+10, 1):
        # for angle in np.arange(-180, 180, 0.1):
            M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1)
            rotated = cv2.warpAffine(img_xy2, M, (w, h))
            img1 = cv2.normalize(img_xy_original, None, 0, 10000, cv2.NORM_MINMAX, cv2.CV_32F)
            img2 = cv2.normalize(rotated, None, 0, 10000, cv2.NORM_MINMAX, cv2.CV_32F)

            result = cv2.matchTemplate(img1, img2, cv2.TM_CCORR_NORMED)
            (_, maxVal, _, _) = cv2.minMaxLoc(result)
            if maxVal > best_corr:
                best_angle = angle
                best_corr = maxVal

        for angle in np.arange(best_angle-1, best_angle+1, 0.1):
        # for angle in np.arange(-180, 180, 0.1):
            M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1)
            rotated = cv2.warpAffine(img_xy2, M, (w, h))
            img1 = cv2.normalize(img_xy_original, None, 0, 10000, cv2.NORM_MINMAX, cv2.CV_32F)
            img2 = cv2.normalize(rotated, None, 0, 10000, cv2.NORM_MINMAX, cv2.CV_32F)

            result = cv2.matchTemplate(img1, img2, cv2.TM_CCORR_NORMED)
            (_, maxVal, _, _) = cv2.minMaxLoc(result)
            if maxVal > best_corr:
                best_angle = angle
                best_corr = maxVal

        # 旋转 img_xy2 并使其与 img_xy_original 对齐
        M = cv2.getRotationMatrix2D((w // 2, h // 2), best_angle, 1)
        img_xy2 = cv2.warpAffine(img_xy2, M, (w, h))

        img_xy_original = img_xy2

        parm.append(best_angle)
        parmlist.append(parm)



        if save_figs:
            xeuvi_cmap=plt.get_cmap('sdoaia193')
            plt.figure('img', figsize=(8,8),dpi=200)
            plt.imshow(img_xy2,origin='lower',vmin=1, vmax=5000,cmap=xeuvi_cmap)
            plt.xlabel('Solar-X [pixel]',fontsize=10)
            plt.ylabel('Solar-Y [pixel]',fontsize=10)
            # plt.title('XEUVI 19.5nm '+header['DATE-OBS'])
            fig1 = plt.gcf()
            fig1.savefig("{} image{:0>3d}_{:0>2d}.jpeg".format(figs_dir,numi,t),format='jpeg')
            plt.close('all')
            display.clear_output(wait=True)
        if save_fits:
            #header['crpix1']=sz_xy[0]/2-0.5
            #header['crpix2']=sz_xy[1]/2-0.5
            header['cunit1'] = 'arcsec'
            header['cunit2'] = 'arcsec'
            xeuvi_map=Map(img_xy2,header)
            xeuvi_map.save("{} image{:0>3d}.fits".format(fits_dir,numi),overwrite=True)


if save_npy:
    np.save(npy_path,np.array(parmlist))
