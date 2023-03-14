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
import h5py
import cv2
from sunpy.net import Fido, attrs as attrs
from astropy import units as u
import datetime
from tqdm import tqdm

# The function of the limb, used for fitting
def limb_curve(theta,x_offset,y_offset,r_sun):
    deg2rad=np.pi/180
    return np.sqrt((r_sun*np.cos(theta*deg2rad)-x_offset)**2\
                   +(r_sun*np.sin(theta*deg2rad)-y_offset)**2)

# The ouput settings
refer_aia=True  # True if you want to refer to aia image for alignment
save_figs=True  # True if you want to save figures
save_fits=False   # True if you want to save calibrated data in '.fits' format
save_npy=True   # True if you want to save the offset parameters of the solar disk center and rotation angle

# File path and the output directory
# The directory where the input hdf5 files exist
work_dir='/Users/gyh/Desktop/research/X-EUVI数据质量提升/ZhangJL/test_pos_correction/'
# work_dir='/Users/gyh/Desktop/research/X-EUVI数据质量提升/ZhangJL/220419/'


aia_dir=work_dir+'aia_data/'
# os.mkdir(aia_dir)
#aia_fits=work_dir+'aia_data/'+[i for i in os.listdir(work_dir+'aia_data/') if i.endswith('.fits')][0]
# os.listdir(aia_fits)

# results = Fido.search(attrs.Time('2022-07-21T00:00:00','2022-07-21T00:00:12'),attrs.Instrument("aia"),\
#                       attrs.Wavelength(193*u.angstrom),attrs.Physobs.intensity)
# download_file = Fido.fetch(results[0,0], path=aia_dir)

download_file = os.listdir(aia_dir)
aia_map = sunpy.map.Map(aia_dir+download_file[0])
aia_map = register(update_pointing(aia_map))


sz_tr=[2896, 18198]
r_sun=1460.          # The estimated solar radius in pixel
# r_sun_range=[1520,1640]  # The possible range of the solar radius in pixel
r_sun_range=[1300,1720]
fit_parm=[0,0,395]      # The initial settings of the fitting parameters [x_offset,y_offset,r_sun]
tf_rad=[300,1200]
filter_kernel=1      # The kernel length for medium filter

#load reference aia images
img_xy0 = aia_map.data.T


# file=fits.open('/Users/gyh/Desktop/research/X-EUVI数据质量提升/solar image from liuxianyu/195_images.fits')
# data=file[0].data
# img_xy0=data[0].T
# img_xy0=cv2.resize(img_xy0, (2048, 2048), interpolation=cv2.INTER_LINEAR) #congrid to 1024*1024

img_xy0 = img_xy0[::2,::2]  #congrid to 2048*2048
# img_xy0 = img_xy0[::4,::4]  #congrid to 1024*1024

sz_tyx=np.shape(img_xy0)
maxRadius=np.round(max(sz_tyx[0],sz_tyx[1])/2**0.5).astype('int')


# img_xy=cv2.warpAffine(img_xy0, M, (img_xy0.shape[1], img_xy0.shape[0]))

# img_xy=img_xy0*(img_xy0>0)
img_xy=img_xy0
sz_xy=(sz_tyx[1],sz_tyx[0])

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
parm0, cov = curve_fit(limb_curve, t_list[:], limb_list_smooth[:],p0=fit_parm,method='trf',loss='cauchy')


# random_array = np.random.uniform(-20.0, 20.0, (50, 2))
random_array = np.random.uniform(-5.0, 5.0, (50, 2))
parm_x = []
parm_y = []

# random_array=[[0.,0.]]
for [dx,dy] in tqdm(random_array):

    #define shift matrix
    M = np.float32([[1, 0, -dx], [0, 1, -dy]])

    #shift the image
    img_xy=cv2.warpAffine(img_xy0, M, (img_xy0.shape[1], img_xy0.shape[0]))

    img_xy=img_xy*(img_xy>0)
    sz_xy=(sz_tyx[1],sz_tyx[0])

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
    center=(sz_xy[0]/2-0.5-parm[0],sz_xy[1]/2-0.5-parm[1])
    flags=8  #WarpFillOutliers 8   WarpInverseMap 16
    img_tr=warpPolar(img_xy,dsize=sz_tr,center=center,maxRadius=maxRadius,flags=flags)

    parm_x.append(parm[0])
    parm_y.append(parm[1])

# parm0=[-2.69736732e-01, -2.68807854e-01,  8.00513936e+02]
# plt.scatter(random_array[:,0],parm_x)

plt.plot(parm_x-random_array[:,0]-parm0[0])
plt.plot(parm_y-random_array[:,1]-parm0[1])
plt.title('Error for position correction')
plt.xlabel('Test case')
plt.ylabel('Error (pixel)')
plt.legend(['x','y'])
plt.savefig(work_dir+'error_2048_for_max_shift_20.png')
# plt.savefig(work_dir+'error_1024_for_max_shift_10.png')
print(parm0)

# parm=list(parm)
#
# center=(sz_xy[0]/2-0.5-parm[0],sz_xy[1]/2-0.5-parm[1])
# flags=8  #WarpFillOutliers 8   WarpInverseMap 16
# img_tr=warpPolar(img_xy,dsize=sz_tr,center=center,maxRadius=maxRadius,flags=flags)



