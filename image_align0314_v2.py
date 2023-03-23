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


# work_dir='/Users/gyh/Desktop/research/X-EUVI数据质量提升/弯曲网格事件/0421_orig/'
# work_dir='/Users/gyh/Desktop/research/X-EUVI数据质量提升/ZhangJL/220419/'
work_dir='/Users/gyh/Desktop/research/X-EUVI数据质量提升/鬼像对齐/29/FY3E_XEUVI_20220429_1800_M1.3/' # 初始X-EUVI数据（.hdf5）的路径

figs_dir=work_dir+'fig/' # 对齐后的图像存储路径
aia_dir=work_dir+'aia_data/' # AIA数据存储路径

if not os.path.exists(aia_dir):
    os.mkdir(aia_dir)
    
if not os.path.exists(figs_dir):
    os.mkdir(figs_dir)

filelist=sorted(glob.glob(work_dir+'*_V0.HDF')) # 读取X-EUVI数据文件名列表

# 用于后面的极坐标转换的参数
radiusSize, angleSize = 1024, 1800
MC, NC = 512, 512
center=(MC,NC)

IM=np.zeros((1,1024,1024))  # 用于存储直接读取的原始图像
timelist=[] # 用于存储每张图像的时间
for file in tqdm(filelist):
    # 读取.hdf5文件，观测结果和时间信息分别保存在IM和timelist中
    f=h5py.File(file,'r')
    im=np.array(f['Data/Radiance'][:])
    msec=f['/Time/Msec_Count'][:]
    day=f['/Time/Day_Count'][:]
    if im.size != 0:
        IM=np.vstack((IM,im[:,24:-24, 4:-4])) 
        timelist.extend(read_timelist(msec,day))

IM=IM[1:,:,:]   # 除去第1帧（空白帧），只保留数据

#选出完整太阳像
flag=np.array(predict_day(IM,if1024=1))
IM=IM[np.where(flag==1)]
timelist=np.array(timelist)[np.where(flag==1)]


im1,re=firstalign(IM[:,:,:]) # 基于hough变换进行粗对齐，结果存储为im1
im0=np.zeros(im1.shape) # 用于存储精对齐之后的图像

aia_reference_time=timelist[0] # 初始的aia参考时间
dylist=[] # 用于存储旋转改正得到的旋转角（测试用）
dlist=[] # 用于存储光流法精细对齐的输出参数（测试用）

for i in tqdm(range(im0.shape[0])): # 与AIA图像比较进行旋转改正，再基于光流法做精细对齐
    if i==0:
        print(timelist[i])
        aia_map=give_aiamap(aia_reference_time,aia_dir=aia_dir) # 下载AIA图像，并进行预处理
        aia_reference_time=aia_map.date # AIA图像的精确时间
        _,re_a=firstalign(np.array([to1024(aia_map.data,4)])) # 用hough变换处理AIA图像，得到半径
        scale=4*re_a[0][2]/re[i][2] # 基于X-EUVI和AIA图像中的太阳半径，得到比例尺。比例尺用于缩放AIA图像
        aia_map=reduce_aiamap(aia_map,scale=scale) # 缩放AIA图像。仅做插值抽样，没有做卷积。

    if np.abs((Time(timelist[i])-aia_reference_time).value)*48>1:
    # 如果X-EUVI图像与AIA图像时间相差超过0.5 h，下载新的AIA图像
        # print(timelist[i],Time(timelist[i])-aia_reference_time) # 输出时间相关信息
        aia_reference_time=aia_reference_time+datetime.timedelta(hours=1) # 新AIA图像的大致时间（前一张图像后1 h）
        aia_map=give_aiamap(aia_reference_time,aia_dir=aia_dir) # 下载新的AIA图像，并做预处理
        aia_reference_time=aia_map.date # 新AIA图像的精确时间
        _,re_a=firstalign(np.array([to1024(aia_map.data,4)]))
        scale=4*re_a[0][2]/re[i][2]
        aia_map=reduce_aiamap(aia_map,scale=scale)

    
    aia_map_drot=drot_map(aia_map,timelist[i]) # 根据AIA图像与X-EUVI图像的时间差，对AIA图像较差自转
    aia_img=removenan(aia_map_drot.data.T) # 去除nan值，转置AIA图像以进行旋转对齐（不转置将无法对齐）
    imp0, Ptsetting = pT.convertToPolarImage(aia_img, center,
                                             radiusSize=radiusSize, angleSize=angleSize)  # 将AIA图像转为极坐标
    imp0=np.log(np.maximum(imp0,0)+1) # AIA图像取对数

    imp, Ptsetting = pT.convertToPolarImage(im1[i], center,
                                            radiusSize=radiusSize, angleSize=angleSize) # 粗对齐后的X-EUVI图像转为极坐标
    tmp=np.log(np.maximum(imp,0)+1) # X-EUVI图像取对数

    dy, dx, cor = xcorrcenter(imp0[:,100:500].astype('float32'),
                              tmp[:,100:500].astype('float32'))  # 互相关，得到用于旋转改正的参数
    print(i,dy,dx,cor) # 输出旋转改正得到的参数
    dylist.append(dy) # 存下角度改正量
    imt = immove(imp.astype('float32'),  dy,0).astype('float32') # 旋转改正

    imd=Ptsetting.convertToCartesianImage(imt) # 转回直角坐标
    d, cor, tform = all_align(aia_img.astype('float32'), imd.astype(
       'float32'), winsize=31, step=20, r_t=6, arrow=0) # 用光流法与AIA图像再进行精细对齐
    print(i,d,cor) # 输出光流法对齐参数
    n_iter=0
    while cor<0.9 and n_iter<1:
        # 如果cor结果不理想，则多次进行光流法对齐
        # 使用n_iter可以设置迭代次数
        imd = warp(imd.astype('float32'), tform) # 应用光流法对齐结果进行精细对齐
        d, cor, tform = all_align(aia_img.astype('float32'), imd.astype(
             'float32'), winsize=31, step=20, r_t=2, arrow=0)
        print(i,d,cor)
        n_iter+=1

    im0[i] = warp(imd.astype('float32'), tform) # 应用光流法对齐结果进行精细对齐，结果保存在im0中
    dlist.append(d) # 保存光流法的对齐参数

    # 绘制对齐后的X-EUVI图像，以jpeg格式保存在figs_dir中
    xeuvi_cmap=plt.get_cmap('sdoaia193')
    plt.figure('img', figsize=(8,8),dpi=200)
    plt.imshow(im0[i].T,origin='lower',vmin=1, vmax=2000,cmap=xeuvi_cmap)
    plt.xlabel('Solar-X [pixel]',fontsize=10)
    plt.ylabel('Solar-Y [pixel]',fontsize=10)
    plt.title(str(timelist[i]))
    fig1 = plt.gcf()
    fig1.savefig(figs_dir+str(i)+".jpeg",format='jpeg')
    plt.close('all')
    display.clear_output(wait=True)

# 将最终对齐结果保存为fits文件
fileout='align_'+os.path.basename(file)+'.fits'
fitswrite(fileout, im0.astype('float32'))
