# created on Mar. 14th
# modified on Mar. 28th

import h5py,os,glob
import numpy as np
import polarTransform as pT
from tqdm import tqdm
import sunpy.visualization.colormaps
from IPython import display
from func_by_jkf import *
from func_by_hzy import *
from func_by_zjl_v2 import *


# work_dir='/Users/gyh/Desktop/research/X-EUVI数据质量提升/弯曲网格事件/0421_orig/'
# work_dir='/Users/gyh/Desktop/research/X-EUVI数据质量提升/ZhangJL/220419/'
work_dir='/Users/gyh/Desktop/research/X-EUVI数据质量提升/鬼像对齐/29/FY3E_XEUVI_20220429_1800_M1.3/' # 初始X-EUVI数据（.hdf5）的路径
work_dir='/Users/jiale/Desktop/fengyun_XEUVI/2022-04/24/' 

figs_dir=work_dir+'figures/' # 对齐后的图像存储路径
aia_dir=work_dir+'aia_data/' # AIA数据存储路径

if not os.path.exists(aia_dir):
    os.mkdir(aia_dir)
    
if not os.path.exists(figs_dir):
    os.mkdir(figs_dir)

# 用于后面的极坐标转换的参数
radiusSize, angleSize = 1024, 1800
MC, NC = 512, 512
center=(MC,NC)
method='jsoc' #下载AIA数据的方式  fido下载的数据是4096*4096 jsoc下载的数据是1024*1024

#测试用数据
all_timelist=[] # 用于存储一天内每张图像的时间
dylist=[] # 用于存储旋转改正得到的旋转角（测试用）
dlist=[] # 用于存储光流法精细对齐的输出参数（测试用）
scalelist=[] # 用于存储图像缩放比例
imgi=0

print('图像平移/旋转对齐子程序')
filelist_packed=separate_orbit(sorted(glob.glob(work_dir+'*_V0.HDF'))) # 读取X-EUVI数据文件名列表，并拆分为不同轨道
orbit_num=len(filelist_packed) #一天内轨道数
print('轨道数: '+str(orbit_num))

for orbit_i in range(orbit_num):
    IM=np.zeros((1,1024,1024))  # 用于存储直接读取的原始图像
    timelist=[]
    
    print('读取数据')
    for file in tqdm(filelist_packed[orbit_i]):
        # 读取.hdf5文件，观测结果和时间信息分别保存在IM和timelist中
        f=h5py.File(file,'r')
        im=np.array(f['Data/Radiance'][:])
        msec=f['/Time/Msec_Count'][:]
        day=f['/Time/Day_Count'][:]
        if im.size != 0:
            IM=np.vstack((IM,im[:,24:-24, 4:-4])) 
            timelist.extend(read_timelist(msec,day))
            
    print('轨道 '+str(orbit_i)+': '+str(timelist[0])+'~'+str(timelist[-1]))
    print('开始筛选完整太阳像')

    IM=IM[1:,:,:]   # 除去第1帧（空白帧），只保留数据
    #选出完整太阳像
    flag=np.array(predict_day(IM,if1024=1))
    IM=IM[np.where(flag==1)]
    timelist=np.array(timelist)[np.where(flag==1)]  
    all_timelist.extend(timelist)
    fulldisk_num=len(timelist)
    print('该轨有'+str(fulldisk_num)+'张完整太阳像')
    
    print('开始进行Hough变换对齐')
    im1,re=firstalign(IM[:,:,:]) # 基于hough变换进行粗对齐，结果存储为im1
    im0=np.zeros(im1.shape) # 用于存储精对齐之后的图像
    
    reference_i=fulldisk_num//2
    print('将第'+str(reference_i)+'张XEUVI图像与AIA图像对齐')
    aia_map=give_aiamap(timelist[reference_i],aia_dir=aia_dir,method=method) # 下载AIA图像，并做预处理
    if aia_map != None:
        sz_aia=np.shape(aia_map.data)
        _,re_a=firstalign(np.array([to1024(aia_map.data,sz_aia[0]//1024)]))
        scale=sz_aia[0]//1024*re_a[0][2]/re[reference_i][2]
        scalelist.append(scale) #存下缩放比例
        aia_map=reduce_aiamap(aia_map,scale=scale)
    
        aia_map_drot=drot_map(aia_map,timelist[reference_i]) # 根据AIA图像与X-EUVI图像的时间差，对AIA图像较差自转
        aia_img=removenan(aia_map_drot.data.T) # 去除nan值
        
        imp0, Ptsetting = pT.convertToPolarImage(aia_img, center,
                                             radiusSize=radiusSize, angleSize=angleSize)  # 将AIA图像转为极坐标
        imp0=np.log(np.maximum(imp0,0)+1) # AIA图像取对数

        imp, Ptsetting = pT.convertToPolarImage(im1[reference_i], center,
                                                radiusSize=radiusSize, angleSize=angleSize) # 粗对齐后的X-EUVI图像转为极坐标
        tmp=np.log(np.maximum(imp,0)+1) # X-EUVI图像取对数

        dy, dx, cor = xcorrcenter(imp0[:,100:500].astype('float32'),
                                  tmp[:,100:500].astype('float32'))  # 互相关，得到用于旋转改正的参数
        imt = immove(imp.astype('float32'),  dy,0).astype('float32') # 旋转改正

        imd=Ptsetting.convertToCartesianImage(imt) # 转回直角坐标
        d, cor, tform = all_align(aia_img.astype('float32'), imd.astype(
           'float32'), winsize=31, step=20, r_t=6, arrow=0) # 用光流法与AIA图像再进行精细对齐
        n_iter=0
        while cor<0.9 and n_iter<1:
            # 如果cor结果不理想，则多次进行光流法对齐
            # 使用n_iter可以设置迭代次数
            imd = warp(imd.astype('float32'), tform) # 应用光流法对齐结果进行精细对齐
            d, cor, tform = all_align(aia_img.astype('float32'), imd.astype(
                 'float32'), winsize=31, step=20, r_t=2, arrow=0)
            n_iter+=1

        reference_img = warp(imd.astype('float32'), tform) # 应用光流法对齐结果进行精细对齐，结果保存在img中
        
        reference_header=aia_map.fits_header.copy()
        reference_header['date-obs']=str(timelist[reference_i])
        
        reference_map=make_euvmap(reference_img.T,reference_header)
    
    print('将该轨的图像与参考的XEUVI图像对齐')
    for i in tqdm(range(fulldisk_num)):
        
        reference_map_drot=drot_map(reference_map,timelist[i]) # 根据参考图像与X-EUVI图像的时间差，对AIA图像较差自转
        reference_img=removenan(reference_map_drot.data.T) # 去除nan值，转置参考图像以进行旋转对齐（不转置将无法对齐）
        imp0, Ptsetting = pT.convertToPolarImage(reference_img, center,
                                                 radiusSize=radiusSize, angleSize=angleSize)  # 将参考图像转为极坐标
        imp0=np.log(np.maximum(imp0,0)+1) # 参考图像取对数

        imp, Ptsetting = pT.convertToPolarImage(im1[i], center,
                                                radiusSize=radiusSize, angleSize=angleSize) # 粗对齐后的X-EUVI图像转为极坐标
        tmp=np.log(np.maximum(imp,0)+1) # 未对齐图像取对数

        dy, dx, cor = xcorrcenter(imp0[:,100:500].astype('float32'),
                                  tmp[:,100:500].astype('float32'))  # 互相关，得到用于旋转改正的参数
        print(i,dy,dx,cor) # 输出旋转改正得到的参数
        dylist.append(dy) # 存下角度改正量
        imt = immove(imp.astype('float32'),  dy,0).astype('float32') # 旋转改正

        imd=Ptsetting.convertToCartesianImage(imt) # 转回直角坐标
        d, cor, tform = all_align(reference_img.astype('float32'), imd.astype(
           'float32'), winsize=31, step=20, r_t=6, arrow=0) # 用光流法与参考图像再进行精细对齐
        print(i,d,cor) # 输出光流法对齐参数
        n_iter=0
        while cor<0.9 and n_iter<1:
            # 如果cor结果不理想，则多次进行光流法对齐
            # 使用n_iter可以设置迭代次数
            imd = warp(imd.astype('float32'), tform) # 应用光流法对齐结果进行精细对齐
            d, cor, tform = all_align(reference_img.astype('float32'), imd.astype(
                 'float32'), winsize=31, step=20, r_t=2, arrow=0)
            print(i,d,cor)
            n_iter+=1

        im0[i] = warp(imd.astype('float32'), tform) # 应用光流法对齐结果进行精细对齐，结果保存在im0中
        dlist.append(d) # 保存光流法的对齐参数

        # 绘制对齐后的X-EUVI图像，以jpeg格式保存在figs_dir中
        xeuvi_cmap=plt.get_cmap('sdoaia193')
        plt.figure('img', figsize=(16,8),dpi=200)
        plt.subplot(1,2,1)
        plt.imshow(np.log(im0[i]).T,origin='lower',vmin=5, vmax=9,cmap=xeuvi_cmap)
        plt.xlabel('Solar-X [pixel]',fontsize=10)
        plt.ylabel('Solar-Y [pixel]',fontsize=10)
        plt.title(str(timelist[i]))

        plt.subplot(1,2,2)
        plt.imshow(np.log(im0[i]).T,origin='lower',vmin=5, vmax=9,cmap=xeuvi_cmap)
        plt.xlabel('Solar-X [pixel]',fontsize=10)
        plt.ylabel('Solar-Y [pixel]',fontsize=10)
        plt.xlim([400,700])
        plt.ylim([500,800])
        plt.title(str(timelist[i]))
        fig1 = plt.gcf()
        fig1.savefig(figs_dir+str(imgi)+".jpeg",format='jpeg')
        plt.close('all')
        imgi+=1
        display.clear_output(wait=True)

    # 将最终对齐结果保存为fits文件
    #fileout=figs_dir+'align_orbit'+str(orbit_i)+'.fits'
    #fitswrite(fileout, im0.astype('float32'))
