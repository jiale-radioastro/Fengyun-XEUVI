
import polarTransform as pT
from func_by_jkf import *
from func_by_hzy import *
from func_by_zjl_v2 import *
from func_by_yzh import *
from func_by_sz import *
from fy3_klman_demo_jkf import rm1
from IPython import display


def align_img(img1,img0,center,radiusSize,angleSize):
    # 以img0为参考图将img1对齐，包含旋转对齐和光流法对齐
    # 输出对齐后的图像img3以及欧几里得变换矩阵tform
    # 注意参考图像是否需要转置
    # img2 旋转对齐后的图像
    # img3 光流法对齐后的图像
    # tform1 旋转对齐对应的欧几里得变换矩阵
    # tform2 光流法对齐对应的欧几里得变换矩阵
        
    imp0, Ptsetting = pT.convertToPolarImage(img0, center,
                                        radiusSize=radiusSize, angleSize=angleSize)  # 将参考图像转为极坐标
    imp0_log=np.log(np.maximum(imp0,0)+1) # 参考图像取对数

    imp1, Ptsetting = pT.convertToPolarImage(img1, center,
                                            radiusSize=radiusSize, angleSize=angleSize) # 将原始图像转为极坐标
    imp1_log=np.log(np.maximum(imp1,0)+1) # 原始图像取对数

    dy, dx, cor = xcorrcenter(imp0_log[:,100:500].astype('float32'),
                            imp1_log[:,100:500].astype('float32'))  # 互相关，得到用于旋转改正的参数
    imp2 = immove(imp1.astype('float32'),  dy,0).astype('float32') # 旋转改正

    img2=Ptsetting.convertToCartesianImage(imp2) # 转回直角坐标

    theta=-dy/angleSize*2*np.pi
    shift=[-center[0]*np.cos(theta)+center[1]*np.sin(theta)+center[0],-center[0]*np.sin(theta)-center[1]*np.cos(theta)+center[1]]
    tform1=func(rotation=theta,translation=shift)

    img3=img2[:,:]
    cor=0
    n_iter=0

    while cor<0.9 and n_iter<2:
        # 如果cor结果不理想，则多次进行光流法对齐
        # 使用n_iter可以设置迭代次数
        d, cor, tform_tmp = all_align(img0.astype('float32'), img3.astype(
            'float32'), winsize=31, step=20, r_t=2, arrow=0)
        img3 = warp(img3.astype('float32'), tform_tmp) # 应用光流法对齐结果进行精细对齐
        if n_iter ==0:
            tform2=tform_tmp
        else:
            tform2=func(np.dot(tform2.params,tform_tmp.params))
        n_iter+=1


    tform=func(np.dot(tform1.params,tform2.params))
    return img3,tform


radiusSize, angleSize = 1024, 1800
MC, NC = 512, 512
center=(MC,NC)
down_method='jsoc' #下载AIA数据的方式  fido下载的数据是4096*4096 jsoc下载的数据是1024*1024

ifflat=0 #是否去除平场
if_rmnoise=0 #是否去除噪声
if_reference=0 #是否输入一张参考图
reference_fits=''
indir = '/Volumes/jiale_disk1/projects/fengyun_XEUVI/pipeline/in/'#hdf地址
outdir = '/Volumes/jiale_disk1/projects/fengyun_XEUVI/pipeline/out/'
outfitsdir = outdir+'/fits/'
outpicdir = outdir+'/pics/'
aia_dir = outdir+'/aia_data/'
mkdir(outfitsdir)
mkdir(outpicdir)
mkdir(aia_dir)
imgi=0

if ifflat:
    print('开始生成平场')
    flatim = makeflat(indir,outdir)

if if_reference:
    reference_map=Map(reference_fits)


all_timelist=[] # 用于存储一天内每张图像的时间
scalelist=[] # 用于存储图像缩放比例
tformlist=[] # 用于存储欧几里得变换矩阵

# im定义
# im_tmp存储临时im
# im0存储原始图(经过筛选)
# im1存储经过hough变换后的图
# im2存储经过精对齐后的图
# im3存储去除躁点和弯曲网格的图
    
print('图像平移/旋转对齐子程序')
filelist_packed=separate_orbit(sorted(glob.glob(indir+'*_V0.HDF'))) # 读取X-EUVI数据文件名列表，并拆分为不同轨道
orbit_num=len(filelist_packed) #一天内轨道数
print('轨道数: '+str(orbit_num))
for orbit_i in range(orbit_num):
    im0=np.zeros((1,1024,1024))  # 用于存储直接读取的原始图像
    timelist=[]
    
    print('读取数据')
    for file in tqdm(filelist_packed[orbit_i]):
        # 读取.hdf5文件，观测结果和时间信息分别保存在IM和timelist中
        f=h5py.File(file,'r')
        im_tmp=np.array(f['Data/Radiance'][:])
        msec=f['/Time/Msec_Count'][:]
        day=f['/Time/Day_Count'][:]
        if im_tmp.size != 0:
            im0=np.vstack((im0,im_tmp[:,24:-24, 4:-4])) 
            timelist.extend(read_timelist(msec,day))
            
    print('轨道 '+str(orbit_i)+': '+str(timelist[0])+'~'+str(timelist[-1]))
    print('开始筛选完整太阳像')

    im0=im0[1:,:,:]   # 除去第1帧（空白帧），只保留数据
    #选出完整太阳像
    flag=np.array(predict_orbit(im0,if1024=1))
    im0=im0[np.where(flag==1)]
    timelist=np.array(timelist)[np.where(flag==1)]
    all_timelist.extend(timelist)
    fulldisk_num=len(timelist)
    print('该轨有'+str(fulldisk_num)+'张完整太阳像')
    
    if ifflat:
        if type(flatim).__name__ =='NoneType':
            print('there is no flat'),quit()
        else:
            IM = divide_flat(IM,flatim)

    print('开始进行Hough变换对齐')
    im1,re=firstalign(im0[:,:,:]) # 基于hough变换进行粗对齐，结果存储为im1
    im2=np.zeros(im1.shape) # 用于存储精对齐之后的图像
    
    reference_i=fulldisk_num//2
    print('将第'+str(reference_i)+'张XEUVI图像与AIA图像对齐')
    aia_map=give_aiamap(timelist[reference_i],aia_dir=aia_dir,method=down_method) # 下载AIA图像，并做预处理

    if type(aia_map).__name__ =='NoneType':
        if type(reference_map).__name__ =='NoneType':
            print('既没有AIA图，也没有参考图'),quit()
        else:
            aia_map=reference_map
            
    sz_aia=np.shape(aia_map.data)
    _,re_a=firstalign(np.array([to1024(aia_map.data,sz_aia[0]//1024)]))
    scale=sz_aia[0]//1024*re_a[0][2]/re[reference_i][2]
    scalelist.append(scale) #存下缩放比例
    aia_map,scale0=reduce_aiamap(aia_map,scale=scale)

    aia_map_drot=drot_map(aia_map,timelist[reference_i]) # 根据AIA图像与X-EUVI图像的时间差，对AIA图像较差自转
    aia_img=removenan(aia_map_drot.data.T) # 去除nan值
    
    reference_img,tform2=align_img(im1[reference_i],aia_img,center=center,radiusSize=radiusSize,angleSize=angleSize)
    
    reference_header=aia_map_drot.fits_header.copy()
    
    reference_map=make_euvmap(reference_img.T,reference_header)
    
    print('将该轨的图像与参考的XEUVI图像对齐')
    for i in tqdm(range(fulldisk_num)):
        
        reference_map_drot=drot_map(reference_map,timelist[i]) # 根据参考图像与X-EUVI图像的时间差，对AIA图像较差自转
        reference_img=removenan(reference_map_drot.data.T) # 去除nan值，转置参考图像以进行旋转对齐（不转置将无法对齐）

        im2[i],tform2=align_img(im1[i],reference_img,center=center,radiusSize=radiusSize,angleSize=angleSize)
        tform1=func(rotation=0, translation=[-re[i][0], -re[i][1]])
        tform=func(np.dot(tform1.params,tform2.params))
        tformlist.append(tform.params)

    im3 = im2[:,:,:]
    if if_rmnoise:
        im3 = rm_noise(im3)
    im3 = rm1(im3)#去除弯曲网格
    print('开始保存fits')
    for i in tqdm(range(fulldisk_num)):
        fy3_writefits(im3[i],outfitsdir,'',Scale=scale0,obstime=timelist[i],R_SUN=re[i][2])


