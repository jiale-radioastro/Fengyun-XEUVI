from sunpy.map import Map,make_fitswcs_header
from astropy.wcs import WCS
from scipy.interpolate import RectBivariateSpline
from scipy.ndimage import gaussian_filter
from sunpy.coordinates import propagate_with_solar_surface, sun

def give_tform(img1, img0=None, center=(512,512), radiusSize=1024, angleSize=1800):
    # 输入待对齐的二维图像img1和参考图像img0
    # 输出欧几里得变换矩阵tform以及太阳半径re(以像素数为单位)
    # img1和img0的尺寸必须相同，像素大小相近
    # 对齐步骤包括霍夫变换对齐、旋转对齐和光流法对齐
    # 若无参考图像，只使用霍夫变换对齐
    # 注意参考图像是否需要转置
    # img2 霍夫变换对齐后的图像   tform1 霍夫变换对应的变换矩阵
    # img3 旋转对齐后的图像      tform2 旋转对齐对应的变换矩阵
    # img4 光流法对齐后的图像    tform3 光流法对齐对应的变换矩阵
    
    # 去除图像中的nan值
    img1 = removenan(img1)
    if not type(img0).__name__ == 'NoneType':
        img0 = removenan(img0)
        if np.shape(img1)!=np.shape(img0):
            print('警告：参考图像与待对齐图像大小不一致，对齐可能出错')
    sz = np.shape(img1)
    
    # 霍夫变换对齐
    img1 = img1.reshape([1,sz[0],sz[1]])
    img2,re = firstalign(img1)
    img2 = img2[0,:,:]
    tform1 = func(rotation=0, translation=[-re[0][0], -re[0][1]])
    if type(img0).__name__ == 'NoneType':
        return tform1, re[0][2]
    
    # 旋转对齐
    imp0, Ptsetting = pT.convertToPolarImage(img0, center,
                                        radiusSize=radiusSize, angleSize=angleSize)  # 将参考图像转为极坐标
    imp0_log = np.log(np.maximum(imp0,0)+1) # 参考图像取对数
    imp2, Ptsetting = pT.convertToPolarImage(img2, center,
                                            radiusSize=radiusSize, angleSize=angleSize) # 将原始图像转为极坐标
    imp2_log = np.log(np.maximum(imp2,0)+1) # 原始图像取对数
    dy, dx, cor = xcorrcenter(imp0_log[:,100:500].astype('float32'),\
                            imp2_log[:,100:500].astype('float32'))  # 互相关，得到用于旋转改正的参数
    imp3 = immove(imp2.astype('float32'),  dy,0).astype('float32') # 旋转改正
    img3 = Ptsetting.convertToCartesianImage(imp3) # 转回直角坐标
    theta = -dy/angleSize*2*np.pi
    shift = [-center[0]*np.cos(theta)+center[1]*np.sin(theta)+center[0],\
           -center[0]*np.sin(theta)-center[1]*np.cos(theta)+center[1]]
    tform2 = func(rotation=theta,translation=shift)
    
    # 光流法对齐
    img4 = img3.copy()
    cor = 0
    n_iter = 0
    while cor<0.9 and n_iter<2:
        # 如果cor结果不理想，则多次进行光流法对齐
        # 使用n_iter可以设置迭代次数
        d, cor, tform_tmp = all_align(img0.astype('float32'), img4.astype(
            'float32'), winsize=31, step=20, r_t=2, arrow=0)
        img4 = warp(img4.astype('float32'), tform_tmp) # 应用光流法对齐结果进行精细对齐
        if n_iter == 0:
            tform3 = tform_tmp
        else:
            tform3 = func(tform3.params @ tform_tmp.params)
        n_iter += 1
        
    tform = func(tform1.params @ tform2.params @ tform3.params)
    return tform, re[0][2]

def align_map(fy_map, aia_map=None, refer_map=None, rot_corr=True, center=(512,512), radiusSize=1024, angleSize=1800):
    # 输入风云未对齐的图像fy_map, aia图像aia_map，以及一个参考图像refer_map
    # 当存在aia_map时，refer_map由aia_map替换
    # 若既不存在aia_map，也不存在refer_map，只做霍夫变换对齐
    # rot_corr考虑太阳自转对对齐的影响，默认为True
    # 输出对齐后的map以及变换矩阵
    
    # 注意！！ 这里fy_map, aia_map, refer_map采用的data排列格式都是aia默认图像的排列格式，即第一个维度是行，第二个维度是列
    
    if type(aia_map).__name__ == 'NoneType' and type(refer_map).__name__ == 'NoneType':
        fy_img = fy_map.data
        tform , _ = give_tform(fy_img)
        fy_map.data = warp(fy_img,tform)
    else:
        if not type(aia_map).__name__ == 'NoneType':
            refer_map = aia_map

        if rot_corr:
            try:
                refer_map = drot_map(refer_map, fy_map.date)
            except:
                print('自转矫正失败，可能与参考图像的头文件有关，请检查相关参数')

        sz = np.shape(fy_map)
        sz_refer = np.shape(refer_map)
        fy_img = fy_map.data
        refer_img = refer_map.data
        _, fy_re = give_tform(fy_img)
        _, refer_re = give_tform(refer_img)
        refer_img2 = rescale(refer_img,scale=refer_re/fy_re,sz2=(1024,1024))

        # 对齐程序
        tform , _ = give_tform(fy_img,img0=refer_img2, center=center, radiusSize=radiusSize, angleSize=angleSize)
        fy_map.data - warp (fy_img,tform)
        
    return fy_map, tform
    
def drot_map(aia_map,out_time):
    out_frame = Helioprojective(observer='earth', obstime=out_time,
                                rsun=aia_map.coordinate_frame.rsun)
    out_center = SkyCoord(0*u.arcsec, 0*u.arcsec, frame=out_frame)
    header = make_fitswcs_header(aia_map.data.shape,
                                           out_center,
                                           scale=u.Quantity(aia_map.scale))
    out_wcs = WCS(header)
    with propagate_with_solar_surface():
        out_warp = aia_map.reproject_to(out_wcs)
    data1 = out_warp.data
    header1 = aia_map.fits_header
    header1['date-obs'] = str(out_time)
    header1['CRLN_OBS'] = aia_map.observer_coordinate.lon.deg+sun.L0(time=out_time).deg
    header1['CRLT_OBS'] = aia_map.observer_coordinate.lat.deg
    header1['DSUN_OBS'] = aia_map.observer_coordinate.radius.m
    return Map((data1,header1))

def rescale(img,scale,sz2=(1024,1024)):
    img=removenan(img)
    sz=np.shape(img)
    img=gaussian_filter(img,(scale,scale),0) #高斯滤波，平滑AIA图像
    x=np.arange(0,sz[0],1)
    y=np.arange(0,sz[1],1)
    f=RectBivariateSpline(x,y,img)
    t0=np.arange(0,sz2[0],1)
    offset=(sz[0]-1)/2-scale*(sz2[0]-1)/2
    x1=scale*t0+offset
    y1=scale*t0+offset
    return f(x1,y1)
