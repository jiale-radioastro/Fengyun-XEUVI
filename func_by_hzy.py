import numpy as np
from tqdm import tqdm
from astropy.io import fits
from skimage.morphology import  opening
from skimage.morphology import remove_small_objects as rs
from func_by_jkf import imnorm,fitswrite
import cv2
def _predict_(X):#[samples,4]
    coef = np.array([9.07331219e-03,4.83963960e+00,-1.58571516e+00,-1.64596653e-03,-3.46679025e-01,1.42157963e+00])#max,mean,median,std
    bias = -2.02218094

    out = X@coef+bias
    
    return 1 if out>0 else -1

method = cv2.HOUGH_GRADIENT
def houghcircle(im):
    imorg = im.astype('float32')
    x = imorg-opening(imorg, np.ones((31, 31)))  # 用形态学提取高频结构
    x0 = x > (x.mean()+1*x.std())  # 二值化，均值标准差
    x0 = rs(x0, 15)  # 去掉小面积

    x = np.minimum(x0*x, 500)  # 控制最大值
    x = imnorm(x)*255  # 归255化
    c_hough = cv2.HoughCircles(np.uint8(x), method, 1, 150, param1=10, param2=10,minRadius=385,maxRadius=400)
    #print(c_hough)
    return c_hough
def _predict3_(pic,th):
    yuzhi= 0.8
    clow = 415
    chigh = 609
    try:
        xmax = np.median(pic,1)
        ymax = np.median(pic,0)
        xlow = np.min(np.where(xmax>yuzhi))
        xhigh = np.max(np.where(xmax>yuzhi))
        ylow = np.min(np.where(ymax>yuzhi))
        yhigh = np.max(np.where(ymax>yuzhi))
        jiangemin = 1000#np.min((xhigh-xlow,yhigh-ylow))
        xc = (xlow+xhigh)/2
        yc = (ylow+yhigh)/2
        c = (xc>clow)and(xc<chigh)and(yc>clow)and(yc<chigh)
    except:
        return -1
    return 1 if ((jiangemin>760) and c ) else -1#and (np.max((xhigh-xlow,yhigh-ylow))<1000)

def predict(img,th,ifhough=0):#输入图片，输出为1为有完整图像，-1为残缺图像
    img0 = img.copy()
    img = img.astype(np.float32)/th
    feature = np.array([np.max(img),np.median(img),np.std(img),np.max(img[256:-256,256:-256]),np.median(img[256:-256,256:-256]),np.std(img[256:-256,256:-256])])
    if _predict_(feature)==1:
        if _predict3_(img,th=th)==1:
            if np.sum(img0==0)>20:
                return -1
            if ifhough:
                hc = houghcircle(img0)#Hough circles
                if (type(hc).__name__ =='NoneType') or (hc.shape[1]!=1):
                    return -1
            return 1

    else:
        return -1


def predict_orbit(imglist,if1024=0,byhough=1,houghnum=5):#输入一个list，包含一天的图像[img1,img2,...]
    #4月份的数据不是1024*1024,如果输入这样的数据，选择if1024=1，进行裁剪
    #byhough=1时对一轨的头尾部分进行hough检测，多个圆时判定为重影
    #houghnum设定对一轨头尾多少张图检测重影
    il = np.array(imglist)
    if if1024!=1:
        il = il[:,24:-24, 4:-4]
    med1 = np.median(il[:,256:-256,256:-256],(1,2))#每张图片中间部分的中值
    th = np.median(med1)#所有中值的中值
    out=[]
    lens = il.shape[0]
    if byhough:
        for i in range(lens):
            if (i<houghnum) or ((lens-i)<houghnum):
                out.append(predict(il[i],th,ifhough=1))
            else:
                out.append(predict(il[i],th,ifhough=0))
    else:
        for i in range(lens):
            out.append(predict(il[i],th,ifhough=0))


    return out#输出list中为1的图像是有完整太阳像，-1是没有
def fy3_writefits(im,dir,name,**kw):
    hd = fits.Header()
    scale0 = kw['Scale']
    hd['CTYPE1'] = 'HPLN-TAN'  # 'solar_x '
    hd['CTYPE2'] = 'HPLT-TAN'  # 'solar_y '
    hd['CUNIT1'] = 'arcsec '
    hd['CUNIT2'] = 'arcsec '
    hd['CRPIX1'] = 512.5
    hd['CRPIX2'] = 512.5
    hd['CRVAL1'] = 0
    hd['CRVAL2'] = 0
    hd['CDELT1'] = scale0
    hd['CDELT2'] = scale0
    hd['CROTA1'] = 0.00000000000
    hd['CROTA2'] = 0.00000000000
    hd['CROTACN1'] = 0.00000000000
    hd['CROTACN2'] = 0.00000000000
    
    hd['R_SUN'] = kw['R_SUN']
    hd['RSUN_REF'] = 696000000
    hd['RSUN_OBS'] = kw['R_SUN']*scale0
    
    hd['DATE-OBS'] = kw['obstime'].isot
    hd['DATOBSJD'] = kw['obstime'].jd
    fname = 'fy3_l1_{}.fits'.format(kw['obstime'].strftime('%Y%m%d_%H%M%S'))
    fitswrite(dir+'/'+fname,im.astype('int16'),header=hd)