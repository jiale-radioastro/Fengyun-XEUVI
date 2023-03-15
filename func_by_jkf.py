# -*- coding: utf-8 -*-
"""
Created on Wed Oct 12 19:50:47 2022

@author: jkf
"""
import numpy as np
from matplotlib import pyplot as plt
import glob
import os,sunpy.map
import time
from statistics import mode
import cv2
import sys
from skimage.transform import warp, rotate
from skimage.morphology import remove_small_objects as rs
from scipy.ndimage import binary_fill_holes as fh
import pywt
from skimage.morphology import erosion, opening, closing, dilation, label
from skimage import feature, filters
from scipy.optimize import minimize
from tqdm import tqdm
from scipy.signal import savgol_filter as sf, medfilt2d, medfilt
from skimage.measure import regionprops
from astropy.io import fits
import scipy.fftpack as fft
import shutil
import astropy.units as u
from astropy.coordinates import SkyCoord
from sunpy.coordinates import Helioprojective
from sunpy.physics.differential_rotation import solar_rotate_coordinate
from sunpy.time import parse_time
from skimage.transform import EuclideanTransform, SimilarityTransform
from skimage.measure import ransac
from astropy.time import Time
from sunpy.coordinates.sun import P, B0, angular_radius
from scipy.interpolate import interp1d

M, N = 1024, 1024
MC, NC = 512, 512
Y, X = np.mgrid[:M, :N]
wavelet = 'db3'
wavelevel = 6
method = cv2.HOUGH_GRADIENT
# init para
JD0 = 2459847.5233796295
scale_0 = 1.22906374931
scale0 = 1.2290
rsunphoto_0 = 956.2798847823575
rsunpix_0 = 784.32
rsun_0 = scale_0*rsunpix_0
gain_Rsun = rsun_0/rsunphoto_0


#lab = np.load('dark_label113.npy')
M_0 = np.array([1, 0, 0, 0, 1, 0, 0, 0, 1]).reshape(3, 3)


#########################################################


func = EuclideanTransform
err = []


def showim(im, k=3, cmap='gray'):
    mi = np.max([im.min(), im.mean() - k * im.std()])
    mx = np.min([im.max(), im.mean() + k * im.std()])
    if len(im.shape) == 3:
        plt.imshow(im, vmin=mi, vmax=mx)
    else:
        plt.imshow(im, vmin=mi, vmax=mx, cmap=cmap, interpolation='nearest')

    return
# 光流对齐函数


def align_opflow(im1, im2, winsize=11, step=5, r_t=2, arrow=0):
    mask =disk(1024,1024,300)

    flow = cv2.calcOpticalFlowFarneback(im1*mask, im2*mask, flow=None, pyr_scale=0.5,
                                        levels=5, winsize=winsize, iterations=10, poly_n=5, poly_sigma=1.2, flags=0)

    # 按照step 选取有效参考点
    h, w = im1.shape
    x1, y1 = np.meshgrid(np.arange(w), np.arange(h))
    x2 = x1.astype('float32')+flow[:, :, 0]
    y2 = y1.astype('float32')+flow[:, :, 1]
    x1 = x1[mask][::step]
    y1 = y1[mask][::step]
    x2 = x2[mask][::step]
    y2 = y2[mask][::step]

    src = (np.vstack((x1.flatten(), y1.flatten())).T)  # 源坐标
    dst = (np.vstack((x2.flatten(), y2.flatten())).T)  # 目标坐标
    s = dst-src  # 位移量
    Dlt0 = ((np.abs(s[:, 0]) > 0)*1.0 + (np.abs(s[:, 1]) > 0)) > 0  # 判断是否无位移

    if Dlt0.sum() > 0:  # 处理有位移的图像
        dst = dst[Dlt0]
        src = src[Dlt0]
        s = s[Dlt0]
        #####筛选有效参考点################

        # 如果要考虑旋转，可以使用这个函数。但旋转也会带来更大的累计误差。慎重使用。同时返回量是一个齐次矩阵。代码要改很多
        model, D = ransac((src, dst), func, min_samples=200,
                          residual_threshold=r_t, max_trials=200)

        ###########如果需要，画光流场##################
        if arrow == 1:
            plt.figure()
            showim(im1)
            x = src[D, 0]
            y = src[D, 1]
            fx = s[D, 0]
            fy = -s[D, 1]
            plt.quiver(x, y, fx, fy, color='r', scale=0.2,
                       scale_units='dots', minshaft=2)

        try:
            flag = D.sum()/Dlt0.sum()  # 有效控制点占比， 用于评价配准的概率
            d = [model.rotation, model.translation[0], model.translation[1]]
        except:
            flag = -999
            d = [0, 0, 0]

    return d, model, flag, flow


def all_align(im1, im2, winsize=31, step=50, r_t=1, arrow=0):  # 三个波段彼此求偏移量，并最小二乘求解。
    mask =disk(1024,1024,300)
    im1 = im1/np.median(im1[mask])*10000
    im2 = im2/np.median(im2[mask])*10000

    d, model, flag, flow = align_opflow(
        im1, im2, winsize=winsize, step=step, r_t=r_t, arrow=arrow)
    d = removenan(d)

    # tform = func(rotation=d[0],translation=[0,0])

    # imi = warp(imi, tform, output_shape=(imi.shape[0], imi.shape[1]))
    # tx=800*(1-np.cos(d[0])-np.sin(d[0]))
    # ty=800*(1-np.cos(d[0])+np.sin(d[0]))
    # tx=627*d[0]+2.25
    # ty=-627*d[0]-2.25
    tform = func(rotation=d[0], translation=[d[1], d[2]])
    #im3 = warp(imi, tform, output_shape=(imi.shape[0], imi.shape[1]))
    return d, flag, tform


def drot(t1, t2):  # calculation solar ratation
    start_time = parse_time(t1, format='jd')
    c = SkyCoord(0*u.arcsec, 0*u.arcsec, obstime=start_time,
                 observer="earth", frame=Helioprojective)
    c_droted = solar_rotate_coordinate(
        c, time=parse_time(t2, format='jd'), observer=None)
    return c_droted


def removenan(im, key=0):
    """
    remove NAN and INF in an image
    """
    im2 = np.copy(im)
    arr = np.isnan(im2)
    im2[arr] = key
    arr2 = np.isinf(im2)
    im2[arr2] = key

    return im2


def filter4dark():
    filt = disk(M, N, 1024)*1-disk(M, N, 20)
    filt = filters.gaussian(filt*1.0, 3)
    return 1-filt


def removedarkspots(im):
    # tot=lab.max()
    im0 = im.copy()
    f = fft.fft2(im0)
    f = fft.fftshift(f)*filter4dark()
    # f=fft.ifftshift(f)
    im1 = np.abs(fft.ifft2(f))
    im0 = im0*(1-lab)+lab*im1
    return im0


def removeray(im, T=0.2):
    c = medfilt2d(im, 3)
    # d=np.abs(removenan(im/c)-1)>T
    d = np.abs(removenan(im-c)) > (T*c)
    out = c*d+(1-d)*im
    return out, d


def makeGaussian(size, sigma=3, center=None):

    x = np.arange(0, size, 1, float)
    y = x[:, np.newaxis]

    if center is None:
        x0 = y0 = size // 2
    else:
        x0 = center[0]
        y0 = center[1]

    return np.exp(-4*np.log(2) * ((x-x0)**2 + (y-y0)**2) / sigma**2)


def tukey(width, alpha=0.5):

    from scipy import signal

    s = signal.windows.tukey(width, alpha=alpha)
    s = s[np.newaxis]
    z = np.dot(s.T, s)
    return z


def disk(M, N, r0):
    X, Y = np.meshgrid(np.arange(int(-(N / 2)), int(N / 2)),
                       np.linspace(-int(M / 2), int(M / 2) - 1, M))
    r = (X) ** 2 + (Y) ** 2
    r = (r ** 0.5)
    im = r < r0
    return im


def mdisk(M, N, r0, dx, dy):  # 偏移中心的圆模板
    Disk = disk(M, N, r0)
    mdisk = immove2(Disk, dx, dy)
    return mdisk


def immove2(im, dx=0, dy=0):  # 图像平移
    im2 = im.copy()
    tform = SimilarityTransform(translation=(dx, dy))
    im2 = warp(im2, tform.inverse, output_shape=(
        im2.shape[0], im2.shape[1]), mode='constant', cval=0)

    return im2


def fitcircle(x, y, xc, yc, w=1):  # 圆拟合

    def f_2(para):
        """ calculate the algebraic distance between the data points and the mean circle centered at c=(xc, yc) """
        Ri = np.sqrt((x-para[0])**2 + (y-para[1])**2)-para[2]
        return (w*Ri*Ri).sum()

    x0 = [xc, yc, 783]
    result = minimize(f_2, x0, method='nelder-mead', tol=1e-6,
                      bounds=((xc-100, xc+100), (yc-100, yc+100), (760, 820)))
    #result=minimize(f_2, x0,method='L-BFGS-B',tol=1e-6,bounds=((xc-100,xc+100),(yc-100,yc+100),(760,820)))

    return result.x

# D1=disk(1024,1024,50)
# D2=disk(1024,1024,80)
# D3=disk(1024,1024,100)
# Ring1=D1^D2
# Ring2=D2^D3
# win=tukey(1024)
# fx=np.linspace(0,511,512)
# fx=fx/1024


def removestrip(img):  # 剔除条
    # img=np.maximum(img,0)
    # img=np.log(img+1)
    f = pywt.wavedec2(img.copy(), wavelet, mode='symmetric', level=wavelevel)
    g = f.copy()
    for i in range(wavelevel):
        (LHY, HLY, HHY) = f[i+1]
        s = np.median(LHY, axis=1)
        LHY = LHY-s[:, np.newaxis]
        g[i+1] = (LHY, HLY, HHY)
    im = pywt.waverec2(g, wavelet, mode='symmetric')
    # im=np.exp(im)-1
    return im


# def evaluate(img):
#     imsub=img[512:-512,512:-512]
#     imsub=imsub/np.mean(imsub)#*100

#     f=fft.fft2(imsub)
#     f=np.abs(fft.fftshift(f))
#     pro,impolar=polpow(f,1)
#     pro_l=np.log(pro+1)
#     pro_s=sf(pro_l,31,3)
#     noise=np.median(pro_l[-100:])+3*pro_l[-100:].std()
#     D=pro_s<noise
#     s=np.where(D)
#     s=s[0][0]

#     q=np.log10(np.median(pro[50:80]))-np.log10(np.median(pro[80:120]))
#     q=(q-2)*10
#     cutoff=1/fx[s]*1.23
#     return q,cutoff,pro

def zscore2(im):
    im = (im - np.mean(im)) / im.std()
    return im


def xcorrcenter(standimage, compimage, R0=2, flag=0):
    # if flag==1,standimage 是FFT以后的图像，这是为了简化整数象元迭代的运算量。直接输入FFT以后的结果，不用每次都重复计算
    try:
        M, N = standimage.shape

        standimage = zscore2(standimage)
        s = fft.fft2(standimage)

        compimage = zscore2(compimage)
        c = np.fft.ifft2(compimage)

        sc = s * c
        im = np.abs(fft.fftshift(fft.ifft2(sc)))  # /(M*N-1);%./(1+w1.^2);
        cor = im.max()
        if cor == 0:
            return 0, 0, 0

        M0, N0 = np.where(im == cor)
        m, n = M0[0], N0[0]

        if flag:
            m -= M / 2
            n -= N / 2
            # 判断图像尺寸的奇偶
            if np.mod(M, 2):
                m += 0.5
            if np.mod(N, 2):
                n += 0.5

            return m, n, cor
        # 求顶点周围区域的最小值
        immin = im[(m - R0):(m + R0 + 1), (n - R0):(n + R0 + 1)].min()
        # 减去最小值
        im = np.maximum(im - immin, 0)
        # 计算重心
        x, y = np.mgrid[:M, :N]
        area = im.sum()
        m = (np.double(im) * x).sum() / area
        n = (np.double(im) * y).sum() / area
        # 归算到原始图像
        m -= M / 2
        n -= N / 2
        # 判断图像尺寸的奇偶
        if np.mod(M, 2):
            m += 0.5
        if np.mod(N, 2):
            n += 0.5
    except:
        print('Err in align_Subpix routine!')
        m, n, cor = 0, 0, 0
    return m, n, cor


def initpara(im0, X, Y):
    D = im0 > im0.mean()
    D = fh(closing(rs(D), np.ones((19, 19))))
    arr2 = rs(D, 1000000)
    sa = (arr2*1).sum()
    if sa>0:
        xc = ((X * arr2).astype(float).sum()) / sa
        yc = ((Y * arr2).astype(float).sum()) / sa
    else:
        xc,yc=1023.5,1023.5

    edge = dilation(arr2)-arr2*1
    # edge=arr2*1-erosion(arr2)
    # s=regionprops(arr2*1)
    # s=s[0].bbox
    # win=np.zeros((M,N))
    # win[s[0]+1:s[2]-1,s[1]+1:s[3]-1]=1
    # edge=edge*win
    return rs(edge > 0, 500, connectivity=2), xc, yc, arr2


def fit_edge(w, sd, X, Y, xc, yc):  # 边缘拟合

    for i in range(10):
        X = X[sd]  # 圆拟合坐标
        Y = Y[sd]
        w = w[sd]
        circle_fit = fitcircle(X, Y, xc, yc, w=w*w)  # 圆拟合
        if (((circle_fit[:2]-[xc, yc])**2).sum() < 0.1) and i > 2:
            xc, yc = circle_fit[0], circle_fit[1]
            break

        r = (X-circle_fit[0])**2+(Y-circle_fit[1])**2
        dr = np.sqrt(r)-circle_fit[2]

        sd = dr < (1*dr.std())
        if sd.sum() < 1000:
            break
        xc, yc = circle_fit[0], circle_fit[1]
    return xc, yc, circle_fit[2]

def immove(image, dx, dy):
    """
    image shift by subpix
    """
    # The shift corresponds to the pixel offset relative to the reference image
    from scipy.ndimage import fourier_shift
    if dx == 0 and dy == 0:
        offset_image = image
    else:
        shift = (dx, dy)
        offset_image = fourier_shift(fft.fft2(image), shift)
        offset_image = np.real(fft.ifft2(offset_image))

    return offset_image
def imnorm(im, mx=0, mi=0):
    #   图像最大最小归一化 0-1
    if mx != 0 and mi != 0:
        pass
    else:
        mi, mx = np.min(im), np.max(im)

    im2 = removenan((im - mi) / (mx - mi))

    arr1 = (im2 > 1)
    im2[arr1] = 1
    arr0 = (im2 < 0)
    im2[arr0] = 0

    return im2


def fit_localmax(im0, xc, yc, rsun, w_out=10, w_in=50, flag=1):  # 局部最大值拟合
    circle_fit = [xc, yc, rsun]

    D1 = mdisk(M, N, circle_fit[2]+w_out, circle_fit[0]-MC, circle_fit[1]-NC)
    D2 = mdisk(M, N, circle_fit[2]-w_in, circle_fit[0]-MC, circle_fit[1]-NC)
    mask = (D1-D2)*(im0 > 0)
    mask = mask > 0
    local_maxi = feature.peak_local_max(
        im0, footprint=np.ones((31, 31)), labels=mask)

    Xl = local_maxi[:, 1]*1.0  # 圆拟合坐标
    Yl = local_maxi[:, 0]*1.0
    w = im0[local_maxi[:, 0], local_maxi[:, 1]]
    T = 0
    for i in range(5):

        circle_fit = fitcircle(Xl.copy(), Yl.copy(), xc, yc, w=w)  # 圆拟合
        if ((circle_fit[:2]-[xc, yc])**2).sum() < 0.01:
            xc, yc = circle_fit[0], circle_fit[1]
            break

        r = (Xl-circle_fit[0])**2+(Yl-circle_fit[1])**2
        dr = np.sqrt(r)-circle_fit[2]
        T = dr.std()
        if flag == 1:
            T = T
        else:
            T = min(T, 1.5)
        sd = np.abs(dr) < (1*T)

        if sd.sum() < 100:
            break
        Xl = Xl[sd]  # 圆拟合坐标
        Yl = Yl[sd]
        w = w[sd]
        xc, yc = circle_fit[0], circle_fit[1]

    return circle_fit[0], circle_fit[1], circle_fit[2], Xl, Yl, T


def fitswrite(fileout, im, header=None):

    if os.path.exists(fileout):
        os.remove(fileout)
    if header is None:
        fits.writeto(fileout, im, output_verify='fix',
                     overwrite=True, checksum=False)
    else:
        fits.writeto(fileout, im, header, output_verify='fix',
                     overwrite=True, checksum=False)


def fitsread(filein):
    head = '  '
    hdul = fits.open(filein)

    try:
        data0 = hdul[0].data.astype(np.float32)
        head = hdul[0].header
    except:
        hdul.verify('silentfix')
        data0 = hdul[1].data
        head = hdul[1].header

    return data0, head


def toMP4(dirin, Mp4file, jpgdir='JPG', fps=20.0):
    filelist = sorted(glob.glob(dirin+'\\'+jpgdir+'\\*.png'))
    frame = cv2.imread(filelist[0])
    print(dirin, Mp4file)
    fps = fps  # 帧率
    fourcc = cv2.VideoWriter_fourcc(*'XVID')  # 视频编码器
    # 视频分辨率,与原始图片保持一致,或者将图片皆resize到訪分辨率
    size = (frame.shape[1]-10, frame.shape[0]-10)
    out = cv2.VideoWriter(dirin+'\\'+Mp4file+'.mp4',
                          fourcc, fps, size)  # 定义输出文件及其它参数

    for image_file in tqdm(filelist):
        frame = cv2.imread(image_file)
        out.write(frame[:size[1], :size[0], :])
        # tqdm.write(image_file)

        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    out.release()
    cv2.destroyAllWindows()
    return


def mkdir(path):    # 引入模块
    path = path.strip()
    # 去除尾部 \ 符号
    path = path.rstrip("\\")

    # 判断路径是否存在
    isExists = os.path.exists(path)

    # 判断结果
    if isExists:
        return False
    else:
        os.makedirs(path)
        return True


def rmdir(path):

    isExists = os.path.exists(path)

    # 判断结果
    if isExists:

        shutil.rmtree(path)
    return


def drawlog(result, datadir):
    import matplotlib.pyplot as plt
    plt.figure()
    plt.subplot(311)
    plt.plot(result[:, 1].astype('float')-2459825.5,
             result[:, 4].astype('float')/783, '.', label='Rsun_pix')
    plt.plot(result[:, 1].astype('float')-2459825.5,
             result[:, 6].astype('float')/1.229, '.', label='Scale')
    plt.legend()

    plt.subplot(312)
    plt.plot(result[:, 1].astype('float')-2459825.5,
             result[:, 8].astype('float')/50, '+', label='Bright')
    plt.plot(result[:, 1].astype('float')-2459825.5,
             result[:, 7].astype('float'), '*', label='FOV')
    plt.plot(result[:, 1].astype('float')-2459825.5,
             result[:, 11].astype('float'), '.', label='Quality')

    plt.legend()
    plt.subplot(313)
    plt.plot(result[:, 1].astype('float')-2459825.5,
             result[:, 12].astype('float'), '*', label='dx')
    plt.plot(result[:, 1].astype('float')-2459825.5,
             result[:, 13].astype('float'), '.', label='dy')
    plt.plot(result[:, 1].astype('float')-2459825.5, 1 -
             result[:, 14].astype('float'), '+', label='1-Correlation')
    plt.legend()
    plt.suptitle(os.path.basename(datadir))
    plt.tight_layout()

    plt.suptitle(datadir)


def sutri465():

    import numpy as np
    import matplotlib as mpl

    cmap465 = [[0., 0., 0.],
               [0.00784314, 0., 0.00392157],
               [0.01960784, 0., 0.01176471],
               [0.02745098, 0., 0.01960784],
               [0.03921569, 0., 0.02745098],
               [0.04705882, 0., 0.03529412],
               [0.05882353, 0.00392157, 0.04313725],
               [0.07058824, 0.00392157, 0.04705882],
               [0.07843137, 0.00392157, 0.05490196],
               [0.09019608, 0.00392157, 0.0627451],
               [0.09803922, 0.00392157, 0.07058824],
               [0.10980392, 0.00784314, 0.07843137],
               [0.12156863, 0.00784314, 0.08627451],
               [0.12941176, 0.00784314, 0.09019608],
               [0.14117647, 0.00784314, 0.09803922],
               [0.14901961, 0.00784314, 0.10588235],
               [0.16078431, 0.01176471, 0.11372549],
               [0.16862745, 0.01176471, 0.12156863],
               [0.18039216, 0.01176471, 0.12941176],
               [0.19215686, 0.01176471, 0.13333333],
               [0.2, 0.01176471, 0.14117647],
               [0.21176471, 0.01568627, 0.14901961],
               [0.21960784, 0.01568627, 0.15686275],
               [0.23137255, 0.01568627, 0.16470588],
               [0.24313725, 0.01568627, 0.17254902],
               [0.25098039, 0.01568627, 0.17647059],
               [0.2627451, 0.01960784, 0.18431373],
               [0.27058824, 0.01960784, 0.19215686],
               [0.28235294, 0.01960784, 0.2],
               [0.29019608, 0.01960784, 0.20784314],
               [0.30196078, 0.01960784, 0.21568627],
               [0.31372549, 0.02352941, 0.21960784],
               [0.32156863, 0.02352941, 0.22745098],
               [0.33333333, 0.02352941, 0.23529412],
               [0.34117647, 0.02352941, 0.24313725],
               [0.35294118, 0.02352941, 0.25098039],
               [0.36470588, 0.02745098, 0.25882353],
               [0.37254902, 0.02745098, 0.2627451],
               [0.38431373, 0.02745098, 0.27058824],
               [0.39215686, 0.02745098, 0.27843137],
               [0.40392157, 0.02745098, 0.28627451],
               [0.41176471, 0.02745098, 0.29411765],
               [0.42352941, 0.03137255, 0.30196078],
               [0.43529412, 0.03137255, 0.30588235],
               [0.44313725, 0.03137255, 0.31372549],
               [0.45490196, 0.03137255, 0.32156863],
               [0.4627451, 0.03137255, 0.32941176],
               [0.4745098, 0.03529412, 0.3372549],
               [0.48627451, 0.03529412, 0.34509804],
               [0.49411765, 0.03529412, 0.34901961],
               [0.50588235, 0.03529412, 0.35686275],
               [0.51372549, 0.03529412, 0.36470588],
               [0.5254902, 0.03921569, 0.37254902],
               [0.53333333, 0.03921569, 0.38039216],
               [0.54509804, 0.03921569, 0.38823529],
               [0.55686275, 0.03921569, 0.39215686],
               [0.56470588, 0.03921569, 0.4],
               [0.57647059, 0.04313725, 0.40784314],
               [0.58431373, 0.04313725, 0.41568627],
               [0.59607843, 0.04313725, 0.42352941],
               [0.60784314, 0.04313725, 0.43137255],
               [0.61568627, 0.04313725, 0.43529412],
               [0.62745098, 0.04705882, 0.44313725],
               [0.63529412, 0.04705882, 0.45098039],
               [0.64705882, 0.04705882, 0.45882353],
               [0.65490196, 0.04705882, 0.46666667],
               [0.66666667, 0.04705882, 0.4745098],
               [0.67843137, 0.05098039, 0.47843137],
               [0.68627451, 0.05098039, 0.48627451],
               [0.69803922, 0.05098039, 0.49411765],
               [0.70588235, 0.05098039, 0.50196078],
               [0.71764706, 0.05098039, 0.50980392],
               [0.72941176, 0.05490196, 0.51764706],
               [0.73333333, 0.05882353, 0.52156863],
               [0.74117647, 0.06666667, 0.5254902],
               [0.74509804, 0.0745098, 0.52941176],
               [0.75294118, 0.07843137, 0.52941176],
               [0.75686275, 0.08627451, 0.53333333],
               [0.76470588, 0.09019608, 0.5372549],
               [0.76862745, 0.09803922, 0.54117647],
               [0.77647059, 0.10588235, 0.54509804],
               [0.78039216, 0.10980392, 0.54509804],
               [0.78431373, 0.11764706, 0.54901961],
               [0.79215686, 0.12156863, 0.55294118],
               [0.79607843, 0.12941176, 0.55686275],
               [0.80392157, 0.13333333, 0.55686275],
               [0.80784314, 0.14117647, 0.56078431],
               [0.81568627, 0.14901961, 0.56470588],
               [0.81960784, 0.15294118, 0.56862745],
               [0.82745098, 0.16078431, 0.57254902],
               [0.83137255, 0.16470588, 0.57254902],
               [0.83921569, 0.17254902, 0.57647059],
               [0.84313725, 0.18039216, 0.58039216],
               [0.85098039, 0.18431373, 0.58431373],
               [0.85490196, 0.19215686, 0.58431373],
               [0.8627451, 0.19607843, 0.58823529],
               [0.86666667, 0.20392157, 0.59215686],
               [0.87058824, 0.21176471, 0.59215686],
               [0.8745098, 0.21568627, 0.59607843],
               [0.8745098, 0.22352941, 0.59607843],
               [0.87843137, 0.23137255, 0.59607843],
               [0.88235294, 0.23529412, 0.6],
               [0.88627451, 0.24313725, 0.6],
               [0.89019608, 0.24705882, 0.6],
               [0.89411765, 0.25490196, 0.60392157],
               [0.89411765, 0.2627451, 0.60392157],
               [0.89803922, 0.26666667, 0.60392157],
               [0.90196078, 0.2745098, 0.60392157],
               [0.90588235, 0.28235294, 0.60784314],
               [0.90980392, 0.28627451, 0.60784314],
               [0.90980392, 0.29411765, 0.60784314],
               [0.91372549, 0.29803922, 0.61176471],
               [0.91764706, 0.30588235, 0.61176471],
               [0.92156863, 0.31372549, 0.61176471],
               [0.9254902, 0.31764706, 0.61568627],
               [0.9254902, 0.3254902, 0.61568627],
               [0.92941176, 0.33333333, 0.61568627],
               [0.93333333, 0.3372549, 0.61960784],
               [0.9372549, 0.34509804, 0.61960784],
               [0.94117647, 0.34901961, 0.61960784],
               [0.94509804, 0.35686275, 0.62352941],
               [0.94509804, 0.36470588, 0.62352941],
               [0.94901961, 0.36862745, 0.62352941],
               [0.95294118, 0.37647059, 0.62352941],
               [0.95686275, 0.38431373, 0.62745098],
               [0.96078431, 0.38823529, 0.62745098],
               [0.96078431, 0.39607843, 0.62745098],
               [0.96470588, 0.4, 0.63137255],
               [0.96862745, 0.40784314, 0.63137255],
               [0.96862745, 0.41568627, 0.63529412],
               [0.96862745, 0.41960784, 0.63529412],
               [0.96862745, 0.42745098, 0.63921569],
               [0.96862745, 0.43529412, 0.64313725],
               [0.96862745, 0.44313725, 0.64313725],
               [0.97254902, 0.44705882, 0.64705882],
               [0.97254902, 0.45490196, 0.64705882],
               [0.97254902, 0.4627451, 0.65098039],
               [0.97254902, 0.46666667, 0.65490196],
               [0.97254902, 0.4745098, 0.65490196],
               [0.97254902, 0.48235294, 0.65882353],
               [0.97254902, 0.49019608, 0.6627451],
               [0.97254902, 0.49411765, 0.6627451],
               [0.97254902, 0.50196078, 0.66666667],
               [0.97254902, 0.50980392, 0.66666667],
               [0.97647059, 0.51764706, 0.67058824],
               [0.97647059, 0.52156863, 0.6745098],
               [0.97647059, 0.52941176, 0.6745098],
               [0.97647059, 0.5372549, 0.67843137],
               [0.97647059, 0.54117647, 0.68235294],
               [0.97647059, 0.54901961, 0.68235294],
               [0.97647059, 0.55686275, 0.68627451],
               [0.97647059, 0.56470588, 0.68627451],
               [0.97647059, 0.56862745, 0.69019608],
               [0.97647059, 0.57647059, 0.69411765],
               [0.97647059, 0.58431373, 0.69411765],
               [0.98039216, 0.58823529, 0.69803922],
               [0.98039216, 0.59607843, 0.70196078],
               [0.98039216, 0.60392157, 0.70196078],
               [0.98039216, 0.61176471, 0.70588235],
               [0.98039216, 0.61568627, 0.70588235],
               [0.98039216, 0.62352941, 0.70980392],
               [0.98039216, 0.62745098, 0.70980392],
               [0.98039216, 0.63137255, 0.71372549],
               [0.98039216, 0.63921569, 0.71372549],
               [0.98039216, 0.64313725, 0.71372549],
               [0.98039216, 0.64705882, 0.71764706],
               [0.98039216, 0.65098039, 0.71764706],
               [0.98039216, 0.65490196, 0.71764706],
               [0.98431373, 0.6627451, 0.72156863],
               [0.98431373, 0.66666667, 0.72156863],
               [0.98431373, 0.67058824, 0.72156863],
               [0.98431373, 0.6745098, 0.7254902],
               [0.98431373, 0.67843137, 0.7254902],
               [0.98431373, 0.68235294, 0.7254902],
               [0.98431373, 0.69019608, 0.72941176],
               [0.98431373, 0.69411765, 0.72941176],
               [0.98431373, 0.69803922, 0.73333333],
               [0.98431373, 0.70196078, 0.73333333],
               [0.98431373, 0.70588235, 0.73333333],
               [0.98431373, 0.71372549, 0.7372549],
               [0.98431373, 0.71764706, 0.7372549],
               [0.98431373, 0.72156863, 0.7372549],
               [0.98431373, 0.7254902, 0.74117647],
               [0.98431373, 0.72941176, 0.74117647],
               [0.98823529, 0.7372549, 0.74117647],
               [0.98823529, 0.74117647, 0.74509804],
               [0.98823529, 0.74509804, 0.74509804],
               [0.98823529, 0.74901961, 0.74509804],
               [0.98823529, 0.75294118, 0.74901961],
               [0.98823529, 0.75686275, 0.74901961],
               [0.98823529, 0.76470588, 0.74901961],
               [0.98823529, 0.76862745, 0.75294118],
               [0.98823529, 0.77254902, 0.75294118],
               [0.98823529, 0.77647059, 0.75686275],
               [0.98823529, 0.78039216, 0.76078431],
               [0.98823529, 0.78431373, 0.76470588],
               [0.98823529, 0.78431373, 0.76862745],
               [0.98823529, 0.78823529, 0.77254902],
               [0.98823529, 0.79215686, 0.77254902],
               [0.98823529, 0.79607843, 0.77647059],
               [0.98823529, 0.8, 0.78039216],
               [0.98823529, 0.80392157, 0.78431373],
               [0.98823529, 0.80392157, 0.78823529],
               [0.98823529, 0.80784314, 0.79215686],
               [0.98823529, 0.81176471, 0.79607843],
               [0.98823529, 0.81568627, 0.8],
               [0.98823529, 0.81960784, 0.80392157],
               [0.98823529, 0.82352941, 0.80784314],
               [0.99215686, 0.82745098, 0.81176471],
               [0.99215686, 0.82745098, 0.81176471],
               [0.99215686, 0.83137255, 0.81568627],
               [0.99215686, 0.83529412, 0.81960784],
               [0.99215686, 0.83921569, 0.82352941],
               [0.99215686, 0.84313725, 0.82745098],
               [0.99215686, 0.84705882, 0.83137255],
               [0.99215686, 0.84705882, 0.83529412],
               [0.99215686, 0.85098039, 0.83921569],
               [0.99215686, 0.85490196, 0.84313725],
               [0.99215686, 0.85882353, 0.84705882],
               [0.99215686, 0.8627451, 0.84705882],
               [0.99215686, 0.86666667, 0.85098039],
               [0.99215686, 0.86666667, 0.85490196],
               [0.99215686, 0.87058824, 0.85882353],
               [0.99215686, 0.8745098, 0.8627451],
               [0.99215686, 0.87843137, 0.86666667],
               [0.99215686, 0.88235294, 0.87058824],
               [0.99215686, 0.88235294, 0.87058824],
               [0.99215686, 0.88627451, 0.8745098],
               [0.99215686, 0.89019608, 0.87843137],
               [0.99215686, 0.89411765, 0.88235294],
               [0.99215686, 0.89411765, 0.88235294],
               [0.99215686, 0.89803922, 0.88627451],
               [0.99607843, 0.90196078, 0.89019608],
               [0.99607843, 0.90588235, 0.89019608],
               [0.99607843, 0.90588235, 0.89411765],
               [0.99607843, 0.90980392, 0.89803922],
               [0.99607843, 0.91372549, 0.90196078],
               [0.99607843, 0.91764706, 0.90196078],
               [0.99607843, 0.91764706, 0.90588235],
               [0.99607843, 0.92156863, 0.90980392],
               [0.99607843, 0.9254902, 0.90980392],
               [0.99607843, 0.92941176, 0.91372549],
               [0.99607843, 0.92941176, 0.91764706],
               [0.99607843, 0.93333333, 0.91764706],
               [0.99607843, 0.9372549, 0.92156863],
               [0.99607843, 0.94117647, 0.9254902],
               [0.99607843, 0.94117647, 0.92941176],
               [0.99607843, 0.94509804, 0.92941176],
               [1., 0.94901961, 0.93333333],
               [1., 0.95294118, 0.9372549],
               [1., 0.95294118, 0.9372549],
               [1., 0.95686275, 0.94117647],
               [1., 0.96078431, 0.94509804],
               [1., 0.96470588, 0.94901961],
               [1., 0.96470588, 0.94901961],
               [1., 1., 1.]]
    cmap465 = mpl.colors.ListedColormap(colors=cmap465)
    return cmap465


def polpow(im0, order=1, method=0):
  #  pip install polarTransform
    import polarTransform as pT

    m = im0.shape[0]//2
    impolar, Ptsetting = pT.convertToPolarImage(
        im0, finalRadius=m, radiusSize=m, angleSize=360, order=order)
    if method == 0:
        profile = np.median(impolar, axis=0)
    elif method == 2:
        profile = np.max(impolar, axis=0)
    else:
        profile = np.mean(impolar, axis=0)

    return profile, impolar


def hanning_win(size):
    std = size/10
    from scipy import signal
    profile = signal.windows.hann(size)
    win = profile.reshape(-1, 1)*profile
    return win


hwin = hanning_win(448)


def eval_spec(im):
    f = np.abs(np.fft.fftshift(np.fft.fft2(
        (np.maximum(im[800:-800, 800:-800]-im.mean(),0))*hwin)))
    D = disk(448, 448, 20)
    f = f*D

    f = np.maximum(f-f[D].min(), 0)

    Y, X = np.mgrid[-224:224, -224:224]
    im = f
    imsum = im.sum()
    jxx = (X*X*im).sum()/imsum
    jxy = (X*Y*im).sum()/imsum
    jyy = (Y*Y*im).sum()/imsum
    lamda1 = (jxx+jyy+np.sqrt((jxx+jyy)**2-4*(jxx*jyy-jxy*jxy)))
    lamda2 = (jxx+jyy-np.sqrt((jxx+jyy)**2-4*(jxx*jyy-jxy*jxy)))
    e = 2-lamda1/lamda2
    return e


def xcorrcenter(standimage, compimage, R0=2, flag=0):
    # if flag==1,standimage 是FFT以后的图像，这是为了简化整数象元迭代的运算量。直接输入FFT以后的结果，不用每次都重复计算
    try:
        M, N = standimage.shape

        standimage = zscore2(standimage)
        s = fft.fft2(standimage)

        compimage = zscore2(compimage)
        c = np.fft.ifft2(compimage)

        sc = s * c
        im = np.abs(fft.fftshift(fft.ifft2(sc)))  # /(M*N-1);%./(1+w1.^2);
        cor = im.max()
        if cor == 0:
            return 0, 0, 0

        M0, N0 = np.where(im == cor)
        m, n = M0[0], N0[0]

        if flag:
            m -= M / 2
            n -= N / 2
            # 判断图像尺寸的奇偶
            if np.mod(M, 2):
                m += 0.5
            if np.mod(N, 2):
                n += 0.5

            return m, n, cor
        # 求顶点周围区域的最小值
        immin = im[(m - R0):(m + R0 + 1), (n - R0):(n + R0 + 1)].min()
        # 减去最小值
        im = np.maximum(im - immin, 0)
        # 计算重心
        x, y = np.mgrid[:M, :N]
        area = im.sum()
        m = (np.double(im) * x).sum() / area
        n = (np.double(im) * y).sum() / area
        # 归算到原始图像
        m -= M / 2
        n -= N / 2
        # 判断图像尺寸的奇偶
        if np.mod(M, 2):
            m += 0.5
        if np.mod(N, 2):
            n += 0.5
    except:
        print('Err in align_Subpix routine!')
        m, n, cor = 0, 0, 0
    return m, n, cor


def removestrip2(img, order=107):  # 剔除条纹

    f = pywt.wavedec2(img.copy(), wavelet, mode='symmetric', level=wavelevel)
    g = f.copy()
    for i in range(wavelevel):
        (LHY, HLY, HHY) = f[i+1]
        s = medfilt2d(LHY, (1, order))
        LHY = LHY-s
        g[i+1] = (LHY, HLY, HHY)
    im = pywt.waverec2(g, wavelet, mode='symmetric')

    return im


def checkdata(filelist, Flip):
    im = []
    header = []

    for file in tqdm(filelist):

        imorg, hd = fitsread(file)  # 读取fits 文件
       # imorg=removestrip(imorg) #原始信号imorg消除cmos条纹

        if Flip == 0:
            imorg = np.flip(imorg, 0)  # 上下镜像反转
        im.append(imorg)
        header.append(hd)
    im = np.array(im).astype('int16')
    return im, header


def createdark(im, header, filelist, darkcorr):
    print('Wait a moment')

    if darkcorr == 0:  # 如果未做暗场改正
        im_median = np.median(im[:, 512:-512, 512:-512],
                              axis=(-2, -1))  # 计算中间区域的中值
        se = im_median < 130  # 如果中值小于130,认为是暗场。

        if se.sum() == 0:  # 如果暗场数为0，那就找一个旧暗场来用吧
            dark = fitsread('SUTRI_dark_20221111.fits')[0]
            datalist = filelist
            dataheader = header
            darknum = 0

        else:
            dataheader = [header[i]
                          for i in range(len(header)) if ~se[i]]  # 数据文件头列表
            datalist = [filelist[i]
                        for i in range(len(filelist)) if ~se[i]]  # 数据文件名列表
            # darkarr=im[se]
            # darkheader=[header[i] for i in range(len(header)) if se[i]] #数据文件头列表
            # darklist=[filelist[i] for i in range(len(filelist)) if se[i]] #数据文件名列表
            dark = np.median(im[se], axis=0)  # 平均暗场
            im = im[~se]  # 有效数据矩阵
            darknum = se.sum()  # 暗场数目
    else:  # 如果已做暗场
        datalist = filelist
        dataheader = header
        dark = 0
        darknum = 0

    #fitswrite(outfitsdir+'/dark_'+batchname+'.fits.gz', dark.astype('float32'),header=None)

    print('Dark field files: ', darknum, ' Data files: ', len(datalist))
    return im, datalist, dataheader, dark, darknum


def preprocess(im, datalist, dataheader, dark, darkcorr,foldername='_',g=0.005):
    se = np.ones(len(datalist))
    raysnum = []
    dropnum = 0
    backim = []
    for i in tqdm(range(len(datalist))):

        
        imorg = im[i].astype('float32')-dark  # 扣除暗场
        imorg = removestrip(imorg)  # 原始信号imorg消除cmos条纹
        imorg, ray = removeray(imorg, 0.2)  # 扣除宇宙线
        arr = imorg > imorg.mean()
        raysnum.append((ray*arr).sum())  # 有效区域内的宇宙线数量

        imorg = removedarkspots(imorg)  # 扣除无响应点（黑斑）
        sig=detectsignal(imorg)
        #if imorg[512:-512, 512:-512].std() < 20:
        if sig<g:
            se[i] = 0
            dropnum += 1
            if sig<0.0025 :
                s=imorg/imorg[512:-512,512:-512].mean()
                if s.std()<0.2:
                    backim.append(s)
                    print('background image ',sig)
            else:
                print('drop, too dark ',sig)
            continue  # too dark 完全不处理，跳过！！！ 最好记录在案

        im[i] = imorg.astype('int16')
        
    backim = np.array(backim)
    fitswrite('back_'+os.path.basename(foldername)+'.fits', backim.astype('float32'), header=None)
    raysnum = np.array(raysnum)

    se = se > 0
    im = im[se]  # -dropim[np.newaxis]
    dataheader = [dataheader[i] for i in range(len(dataheader)) if se[i]]
    datalist = [datalist[i] for i in range(len(datalist)) if se[i]]
    tot = len(datalist)
   
    print('Pre-process finished. Drop  ', dropnum,
          'images, and there are ', backim.shape[0],' background images. Now there are ', tot, ' Sun images ')
    return im,backim, datalist, dataheader, tot, raysnum, dropnum
def detectsignal(im): #detect if there is the SUN. if m>0.01, there is a SUN. if m<0.003, all noise
    
    w=20
    w2=1
    # f = fft.fft2(im0)
    # fil=makeGaussian(2048,200)
    # f = fft.fftshift(f)*fil
    # im = np.abs(fft.ifft2(f))
    z0=im[800:-800:w2,800:-800:w2]-im[800+w:-800+w:w2,800+w:-800+w:w2]
 
    z1=im[800:-800:w2,800:-800:w2]-im[800:-800:w2,800+w:-800+w:w2]

    z=z0+z1
    s=np.mean(z)
    D=np.abs(z-s)>3*z.std()
    m=D.sum()/(z1.shape[0]*z1.shape[1])

    return m 
def removeback(im,IM):
 

    tot=im.shape[0]
    for i in range(tot):
        imorg=im[i]
        
        
        arr=~disk(2048,2048,900)
        arr2=imorg>0
        
        arr=arr*1*arr2
        arr=arr>0
        K=np.median(imorg[arr])
        #K=mode(np.int16(imorg[arr]))/mode(np.int16(back_tmp[arr]*100))*100
        #print(imorg[arr].mean(),back_tmp[arr].mean(),K)
        #K=mode(np.int16(imorg[arr]*10.0))/10/np.mean(back_tmp[arr])
        im[i]=((IM[i]-K))
        print(K)
    return im
def firstalign(im):
    result = []  # 处理结果列表
    im0=[]
    
    tot=im.shape[0]
    for i in tqdm(range(tot)):

        imorg = im[i].astype('float32')
        
        flag=0
        if flag==0:
            x = imorg-opening(imorg, np.ones((31, 31)))  # 用形态学提取高频结构
            x0 = x > (x.mean()+1*x.std())  # 二值化，均值标准差
            x0 = rs(x0, 15)  # 去掉小面积

            x = np.minimum(x0*x, 500)  # 控制最大值
            x = imnorm(x)*255  # 归255化
            c_hough = cv2.HoughCircles(np.uint8(x), method, 1, 1000, param1=10, param2=10)
            xc_0, yc_0, rsun_pix0 = c_hough.flatten()[:3]  # 圆参数

        else:  # 如果hough 失败，用边缘提取测
            try:
                print('Cannot find the Sun by Hough，But...')
                # 利用arr区域面积重心估计目标的中心位置,sd为边缘像
                sd, xc, yc, arr = initpara(imorg.copy(), X.copy(), Y.copy())
                xc, yc, rsun_pix = fit_edge(imorg.copy(), sd.copy(
                ), X.copy(), Y.copy(), xc.copy(), yc.copy())  # 对区域边缘做圆拟合
                xc_0, yc_0, rsun_pix0, Xl, Yl, sigma_pix = fit_localmax(
                    imorg*arr, xc.copy(), yc.copy(), rsun_pix.copy())  # 利用局部最大值做圆拟合

            except:
                print('Cannot find the Sun  :(')

                continue

        outpara_0 = [MC-xc_0, NC-yc_0]  # 计算和1023.5的偏移量，

        tmp_0 = immove2(imorg, outpara_0[0], outpara_0[1])  # 移动图像至1023.5
        im0.append(tmp_0)
        result.append([outpara_0[0], outpara_0[1], rsun_pix0])
        
        arr = disk(M, N, 900)  # mask


    im0=np.array(im0)
    return im0,result


def dropbadimage(im, IM, datalist, dataheader, FOVlist, qelist, obsJDlist, result, Matf, FOV_T=0.8, qe_T=0.5):
    re=np.array(result)
    bright=re[:,8].astype('float32')
    se = ((bright>20)*(np.array(FOVlist) > FOV_T) *
          (np.array(qelist) > qe_T)) > 0  # select good images

    im = im[se]
    IM = IM[se]  # orginal images
    dataheader = [dataheader[i] for i in range(len(dataheader)) if se[i]]
    datalist = [datalist[i] for i in range(len(datalist)) if se[i]]
    FOVlist = [FOVlist[i] for i in range(len(FOVlist)) if se[i]]
    qelist = [qelist[i] for i in range(len(qelist)) if se[i]]
    obsJDlist = [obsJDlist[i] for i in range(len(obsJDlist)) if se[i]]
    Matf = [Matf[i] for i in range(len(Matf)) if se[i]]
    result = [result[i] for i in range(len(result)) if se[i]]

    tot2 = se.sum()

    return im, IM, datalist, dataheader, FOVlist, qelist, obsJDlist, result, Matf, tot2


def ccalign(im, datalist, dataheader, FOVlist, qelist, obsJDlist, tot, flag=0):
    print('Wait a momnet')
    fov = np.array(FOVlist)  # field of View
    fsm = medfilt(fov, 19)  # smooth
    T = np.median(fov)  # median of FOV
    Matc = []
    se = (fsm > T) * (np.array(qelist) > 0.5)  # very good images

    if se.sum() <= 5:
        se = (fsm > 1) * (np.array(qelist) > 0.5)

    se = se > 0
    f = np.where(se)[0][0]

    x = np.array(obsJDlist)
    xse = x[se]
    nse = ~se
    xnse = x[nse]
    im_c = np.zeros(im.shape).astype('int16')

    obsJD0 = np.median(np.array(obsJDlist))  # stardard JD time

    for i in tqdm(range(tot)):
        shift = drot(obsJD0, obsJDlist[i])  # calculate solar ratation
        shiftx = shift.Tx.arcsec/scale0  # trans to pixel
        shifty = shift.Ty.arcsec/scale0
        dataheader[i]['ROTX'] = shiftx
        dataheader[i]['ROTY'] = shifty

        im_c[i] = immove2(im[i].astype('float32'), -shiftx, -shifty).astype('int16')  # correct solar ratation

    ims = im_c[:, 512:-512, 512:-512].copy()  # center part of im_c
    last = ims[se][-1]
    ims[nse] = interp1d(x[se], ims[se], kind='linear', axis=0, copy=True,
                        bounds_error=False, fill_value=last, assume_sorted=True)(xnse)
    ims[:f] = ims[f]
    ims = ims.reshape(ims.shape[0], -1)
    kz = min(200, ims.shape[0])
    z = np.ones(kz)/kz
    for i in tqdm(range(ims.shape[1])):  # smooth IMS
        ims[:, i] = np.convolve(ims[:, i].astype(
            'float32'), z, mode='same').astype('int16')

    ims = ims.reshape(im_c[:, 512:-512, 512:-512].shape)

    print('Solar rotation is corrected  ')  # ,se.sum())

    if flag == 0:subshift = []
    shift_y, shift_x = 512, 512
    tf_shift = func(translation=[-shift_x, -shift_y]).params
    tf_shift_inv = func(translation=[shift_x, shift_y]).params
    if flag == 0:
        T2 = T+1#-0.02
    else:  # for last fine subpixel alignment stage
        T2 = 0
    #####################################Stage 5, subpix alignment ###################
    for i in tqdm(range(tot)):
        sub = os.path.basename(datalist[i])  # 文件名中去掉目录名
        shiftx = dataheader[i]['ROTX']
        shifty = dataheader[i]['ROTY']
        # 用中间小区域做亚象元相关
        if flag == 0:
            w = 50
        else:  # for last fine subpixel alignment stage
            w = 150

        if fsm[i] > T2:
            dy, dx, cor = xcorrcenter(ims[i, w:-w, w:-w].astype('float32'), im_c[i,512+w:-512-w, 512+w:-512-w].astype('float32'))  # tanslation value by CC

            if flag == 1:
                im[i] = immove2(im[i].astype('float32'), dx, dy).astype('int16')
            Mat = func(translation=[-dx, -dy]).params
            rot = 0
        else:

            d, cor, tform = all_align(ims[i].astype('float32'), im_c[i, 512:-512, 512:-512].astype(
                'float32'), winsize=51, step=20, r_t=2, arrow=0)  # tform by Optical flow
            Mat = np.dot(tf_shift_inv, np.dot(tform.params, tf_shift))
            im[i] = warp(im[i].astype('float32'), tform)
            # if cor < 0.7:
            #     #qelist[i] = cor
            #     dataheader[i]['CORRELAT'] = cor
            rot = d[0]
            dx = Mat[0, 2]
            dy = Mat[1, 2]

        print('%s %.2f %.2f %.2f %.5f %.2f' %
              (sub, FOVlist[i], dx, dy, rot, cor))
        if flag == 0:
            dataheader[i]['SHIFTX'] = dx
            dataheader[i]['SHIFTY'] = dy
            dataheader[i]['CORRELAT'] = cor
            dataheader[i]['C_angle'] = rot
    
            # cross-correlation shift x,y and correlation
            subshift.append([dx, dy, rot, cor])
        Matc.append(Mat)
    if flag == 0:
        return im, datalist, dataheader, subshift, Matc,ims
    else:
        return im, datalist, dataheader,  Matc,ims

def transformimages(IM,Matc3,Matc2, Matc,Matf, tot2):
    for i in tqdm(range(tot2)):
        Mat = np.dot(Matc3[i], Matc3[i])
       
        Mat = np.dot(Mat, Matc[i])
       
        Mat = np.dot(Mat,Matf[i])

        tform = func(matrix=Mat)

        IM[i] = warp(IM[i].astype('float32'), tform, clip=True, cval=0, preserve_range=False)
    return IM,Mat


def writefinaldata(im, datalist, dataheader, tot, outfitsdir):
    for i in tqdm(range(tot)):
        sub = os.path.basename(datalist[i])
        fitswrite(outfitsdir+'/jkf11_'+sub,
                  im[i].astype('int16'), header=dataheader[i])
    return im


def createlog(result, subshift, foldername):
    result = np.array(result)
    subshift = np.array(subshift)
    result = np.hstack((result, subshift))

    plt.close('all')

    drawlog(result, foldername)

    np.save(os.path.basename(foldername)+'_log', result)
    return result, subshift


def selectgoodimage(im, datalist, dataheader, FOVlist, qelist, subshift, tot, FOV_T, qe_T, shift_T, pic, outpicsdir):
    plt.figure()

    goodnum = 0
    badnum = 0
    debug = 1
    K = 0
    for i in tqdm(range(tot)):
        sub = os.path.basename(datalist[i])
        shiftmax = np.max(abs(subshift[i][0:2]))
        # 如果不能满足这个条件，就是一个bad image, 不能作为对齐标准帧。
        if (FOVlist[i] <= FOV_T) or (qelist[i] < qe_T) or (shiftmax > shift_T):
            gooddata = False
            badnum += 1
            if debug:
                print(sub, ' is a BAD image. FOV=',
                      FOVlist[i], 'Q=', qelist[i])
        else:
            gooddata = True
            goodnum += 1

        if pic == 1:
            if gooddata:  # 是否发布
                outpic = outpicsdir
            else:
                outpic = outpicsdir+'/bad'

            tmp_2 = im[i].astype('float32')
            tmp_2 = np.maximum(tmp_2, 0)
            tmp_2 = np.log10(tmp_2+1)  # 对数

            #figure, axes = plt.subplots()
            if K == 0:
                dis = plt.imshow(tmp_2, vmin=1, vmax=3.6, cmap=sutri465())
                plt.gca().invert_yaxis()
                #ax.set_aspect( 1 )
            else:
                dis.set_data(tmp_2)
            # [512:-512,512:-512]

            # plt.plot(Xl-xc+NC,Yl-yc+MC,'.',ms=1) #画园
            # fit_circle = plt.Circle(( MC, NC ), rsun_pix,color='yellow',fill = False,label='Circle Fit center' )
            # axes.set_aspect( 1 )

            # axes.add_artist( fit_circle )

            plt.title(sub+' Q: '+str(qelist[i])[:4])
            plt.show()
            plt.savefig(outpic+'/'+sub+'.png', dpi=300)
            plt.pause(0.1)
            K = 1

    return goodnum, badnum

def sutri_map(outfitsdir,vmin=-10,vmax=300,FOV_T=0.7, qe_T=0.7,Cor_T=0.6):
        fl=sorted(glob.glob(outfitsdir+'\\jkf11*.fits*'))
        filelist=fl[::1]
         
        bar=tqdm(filelist)#,position=0, file=sys.stdout, desc="desc")
        
        rmdir(outfitsdir+'/map')
        
        mkdir(outfitsdir+'/map')
        for file in bar:
           
            sub=os.path.basename(file)
            sut=sunpy.map.Map(file)
            s=sut.fits_header
            if (s['QUALITY']<qe_T) or (s['FIEDVIEW']<FOV_T):# or (s['CORRELAT']<Cor_T): 
                
                continue
            plt.close('all')
            z=np.maximum(sut.data,0)+1
            z=np.log10(z)
            sut=sunpy.map.Map(z,sut.meta)
            sut.plot(clip_interval=(10, 99.9)*u.percent,cmap=sutri465())
            # dis=sut.plot(vmin=vmin,vmax=vmax,cmap=sutri465())
        #    dis=sut.plot(clip_interval=(0, 99)*u.percent,cmap=sutri465())
            sut.draw_limb()
            sut.draw_grid()    
        
            plt.draw()
            plt.pause(0.01)
        
            plt.savefig(outfitsdir+'/map/'+sub+'.png',dpi=150)
            
        dirin='.//'
        batchname = os.path.basename(outfitsdir)
        
        toMP4(dirin,'map_'+batchname,jpgdir=outfitsdir+'/map',fps=10)
