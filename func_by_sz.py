import numpy as np
from scipy.ndimage import shift
from scipy.signal import medfilt2d

def despike_xeuvi(image, spike_sig=6.):
    #去除点状噪声函数
    #参数spike_sig为阈值参数，表示对点状噪声的定义程度。若其值过高会使得一些噪声不能被识别，若过低则可能去除掉一些有效信息。
    #输入一个二维数组，输出一个去除点状噪声后的二维数组


    image1 = image
    subimage = image1
    cor_mask = np.zeros_like(image, dtype=np.uint8)    # 建立一个空数组
    med_im = medfilt2d(image, kernel_size=5)    # 进行中值滤波
    dev_im=image.astype(np.int64)-med_im.astype(np.int64)  # 数据与滤波后数据作差（注意要保留.astype(np.int64)，防止变量作差时计算溢出）

    sub_stack = np.abs(np.array([
    shift(subimage.astype(np.int64), (-2, -2)).astype(np.int64) - med_im.astype(np.int64),
    shift(subimage.astype(np.int64), (-2, -1)).astype(np.int64) - med_im.astype(np.int64),
    shift(subimage.astype(np.int64), (-2, 0)).astype(np.int64) - med_im.astype(np.int64),
    shift(subimage.astype(np.int64), (-2, 1)).astype(np.int64) - med_im.astype(np.int64),
    shift(subimage.astype(np.int64), (-2, 2)).astype(np.int64) - med_im.astype(np.int64),
    shift(subimage.astype(np.int64), (-1, -2)).astype(np.int64) - med_im.astype(np.int64),
    shift(subimage.astype(np.int64), (-1, -1)).astype(np.int64) - med_im.astype(np.int64),
    shift(subimage.astype(np.int64), (-1, 0)).astype(np.int64) - med_im.astype(np.int64),
    shift(subimage.astype(np.int64), (-1, 1)).astype(np.int64) - med_im.astype(np.int64),
    shift(subimage.astype(np.int64), (-1, 2)).astype(np.int64) - med_im.astype(np.int64),
    shift(subimage.astype(np.int64), (0, -2)).astype(np.int64) - med_im.astype(np.int64),
    shift(subimage.astype(np.int64), (0, -1)).astype(np.int64) - med_im.astype(np.int64),
    shift(subimage.astype(np.int64), (0, 0)).astype(np.int64) - med_im.astype(np.int64),
    shift(subimage.astype(np.int64), (0, 1)).astype(np.int64) - med_im.astype(np.int64),
    shift(subimage.astype(np.int64), (0, 2)).astype(np.int64) - med_im.astype(np.int64),
    shift(subimage.astype(np.int64), (1, -2)).astype(np.int64) - med_im.astype(np.int64),
    shift(subimage.astype(np.int64), (1, -1)).astype(np.int64) - med_im.astype(np.int64),
    shift(subimage.astype(np.int64), (1, 0)).astype(np.int64) - med_im.astype(np.int64),
    shift(subimage.astype(np.int64), (1, 1)).astype(np.int64) - med_im.astype(np.int64),
    shift(subimage.astype(np.int64), (1, 2)).astype(np.int64) - med_im.astype(np.int64),
    shift(subimage.astype(np.int64), (2, -2)).astype(np.int64) - med_im.astype(np.int64),
    shift(subimage.astype(np.int64), (2, -1)).astype(np.int64) - med_im.astype(np.int64),
    shift(subimage.astype(np.int64), (2, 0)).astype(np.int64) - med_im.astype(np.int64),
    shift(subimage.astype(np.int64), (2, 1)).astype(np.int64) - med_im.astype(np.int64),
    shift(subimage.astype(np.int64), (2, 2)).astype(np.int64) - med_im.astype(np.int64)]
    )) # 对数据进行各方向平移并与滤波数据作差，最后sub_stack形成一个[25,x_dimension,y_dimension]的数组


    mad_im = np.median(sub_stack, axis=0)  # 对sub_stack的第一个维度取中值，得到一个[x_dimension,y_dimension]数组
    cor_mask = (dev_im > spike_sig * 1.483 * mad_im).astype(np.uint8)  # 寻找点状噪声
    subimage[np.where(cor_mask)] = med_im[np.where(cor_mask)]  # 将点状噪声处的值设置为中值滤波值

    return subimage

def find_straight_line(reg, n=13, corr=0.97):
    #寻找线性噪声函数
    #输入: reg二维数组，n和corr为噪声阈值参数，具体参考原理文档
    
    reg2 = reg.copy()
    xmax = np.zeros(n, dtype=float)
    ymax = np.zeros(n, dtype=float)

    for i in range(n): #寻找数组内数值最大的n个点
        ix, iy = np.unravel_index(reg.argmax(), reg.shape)
        xmax[i] = ix
        ymax[i] = iy
        reg[ix, iy] = 0
    
    if xmax.all() == 0: xmax=np.arange(1,len(xmax)+1)
    result = np.polyfit(xmax, ymax, 1) #对最大的n个点进行线性拟合
    if abs(np.corrcoef(xmax,ymax))[0,1] > corr: #若线性度高于所设阈值corr，则认为是线性噪声，并进行去噪
        aa = result[1]
        bb = result[0]
        xx = np.arange(reg.shape[0])
        yy = aa + bb * xx
        yy=np.array(yy)

        index = np.logical_and(yy <= reg.shape[1], yy >= 0).astype(np.uint8)
        xxx=[xx[i] for i in range(len(xx)) if index[i] == 1]
        yyy=[yy[i] for i in range(len(yy)) if index[i] == 1]
        xxx=np.array(xxx)
        yyy=np.array(yyy)
        reg = reg2
        for i in range(0,xxx.shape[0]):  #将线噪声及附近点平滑
            reg2[xxx[i], int(yyy[i])] = reg.mean()
            if (int(yyy[i]))+1 < reg2.shape[1]: reg2[xxx[i], int(yyy[i])+1] = reg.mean()
            if (int(yyy[i]))+2 < reg2.shape[1]: reg2[xxx[i], int(yyy[i])+2] = reg.mean()
            if (int(yyy[i]))-1 >= 0 : reg2[xxx[i], int(yyy[i])-1] = reg.mean()
            if (int(yyy[i]))-2 >= 0 : reg2[xxx[i], int(yyy[i])-2] = reg.mean()
      
    return reg2


def deline_xeuvi(image):
    #去除线性噪声
    #输入二维数组，输出去除线性噪声后的数组
    image2=np.copy(image)
    image3=np.copy(image)
    x = np.arange(32) * 32
    y = np.copy(x)
    sz = np.shape(x)


    for j in range(0, sz[0]-1): #将1024*1024的数据分为一个个32*32的小格子，分别进行线性噪声查找
        xxxx = x[j]+16
        for jj in range(0, sz[0]-1):
            yyyy = y[jj]
            reg = image[xxxx:xxxx+31, yyyy:yyyy+31]
            reg2 = find_straight_line(reg) 
            image2[xxxx:xxxx+31, yyyy:yyyy+31] = reg2

    for j in range(0, sz[0]):  #同样分为32*32的小格子，但格点位置向右平移16个像素，再进行一次线性噪声查找
        xxxx = x[j]
        for jj in range(0, sz[0]):
            yyyy = y[jj]
            reg = image2[xxxx:xxxx+31, yyyy:yyyy+31]
            reg2 = find_straight_line(reg) 
            image3[xxxx:xxxx+31, yyyy:yyyy+31] = reg2
    return image3


from tqdm import tqdm
def rm_noise(ims):
    lens = ims.shape[0]
    out = ims.copy()
    for i in tqdm(range(lens)):
        tmp = deline_xeuvi(ims[i])
        tmp = despike_xeuvi(tmp)
        out[i] = tmp
    return out