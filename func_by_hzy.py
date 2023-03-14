import numpy as np

def _predict_(X):#[samples,4]
    coef = np.array([9.07331219e-03,4.83963960e+00,-1.58571516e+00,-1.64596653e-03,-3.46679025e-01,1.42157963e+00])#max,mean,median,std
    bias = -2.02218094

    out = X@coef+bias
    
    return 1 if out>0 else -1

def _predict2_(pic,th):
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

def predict(img,th):#输入图片，输出为1为有完整图像，-1为残缺图像
    img = img.astype(np.float32)/th
    feature = np.array([np.max(img),np.median(img),np.std(img),np.max(img[256:-256,256:-256]),np.median(img[256:-256,256:-256]),np.std(img[256:-256,256:-256])])
    if _predict_(feature)==1:
        return _predict2_(img,th=th)
    else:
        return -1


def predict_day(imglist,if1024=0):#输入一个list，包含一天的图像[img1,img2,...]
    #4月份的数据不是1024*1024,如果输入这样的数据，选择if1024=1，进行裁剪
    il = np.array(imglist)
    if if1024!=1:
        il = il[:,24:-24, 4:-4]
    med1 = np.median(il[:,256:-256,256:-256],(1,2))#每张图片中间部分的中值
    th = np.median(med1)#所有中值的中值
    out=[]
    for i in tqdm(range(il.shape[0])):
        out.append(predict(il[i],th))
    return out#输出list中为1的图像是有完整太阳像，-1是没有
