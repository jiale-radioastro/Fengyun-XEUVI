import datetime
import numpy as np
from tqdm import tqdm
import h5py
from requests_html import HTMLSession
import wget
from astropy.time import Time
from astropy.coordinates import SkyCoord
from astropy.wcs import WCS
from astropy import units as u
from aiapy.calibrate import register, update_pointing
from sunpy.map import Map,make_fitswcs_header
from sunpy.net import Fido, attrs as attrs
from sunpy.coordinates import Helioprojective, propagate_with_solar_surface, sun
from scipy.interpolate import RectBivariateSpline
from scipy.ndimage import gaussian_filter


def read_timelist(msec,day):
    msec1=msec[:,0]*0.1
    day1=day[:,0]
    date = [datetime.date(2000, 1, 1) + datetime.timedelta(days=int(d)) for d in day1]
    delta = [datetime.timedelta(milliseconds=int(ms)) for ms in msec1]
    timelist = [datetime.datetime.combine(date[i],datetime.time(12,0))+delta[i] for i in range(len(date))]
    
    return Time(timelist)

def give_aiamap(reference_time,aia_dir="",method='jsoc'):
    if method=='jsoc':
        download_file=jsoc_download(reference_time,aia_dir=aia_dir)
    elif method=='fido':
        download_file=fido_download(reference_time,aia_dir=aia_dir)
    else:
        print('method输入错误')
        download_file=[]
    if len(download_file)==0:
        print('未成功下载AIA数据')
        return None
    else:
        aia_map = Map(download_file)
        if method=='fido':
            aia_map=register(update_pointing(aia_map))
        print('成功下载'+str(aia_map.date)+'时刻AIA数据')
    return aia_map

def fido_download(reference_time,aia_dir="",max_attempts=6):
    results = Fido.search(attrs.Time(reference_time-0.5*u.hour,reference_time+0.5*u.hour),
                          attrs.Instrument("aia"),attrs.Wavelength(193*u.angstrom),attrs.Physobs.intensity)
    if len(results[0])!=0:
        index=find_neartime(reference_time,results[0,:]['Start Time'])
        download_file=[]
        attempt=0
        while len(download_file)==0 and attempt<max_attempts:
            try:
                download_file = wget.download(results[0,index],aia_dir)
            except:
                download_file=[]
            if len(download_file)==0:
                attempt+=1
        if attempt==max_attempts:
            print('已达最大尝试次数，数据下载失败')
            return []
        return download_file[0]
    else:
        print('半小时附近未找到AIA图像，放弃下载AIA数据')
        return []


def jsoc_download(reference_time,aia_dir="",max_attempts=6):
    year=reference_time.datetime.year
    month=reference_time.datetime.month
    day=reference_time.datetime.day
    hour=reference_time.datetime.hour
    
    jsoc_url='http://jsoc.stanford.edu/data/aia/synoptic/'
    url=jsoc_url+str(year)+'/'+str(month).zfill(2)+'/'+str(day).zfill(2)+'/H'+str(hour).zfill(2)+'00/'
    session = HTMLSession()
    r = session.get(url)
    links193=[]
    for link in r.html.links:
        if len(link)>10 and link[-9:-1]=='0193.fit':
            links193.append(link)
    
    if len(links193)==0:
        jsoc_url='http://jsoc.stanford.edu/data/aia/synoptic/nrt/'
        url=jsoc_url+str(year)+'/'+str(month).zfill(2)+'/'+str(day).zfill(2)+'/H'+str(hour).zfill(2)+'00/'
        session = HTMLSession()
        r = session.get(url)
        for link in r.html.links:
            if len(link)>10 and link[-9:-1]=='0193.fit':
                links193.append(link)

    links193=sorted(links193)
    
    if len(links193)==0:
        print('该小时内未找到AIA图像，放弃下载AIA数据')
        return []

    links_time=[]
    for link in links193:
        year=link[3:7]
        month=link[7:9]
        day=link[9:11]
        hour=link[12:14]
        minute=link[14:16]
        link_time=Time(year+'-'+month+'-'+day+'T'+hour+':'+minute+':00',scale='utc',format='isot')
        links_time.append(link_time)
    index=find_neartime(reference_time,links_time)
    link=links193[index]
    
    download_file=[]
    attempt=0
    while len(download_file)==0 and attempt<max_attempts:
        try:
            download_file = wget.download(url+link,aia_dir)
        except:
            download_file=[]
        if len(download_file)==0:
            attempt+=1
    if attempt==max_attempts:
        print('已达最大尝试次数，数据下载失败')
        return []
    return download_file

def find_neartime(time0,timelist):
    mjd0=time0.mjd
    mjdlist=Time(timelist).mjd
    diflist=np.abs(mjdlist-mjd0)
    return np.where(diflist==np.min(diflist))[0][0]

def to1024(img,scale):
    sz=np.shape(img)
    img=gaussian_filter(img,(scale,scale),0) #高斯滤波，平滑AIA图像
    x=np.arange(0,sz[0],1)
    y=np.arange(0,sz[1],1)
    f=RectBivariateSpline(x,y,img)
    t0=np.arange(0,1024,1)
    offset=(sz[0]-1)/2-scale*511.5
    x1=scale*t0+offset
    y1=scale*t0+offset
    return f(x1,y1)

def reduce_aiamap(aia_map,scale=2.46/0.6):
    data0=aia_map.data
    header1=aia_map.fits_header.copy()
    data1=to1024(data0,scale)
    header1['cdelt1'] = scale*header1['cdelt1']
    header1['cdelt2'] = scale*header1['cdelt2']
    header1['CRPIX1'] = 1023/2
    header1['CRPIX2'] = 1023/2
    header1['R_SUN'] = header1['R_SUN']/scale
    return Map((data1,header1)),header1['cdelt1']
    
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
    data1=out_warp.data
    header1=aia_map.fits_header
    header1['date-obs']=str(out_time)
    header1['CRLN_OBS']=aia_map.observer_coordinate.lon.deg+sun.L0(time=out_time).deg
    header1['CRLT_OBS']=aia_map.observer_coordinate.lat.deg
    header1['DSUN_OBS']=aia_map.observer_coordinate.radius.m
    return Map((data1,header1))

def separate_orbit(filelist,int_hour=1/3,maxframe=770):
    # 将数据文件打包为不同轨的数据文件
    # 对于间隔观测，间隔时间超过int_hour，自动划为新的一轨
    # 对于连续观测，数据文件累计超过770帧（约1.5h），自动划为一轨
    filelist_packed=[]
    filelist_orbit=[]
    timelist=[]
    for file in tqdm(filelist):
        f=h5py.File(file,'r')
        msec=f['/Time/Msec_Count'][:]
        day=f['/Time/Day_Count'][:]
        if len(msec)>0:
            tmp_timelist=read_timelist(msec,day)
            if len(timelist)>0 and ((tmp_timelist[0]-timelist[-1]).value>1/24*int_hour or len(filelist_orbit)>maxframe):
                filelist_packed.append(filelist_orbit)
                filelist_orbit=[file]
            else:
                filelist_orbit.append(file)
            timelist.extend(tmp_timelist)
        if file==filelist[-1] and len(filelist_orbit)>0:
            filelist_packed.append(filelist_orbit)
    return filelist_packed

def make_euvmap(data,header):
    return Map((data,header))

