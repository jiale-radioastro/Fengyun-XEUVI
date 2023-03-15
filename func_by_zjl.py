import datetime
from astropy.time import Time
from aiapy.calibrate import register, update_pointing
from sunpy.map import Map
from sunpy.net import Fido, attrs as attrs
from scipy.interpolate import RectBivariateSpline
from astropy.coordinates import SkyCoord
from astropy.wcs import WCS
from sunpy.coordinates import Helioprojective, propagate_with_solar_surface
from astropy import units as u

def read_timelist(msec,day):
    msec1=msec[:,0]*0.1
    day1=day[:,0]
    date = [datetime.date(2000, 1, 1) + datetime.timedelta(days=int(d)) for d in day1]
    delta = [datetime.timedelta(milliseconds=int(ms)) for ms in msec1]
    timelist = [datetime.datetime.combine(date[i],datetime.time(12,0))+delta[i] for i in range(len(date))]
    
    return Time(timelist)

def give_aiamap(reference_time,aia_dir="",interval=24):
    results = Fido.search(attrs.Time(reference_time,reference_time+datetime.timedelta(seconds=int(interval))),
                          attrs.Instrument("aia"),attrs.Wavelength(193*u.angstrom),attrs.Physobs.intensity)
    if len(results)==0:
        print('AIA image not found in the given interval')
        return []
    download_file=[]
    while download_file==[]:
        try:
            download_file = Fido.fetch(results[0,0], path=aia_dir)
        except:
            continue
    aia_map = Map(download_file[0])
    aia_map=register(update_pointing(aia_map))
    return aia_map

def to1024(img):
    x=np.arange(0,4096,1)
    y=np.arange(0,4096,1)
    f=RectBivariateSpline(x,y,img)
    t0=np.arange(0,1024,1)
    x1=4*t0+1.5
    y1=4*t0+1.5
    return f(x1,y1)

def reduce_aia(a_img,b_img):
    # a_img 4096*4096. b_img 1024*1024
    sz_a=np.shape(a_img)
    sz_b=np.shape(b_img)
    if len(sz_a)==2:
        a_img=np.expand_dims(a_img,axis=0)
    if len(sz_b)==2:
        b_img=np.expand_dims(b_img,axis=0)   
        
    tmp=to1024(a_img[0,:,:])
    a_img0=np.zeros([1,1024,1024])
    a_img0[0,:,:]=tmp
    
    a_img1,re_a=firstalign(a_img0)
    b_img1,re_b=firstalign(b_img)
    scale=4*re_a[0][2]/re_b[0][2]
    offset=2047.5-scale*511.5
    
    x=np.arange(0,4096,1)
    y=np.arange(0,4096,1)
    f=RectBivariateSpline(x,y,a_img[0,:,:])
    t0=np.arange(0,1024,1)
    x1=scale*t0+offset
    y1=scale*t0+offset
    return f(x1,y1),scale

def reduce_aiamap(aia_map,b_img):
    data0=aia_map.data
    header1=aia_map.fits_header.copy()
    data1,scale=reduce_aia(data0,b_img)
    header1['cdelt1'] = scale*0.6
    header1['cdelt2'] = scale*0.6
    header1['CRPIX1'] = 1023/2
    header1['CRPIX2'] = 1023/2
    header1['R_SUN'] = header1['R_SUN']/scale
    return Map((data1,header1))
    
def drot_map(aia_map,out_time):
    out_frame = Helioprojective(observer='earth', obstime=out_time,
                                rsun=aia_map.coordinate_frame.rsun)
    out_center = SkyCoord(0*u.arcsec, 0*u.arcsec, frame=out_frame)
    header = sunpy.map.make_fitswcs_header(aia_map.data.shape,
                                           out_center,
                                           scale=u.Quantity(aia_map.scale))
    out_wcs = WCS(header)
    with propagate_with_solar_surface():
        out_warp = aia_map.reproject_to(out_wcs)
    return out_warp
