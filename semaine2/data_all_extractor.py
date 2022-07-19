import netCDF4 as net
import numpy as np

netd_tmin= net.Dataset("/Users/billxue/Documents/neural_network/tmin.eval.ERA-Int.CRCM5-OUR.day.NAM-22.raw.nc")
netd_tmax = net.Dataset('/Users/billxue/Documents/neural_network/tmax.eval.ERA-Int.CRCM5-OUR.day.NAM-22.raw.nc')
netd_precip = net.Dataset("/Users/billxue/Documents/neural_network/prec.eval.ERA-Int.CRCM5-OUR.day.NAM-22.raw.nc")

t_min = netd_tmin["tmin"]
t_max = netd_tmax["tmax"]
prec = netd_precip["prec"]
lat_g = netd_tmin["lat"]
lon_g = netd_tmin["lon"]

## hardcode for now we will clean this up later
def absolute_to_relative_converter(corners):
    ## we can make this better with some more informed guesses
    nice_enough = []
    for elements in corners:
        for latitude in range(300):
            for longitude in range(340):
                lat = np.ma.getdata(lat_g[latitude,longitude])
                lon = np.ma.getdata(lon_g[latitude,longitude])
                print(f"latitude: {lat}, long: {lon}")
                if abs(lat - elements[0]) < 2 and abs(lon - elements[1]) < 2:
                    nice_enough.append((latitude,longitude))
                    return(latitude,longitude)
        else:
            print("error,please change tolerance")
                
def data_cropper(relative_coordintes, zone):
    pass
