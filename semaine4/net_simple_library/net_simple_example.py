import net_simple
import netCDF4 as net
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
from datetime import datetime, timedelta


netd_tmin = net.Dataset("/Users/billxue/Documents/neural_network/tmin.eval.ERA-Int.CRCM5-OUR.day.NAM-22.raw.nc")

## real coordinates
map_rc = [37,65,-85,-50] ##[r_min_lat, r_max_lat, r_min_lon, r_max_lon]
## relative coordinates
data_rc = [120,180,200,260] 

bob = net_simple.onev_analysis(netd_tmin["tmin"], netd_tmin["time"],data_rc, map_rc, netd_tmin["lat"], netd_tmin["lon"])
average_2 = bob.basic_ops([datetime(1979,1,1), datetime(1979,1,3)])
mine = bob.basic_ops([datetime(1979,1,1), datetime(1979,1,3)], operation = "min")
##basic_stats = bob.basic_stats(), this one is very heavy

winter = bob.season_analysis(0,mode = "min",years = None)
bob.mapper(winter, "winter", "temperature", path = "/Users/billxue/")