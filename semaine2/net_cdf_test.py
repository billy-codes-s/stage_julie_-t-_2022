#!/usr/bin/env python
# coding: utf-8

# In[64]:


import netCDF4 as net
import numpy as np

netd_tmin= net.Dataset("/Users/billxue/Documents/neural_network/tmin.eval.ERA-Int.CRCM5-OUR.day.NAM-22.raw.nc")
netd_tmax = net.Dataset('/Users/billxue/Documents/neural_network/tmax.eval.ERA-Int.CRCM5-OUR.day.NAM-22.raw.nc')
netd_precip = net.Dataset("/Users/billxue/Documents/neural_network/prec.eval.ERA-Int.CRCM5-OUR.day.NAM-22.raw.nc")

t_min = netd_tmin["tmin"]
t_max = netd_tmax["tmax"]
prec = netd_precip["prec"]
lat_g = netd_tmin["lat"][:]
lon_g = netd_tmin["lon"][:]


# In[3]:


get_ipython().run_line_magic('matplotlib', 'inline')


# In[6]:


import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.basemap import Basemap


# In[20]:



map = Basemap(projection='merc',llcrnrlon=-85.,llcrnrlat=45.,urcrnrlon=-50.,urcrnrlat=65.,resolution='i') 
# projection, lat/lon extents 
##and resolution of polygons to draw
# resolutions: c - crude, l - low, i - intermediate, h - high, f - full


# In[34]:


map.drawcoastlines()
map.drawstates()
map.drawcountries()


# In[106]:


parallels = np.arange(7,65,5.) # make latitude lines ever 5 degrees from 30N-50N
meridians = np.arange(-185,-6,10.) # make longitude lines every 5 degrees from 95W to 70W
map = Basemap(projection='merc',llcrnrlon=-183.,llcrnrlat=7.,urcrnrlon=-6.,urcrnrlat=65.,resolution='i') 
map.drawparallels(parallels,labels=[1,0,1,0],fontsize=10)
map.drawmeridians(meridians,labels=[0,1,0,1],fontsize=10)
map.drawcoastlines()
map.drawstates()


# In[42]:


meridians


# In[55]:


xvalues = np.array([x for x in range(10)])
yvalues = np.tanh(np.array([x for x in range(10)]))


# In[56]:


xx, yy = np.meshgrid(xvalues, yvalues)


# In[57]:


yy


# In[58]:


plt.plot(xx, yy, marker='.', color='k', linestyle='none')


# In[60]:


np.arange(20,50, .2)
## really just a fascinating tool that can be used to great effect


# In[65]:


lat_g


# In[66]:


lon_g


# In[76]:


np.max(netd_tmin["lat"])


# In[78]:


np.min(netd_tmin["lat"])


# In[77]:


np.max(netd_tmin["lon"])


# In[82]:


netd_tmin["lon"][:]


# In[70]:


netd_tmin


# In[98]:


def abs_to_rc(corners):
    ## we can make this better with some more informed guesses
    counter = 0
    for elements in corners:
        for latitude in range(300):
            for longitude in range(340):
                lat = np.ma.getdata(lat_g[latitude,longitude])
                lon = np.ma.getdata(lon_g[latitude,longitude])
                if counter%100 == 0:
                    print(f"latitude: {lat}, long: {lon}")
                if abs(lat - elements[0]) < 1 and abs(lon - elements[1]) < 1:
                    return(latitude,longitude)
                counter+=1
        else:
            print("error,please change tolerance")


# In[99]:


abs_to_rc([[65, -85]])


# In[102]:


print(netd_tmin["lat"][217,178])
print(netd_tmin["lon"][217,178])


# In[101]:


abs_to_rc([[45, -50]])


# In[103]:


print(netd_tmin["lat"][166,297])
print(netd_tmin["lon"][166,297])


# In[104]:


netd_precip


# In[113]:


day_qc_map = netd_precip["prec"][1,45:166,178:297]


# ## Data extraction and pointing form the netcdf_files

# In[220]:


import netCDF4 as net
import numpy as np

netd_tmin= net.Dataset("/Users/billxue/Documents/neural_network/tmin.eval.ERA-Int.CRCM5-OUR.day.NAM-22.raw.nc")
netd_tmax = net.Dataset('/Users/billxue/Documents/neural_network/tmax.eval.ERA-Int.CRCM5-OUR.day.NAM-22.raw.nc')
netd_precip = net.Dataset("/Users/billxue/Documents/neural_network/prec.eval.ERA-Int.CRCM5-OUR.day.NAM-22.raw.nc")

t_min = netd_tmin["tmin"]
t_max = netd_tmax["tmax"]
prec = netd_precip["prec"]
lat_g = netd_tmin["lat"][:]
lon_g = netd_tmin["lon"][:]


# ## if you need to find some coordinates from absolute to relative, you can use this (bad, brute force) tool

# In[219]:


def abs_to_rc(corners):
    ## we can make this better with some more informed guesses
    counter = 0
    for elements in corners:
        for latitude in range(300):
            for longitude in range(340):
                lat = np.ma.getdata(lat_g[latitude,longitude])
                lon = np.ma.getdata(lon_g[latitude,longitude])
                if counter%8000 == 0:
                    print(f"latitude: {lat}, long: {lon}")
                if abs(lat - elements[0]) < 1 and abs(lon - elements[1]) < 1:
                    return(latitude,longitude)
                counter+=1
        else:
            print("error,please change tolerance")
            
coor_1 = abs_to_rc([[65, -85]])
coor_2 = abs_to_rc([[45, -50]])


# ## relative mesh (if we need it for some reason) and data, pretty useless, we don't need this 

# In[177]:


qc_lat_rel= np.arange(150,217,1)
qc_long_rel= np.arange(190, 280, 1)
qc_lat_rel = np.resize(qc_lat_rel,(1,qc_lat_rel.shape[0]))
qc_long_rel = np.resize(qc_long_rel, (1, qc_long_rel.shape[0]))
xx, yy = np.meshgrid(qc_lat_rel, qc_long_rel)
xx.shape


# In[147]:


print(qc_lat_rel)
print(qc_long)


# ## preparing the data on the map (average? sum? things like that)

# In[229]:



prec_data_area = (1/3000) * np.sum(prec[1:3000,130:217,190:280], axis = 0)


# ## absolute mesh to locate_on_map and drawing of map

# In[277]:


## these are the coordinates (lat, long) of every point
qc_lat = netd_tmin["lat"][130:217, 190:280]
qc_lon = netd_tmin["lon"][130:217, 190:280]
print(qc_lat.shape)
print(qc_lon.shape)

## here we make it into.. coordinates (like cartasien if I had to guess).. I guess?
x,y = map(qc_lon,qc_lat)

# you have to update both because changing the paramaters of the map will change the coordinates
map = Basemap(projection='merc',llcrnrlon=-90.,llcrnrlat=35.,urcrnrlon=-45.,urcrnrlat=65.,resolution='i')
prep = map.contourf(x,y,prec_data_area)
cb = map.colorbar(prep, "bottom", size = "5%", pad = "2%")
parallels = np.arange(7,65,5.) # make latitude lines ever 5 degrees from 30N-50N
meridians = np.arange(-185,-6,10.) # make longitude lines every 5 degrees from 95W to 70W 
map.drawcoastlines()
map.drawstates()
plt.title("prec qc test 1")
cb.set_label('mm de pluie')
plt.savefig('/Users/billxue/Documents/expl.pdf', format='pdf', dpi=1200)
plt.show()
## you have to run it in the same cell, supposedly I have no clue why do you think it creates a new plot obj??


# In[ ]:




