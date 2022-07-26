#!/usr/bin/env python
# coding: utf-8

# # Data extraction

# In[4]:


import netCDF4 as net
import numpy as np

netd_tmin = net.Dataset("/Users/billxue/Documents/neural_network/tmin.eval.ERA-Int.CRCM5-OUR.day.NAM-22.raw.nc")
netd_tmax = net.Dataset('/Users/billxue/Documents/neural_network/tmax.eval.ERA-Int.CRCM5-OUR.day.NAM-22.raw.nc')
netd_precip = net.Dataset("/Users/billxue/Documents/neural_network/prec.eval.ERA-Int.CRCM5-OUR.day.NAM-22.raw.nc")

time = netd_tmin["time"][:]
t_min = netd_tmin["tmin"]
t_max = netd_tmax["tmax"]
prec = netd_precip["prec"]
lat_g = np.ma.getdata(netd_tmin["lat"][:])
lon_g = np.ma.getdata(netd_tmin["lon"][:])


# # Mapping

# #### Libraries for mapping

# In[5]:


import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.basemap import Basemap


# #### Coordinates and setting basemap

# In[6]:


## we figured out these coordinates by trial and error
r_min_lat =120
r_max_lat =180
r_min_lon =200
r_max_lon =260

qc_lat = lat_g[r_min_lat:r_max_lat, r_min_lon:r_max_lon]
qc_lon = lon_g[r_min_lat:r_max_lat, r_min_lon:r_max_lon]
map_lat = [37, 65]
map_lon = [-85, -50]

parallels = np.arange(map_lat[0],map_lat[1],5) # make latitude lines ever 5 degrees from 30N-50N
meridians = np.arange(map_lon[0],map_lon[1],10) # make longitude lines every 5 degrees from 95W to 70W

map = Basemap(projection='merc',llcrnrlon=map_lon[0],llcrnrlat=map_lat[0],urcrnrlon=map_lon[1],urcrnrlat=map_lat[1],resolution='i')
x,y = map(qc_lon,qc_lat)
map.drawcoastlines()
map.drawstates()
map.drawparallels(parallels,labels=[1,0,1,0],fontsize=10)
map.drawmeridians(meridians,labels=[0,1,0,1],fontsize=10)


# #### drawing contour and filling with the appropriate data

# In[7]:


prec_data_area = t_max[13,r_min_lat:r_max_lat, r_min_lon:r_max_lon]
prep = map.contourf(x,y,prec_data_area)
cb = map.colorbar(prep, "bottom", size = "5%", pad = "2%")
map.drawcoastlines()
map.drawstates()
map.drawparallels(parallels,labels=[1,0,1,0],fontsize=10)
map.drawmeridians(meridians,labels=[0,1,0,1],fontsize=10)


# # Analysis

# #### time conversion module

# In[8]:


from datetime import datetime, timedelta
def date_generator(unit, datetimes, base_date = datetime(1979, 1, 1, hour = 12), delta = timedelta(days = 1)):
    ## this generator includes both dates
    datetimes.sort()
    
    for datet in datetimes:
        if datet < base_date:
            return 0
    time_diff = (datetimes[1] - datetimes[0])/delta
    date_start = datetimes[0]
    array_of_dates = [date_start]
    for elements in range(int(time_diff)):
        date_start += delta
        array_of_dates.append(date_start)
    return np.ma.getdata(net.date2num(np.array(array_of_dates), unit))
    


# ## temperature max (4 seasons)

# ### winter max average all

# In[9]:


unit = netd_tmin["time"].units
dates_nums = date_generator(unit,[datetime(1980, 12, 21, hour = 12), datetime(1981, 3, 20, hour = 12)])


# In[10]:


net.num2date(time[-1],"days since 1979-1-1 00:00" )


# In[11]:


date_1 = int(net.date2num(datetime(1980, 12, 21, hour = 12), unit)-0.5)
date_2 = int(net.date2num(datetime(1981, 3, 20, hour = 12), unit)-0.5)
delta = datetime(1981, 3, 20, hour = 12) - datetime(1980, 12, 21, hour = 12)


# In[ ]:





# In[12]:


total_deltas = 0
total_temps = np.zeros((qc_lat.shape[0], qc_lat.shape[1]))
for elements in range(34):
    date_1 = int(net.date2num(datetime(1980 + elements, 12, 21, hour = 12), unit)-0.5)
    date_2 = int(net.date2num(datetime(1981 + elements, 3, 20, hour = 12), unit)-0.5)
    sum_current = np.sum(t_max[date_1:date_2,r_min_lat:r_max_lat, r_min_lon:r_max_lon], axis = 0)
    delta = date_2 - date_1
    total_deltas += delta
    total_temps += sum_current
    print(elements, net.num2date(date_1 + 0.5, unit),"----->", net.num2date(date_2 + 0.5, unit))


# In[146]:


average_winter_all = total_temps/total_deltas


# In[147]:


tmax = map.contourf(x,y,average_winter_all)
cb = map.colorbar(tmax, "bottom", size = "5%", pad = "2%")
map.drawcoastlines()
map.drawstates()
map.drawparallels(parallels,labels=[1,0,1,0],fontsize=10)
map.drawmeridians(meridians,labels=[0,1,0,1],fontsize=10)


# In[71]:


days = datetime(1980, 3, 20, hour = 12)- datetime(1979, 12, 21, hour = 12)


# In[76]:


days


# ### spring max average all

# In[149]:


total_deltas = 0
total_temps = np.zeros((qc_lat.shape[0], qc_lat.shape[1]))
for elements in range(34):
    date_1 = int(net.date2num(datetime(1981 + elements, 3, 20, hour = 12), unit)-0.5)
    date_2 = int(net.date2num(datetime(1981 + elements, 6, 20, hour = 12), unit)-0.5)
    sum_current = np.sum(t_max[date_1:date_2,r_min_lat:r_max_lat, r_min_lon:r_max_lon], axis = 0)
    delta = date_2 - date_1
    total_deltas += delta
    total_temps += sum_current
    print(elements, net.num2date(date_1 + 0.5, unit),"----->", net.num2date(date_2 + 0.5, unit))


# In[150]:


average_winter_all = total_temps/total_deltas
tmax = map.contourf(x,y,average_winter_all)
cb = map.colorbar(tmax, "bottom", size = "5%", pad = "2%")
map.drawcoastlines()
map.drawstates()
map.drawparallels(parallels,labels=[1,0,1,0],fontsize=10)
map.drawmeridians(meridians,labels=[0,1,0,1],fontsize=10)


# ### summer max average all

# In[151]:


total_deltas = 0
total_temps = np.zeros((qc_lat.shape[0], qc_lat.shape[1]))
for elements in range(34):
    date_1 = int(net.date2num(datetime(1981 + elements, 6, 20, hour = 12), unit)-0.5)
    date_2 = int(net.date2num(datetime(1981 + elements, 9, 22, hour = 12), unit)-0.5)
    sum_current = np.sum(t_max[date_1:date_2,r_min_lat:r_max_lat, r_min_lon:r_max_lon], axis = 0)
    delta = date_2 - date_1
    total_deltas += delta
    total_temps += sum_current
    print(elements, net.num2date(date_1 + 0.5, unit),"----->", net.num2date(date_2 + 0.5, unit))


# In[152]:


average_winter_all = total_temps/total_deltas
tmax = map.contourf(x,y,average_winter_all)
cb = map.colorbar(tmax, "bottom", size = "5%", pad = "2%")
map.drawcoastlines()
map.drawstates()
map.drawparallels(parallels,labels=[1,0,1,0],fontsize=10)
map.drawmeridians(meridians,labels=[0,1,0,1],fontsize=10)


# ### autumn max average all

# In[153]:


total_deltas = 0
total_temps = np.zeros((qc_lat.shape[0], qc_lat.shape[1]))
for elements in range(34):
    date_1 = int(net.date2num(datetime(1981 + elements, 9, 22, hour = 12), unit)-0.5)
    date_2 = int(net.date2num(datetime(1981 + elements, 12, 21, hour = 12), unit)-0.5)
    sum_current = np.sum(t_max[date_1:date_2,r_min_lat:r_max_lat, r_min_lon:r_max_lon], axis = 0)
    delta = date_2 - date_1
    total_deltas += delta
    total_temps += sum_current
    print(elements, net.num2date(date_1 + 0.5, unit),"----->", net.num2date(date_2 + 0.5, unit))


# In[154]:


average_winter_all = total_temps/total_deltas
tmax = map.contourf(x,y,average_winter_all)
cb = map.colorbar(tmax, "bottom", size = "5%", pad = "2%")
map.drawcoastlines()
map.drawstates()
map.drawparallels(parallels,labels=[1,0,1,0],fontsize=10)
map.drawmeridians(meridians,labels=[0,1,0,1],fontsize=10)


# ## tmin average (4 seasons)

# In[17]:


def season_average(season_index, dataset, time, unit):
    ## solstice 
    seasons = [(12,21,12), (3,21,12), (6,21,12), (9,21,12), (12,21,12)]
    season = seasons[season_index]
    season_next = seasons[season_index + 1]
    
    ## figuring out time 
    start_date = net.num2date(time[0], unit)
    end_date = net.num2date(time[-1], unit)
    
    years = end_date.year - (start_date.year + 1)
    
    total_deltas = 0
    total_temps = np.zeros((x.shape[1], y.shape[1]))
    
    if season_index == 0:
        for elements in range(years):
            date_1 = int(net.date2num(datetime(start_date.year + elements, season[0], season[1], hour = season[2]), unit)-0.5)
            date_2 = int(net.date2num(datetime(start_date.year + 1 + elements, season_next[0], season_next[1], hour = season_next[2]), unit)-0.5)
            sum_current = np.sum(dataset[date_1:date_2,r_min_lat:r_max_lat, r_min_lon:r_max_lon], axis = 0)
            delta = date_2 - date_1
            total_deltas += delta
            total_temps += sum_current  
            print(elements, net.num2date(date_1 + 0.5, unit),"----->", net.num2date(date_2 + 0.5, unit))
            
    else:
        for elements in range(years):
            date_1 = int(net.date2num(datetime(start_date.year + 1 + elements, season[0], season[1], hour = season[2]), unit)-0.5)
            date_2 = int(net.date2num(datetime(start_date.year + 1 + elements, season_next[0], season_next[1], hour = season_next[2]), unit)-0.5)
            sum_current = np.sum(dataset[date_1:date_2,r_min_lat:r_max_lat, r_min_lon:r_max_lon], axis = 0)
            delta = date_2 - date_1
            total_deltas += delta
            total_temps += sum_current  
            print(elements, net.num2date(date_1 + 0.5, unit),"----->", net.num2date(date_2 + 0.5, unit))
    
    return total_temps / total_deltas
    


# In[18]:


season_average(3,t_min, time, unit)


# ### winter min

# In[158]:


unit = netd_tmin["time"].units
total_deltas = 0
total_temps = np.zeros((qc_lat.shape[0], qc_lat.shape[1]))
for elements in range(34):
    date_1 = int(net.date2num(datetime(1980 + elements, 12, 21, hour = 12), unit)-0.5)
    date_2 = int(net.date2num(datetime(1981 + elements, 3, 20, hour = 12), unit)-0.5)
    sum_current = np.sum(t_min[date_1:date_2,r_min_lat:r_max_lat, r_min_lon:r_max_lon], axis = 0)
    delta = date_2 - date_1
    total_deltas += delta
    total_temps += sum_current

else: print(elements, net.num2date(date_1 + 0.5, unit),"----->", net.num2date(date_2 + 0.5, unit))


# In[157]:


average_winter_all = total_temps/total_deltas
tmax = map.contourf(x,y,average_winter_all)
cb = map.colorbar(tmax, "bottom", size = "5%", pad = "2%")
map.drawcoastlines()
map.drawstates()
map.drawparallels(parallels,labels=[1,0,1,0],fontsize=10)
map.drawmeridians(meridians,labels=[0,1,0,1],fontsize=10)


# ### spring min

# In[ ]:


## no


# In[178]:


winter_average = season_average(0,t_min[date_1:date_2,r_min_lat:r_max_lat, r_min_lon:r_max_lon], time, unit)
spring_average = season_average(1,t_min[date_1:date_2,r_min_lat:r_max_lat, r_min_lon:r_max_lon], time, unit)
summer_average = season_average(2,t_min[date_1:date_2,r_min_lat:r_max_lat, r_min_lon:r_max_lon], time, unit)
autumn_average = season_average(3,t_min[date_1:date_2,r_min_lat:r_max_lat, r_min_lon:r_max_lon], time, unit)


# In[179]:


tmax = map.contourf(x,y,winter_average)
cb = map.colorbar(tmax, "bottom", size = "5%", pad = "2%")
map.drawcoastlines()
map.drawstates()
map.drawparallels(parallels,labels=[1,0,1,0],fontsize=10)
map.drawmeridians(meridians,labels=[0,1,0,1],fontsize=10)


# In[181]:


tmax = map.contourf(x,y,spring_average)
cb = map.colorbar(tmax, "bottom", size = "5%", pad = "2%")
map.drawcoastlines()
map.drawstates()
map.drawparallels(parallels,labels=[1,0,1,0],fontsize=10)
map.drawmeridians(meridians,labels=[0,1,0,1],fontsize=10)


# In[182]:


tmax = map.contourf(x,y,autumn_average)
cb = map.colorbar(tmax, "bottom", size = "5%", pad = "2%")
map.drawcoastlines()
map.drawstates()
map.drawparallels(parallels,labels=[1,0,1,0],fontsize=10)
map.drawmeridians(meridians,labels=[0,1,0,1],fontsize=10)


# In[183]:


tmax = map.contourf(x,y,summer_average)
cb = map.colorbar(tmax, "bottom", size = "5%", pad = "2%")
map.drawcoastlines()
map.drawstates()
map.drawparallels(parallels,labels=[1,0,1,0],fontsize=10)
map.drawmeridians(meridians,labels=[0,1,0,1],fontsize=10)


# In[20]:


winter_average_tmax = season_average(0,t_max, time, unit)
spring_average_tmax = season_average(1,t_max, time, unit)
summer_average_tmax = season_average(2,t_max, time, unit)
autumn_average_tmax = season_average(3,t_max, time, unit)


# In[21]:


def mapper(x,y, data, title_name, label = "temperature (*C)"):
    tmax = map.contourf(x,y,data)
    cb = map.colorbar(tmax, "right", size = "5%", pad = "2%")
    cb.set_label(label)
    plt.title(title_name)
    map.drawcoastlines()
    map.drawstates()
    map.drawparallels(parallels,labels=[1,0,0,0],fontsize=10)
    map.drawmeridians(meridians,labels=[0,1,0,0],fontsize=10)
    
    
    plt.savefig(f"/Users/billxue/Documents/neural_network/maps/{'_'.join(title_name.split(' '))}.png")
    plt.show()


# In[240]:


mapper(x, y,winter_average_tmax, "Winter Averages Tmax Qc")


# In[241]:


mapper(x, y,summer_average_tmax, "Summer Averages Tmax Qc")


# In[242]:


mapper(x, y,autumn_average_tmax, "Autumn Averages Tmax Qc")


# In[22]:


mapper(x, y,winter_average_tmax, "Winter Averages Tmax Qc")
mapper(x, y,spring_average_tmax, "Spring Averages Tmax Qc")
mapper(x, y,summer_average_tmax, "Summer Averages Tmax Qc")
mapper(x, y,autumn_average_tmax, "Autumn Averages Tmax Qc")


# In[24]:


winter_average_tmin = season_average(0,t_min, time, unit)
spring_average_tmin = season_average(1,t_min, time, unit)
summer_average_tmin = season_average(2,t_min, time, unit)
autumn_average_tmin = season_average(3,t_min, time, unit)


# In[26]:


mapper(x, y,winter_average_tmin, "Winter Averages Tmin Qc")
mapper(x, y,spring_average_tmin, "Spring Averages Tmin Qc")
mapper(x, y,summer_average_tmin, "Summer Averages Tmin Qc")
mapper(x, y,autumn_average_tmin, "Autumn Averages Tmin Qc")


# In[41]:


pre = t_min[1:3,200:203, 190:193]


# In[47]:


final = np.min(pre, axis = 0)


# In[48]:


pre.shape


# In[49]:


final.shape


# In[50]:


pre


# In[51]:


final


# In[64]:


def season_max(season_index, dataset, time, unit):
    ## solstice 
    seasons = [(12,21,12), (3,21,12), (6,21,12), (9,21,12), (12,21,12)]
    season = seasons[season_index]
    season_next = seasons[season_index + 1]
    
    ## figuring out time 
    start_date = net.num2date(time[0], unit)
    end_date = net.num2date(time[-1], unit)
    
    years = end_date.year - (start_date.year + 1)
    
    total_deltas = 0
    total_temps = np.zeros((x.shape[1], y.shape[1]))
    
    if season_index == 0:
        for elements in range(years):
            date_1 = int(net.date2num(datetime(start_date.year + elements, season[0], season[1], hour = season[2]), unit)-0.5)
            date_2 = int(net.date2num(datetime(start_date.year + 1 + elements, season_next[0], season_next[1], hour = season_next[2]), unit)-0.5)
            sum_current = np.max(dataset[date_1:date_2,r_min_lat:r_max_lat, r_min_lon:r_max_lon], axis = 0)
            delta = date_2 - date_1
            total_deltas += delta
            total_temps += sum_current  
            print(elements, net.num2date(date_1 + 0.5, unit),"----->", net.num2date(date_2 + 0.5, unit))
            
    else:
        for elements in range(years):
            date_1 = int(net.date2num(datetime(start_date.year + 1 + elements, season[0], season[1], hour = season[2]), unit)-0.5)
            date_2 = int(net.date2num(datetime(start_date.year + 1 + elements, season_next[0], season_next[1], hour = season_next[2]), unit)-0.5)
            sum_current = np.max(dataset[date_1:date_2,r_min_lat:r_max_lat, r_min_lon:r_max_lon], axis = 0)
            delta = date_2 - date_1
            total_deltas += delta
            total_temps += sum_current  
            print(elements, net.num2date(date_1 + 0.5, unit),"----->", net.num2date(date_2 + 0.5, unit))
    
    return total_temps / years
    


# In[65]:


def season_min(season_index, dataset, time, unit):
    ## solstice 
    seasons = [(12,21,12), (3,21,12), (6,21,12), (9,21,12), (12,21,12)]
    season = seasons[season_index]
    season_next = seasons[season_index + 1]
    
    ## figuring out time 
    start_date = net.num2date(time[0], unit)
    end_date = net.num2date(time[-1], unit)
    
    years = end_date.year - (start_date.year + 1)
    
    total_deltas = 0
    total_temps = np.zeros((x.shape[1], y.shape[1]))
    
    if season_index == 0:
        for elements in range(years):
            date_1 = int(net.date2num(datetime(start_date.year + elements, season[0], season[1], hour = season[2]), unit)-0.5)
            date_2 = int(net.date2num(datetime(start_date.year + 1 + elements, season_next[0], season_next[1], hour = season_next[2]), unit)-0.5)
            sum_current = np.min(dataset[date_1:date_2,r_min_lat:r_max_lat, r_min_lon:r_max_lon], axis = 0)
            delta = date_2 - date_1
            total_deltas += delta
            total_temps += sum_current  
            print(elements, net.num2date(date_1 + 0.5, unit),"----->", net.num2date(date_2 + 0.5, unit))
            
    else:
        for elements in range(years):
            date_1 = int(net.date2num(datetime(start_date.year + 1 + elements, season[0], season[1], hour = season[2]), unit)-0.5)
            date_2 = int(net.date2num(datetime(start_date.year + 1 + elements, season_next[0], season_next[1], hour = season_next[2]), unit)-0.5)
            sum_current = np.min(dataset[date_1:date_2,r_min_lat:r_max_lat, r_min_lon:r_max_lon], axis = 0)
            delta = date_2 - date_1
            total_deltas += delta
            total_temps += sum_current  
            print(elements, net.num2date(date_1 + 0.5, unit),"----->", net.num2date(date_2 + 0.5, unit))
    
    return total_temps / years


# In[66]:


winter_min_tmin = season_min(0,t_min, time, unit)
spring_min_tmin = season_min(1,t_min, time, unit)
summer_min_tmin = season_min(2,t_min, time, unit)
autumn_min_tmin = season_min(3,t_min, time, unit)


# In[68]:


mapper(x, y,winter_min_tmin, "Winter min Tmin Qc")
mapper(x, y,spring_min_tmin, "Spring min Tmin Qc")
mapper(x, y,summer_min_tmin, "Summer min Tmin Qc")
mapper(x, y,autumn_min_tmin, "Autumn min Tmin Qc")


# In[69]:


winter_max_tmax = season_max(0,t_max, time, unit)
spring_max_tmax = season_max(1,t_max, time, unit)
summer_max_tmax = season_max(2,t_max, time, unit)
autumn_max_tmax = season_max(3,t_max, time, unit)


# In[70]:


mapper(x, y,winter_max_tmax, "Winter Max Tmax Qc")
mapper(x, y,spring_max_tmax, "Spring Max Tmax Qc")
mapper(x, y,summer_max_tmax, "Summer Max Tmax Qc")
mapper(x, y,autumn_max_tmax, "Autumn Max Tmax Qc")


# In[ ]:




