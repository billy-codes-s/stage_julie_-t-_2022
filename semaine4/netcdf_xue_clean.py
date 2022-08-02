#!/usr/bin/env python
# coding: utf-8

# # Data extraction

# ### libraries

# In[10]:


### librairies necessaires, chercher la documentation au besoin
import netCDF4 as net
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
from datetime import datetime, timedelta


# ### data and global variables

# In[24]:


## initier la base de donnés et extraire les données jugées pertinents (temps, lat,lon)
netd_precip = net.Dataset("/Users/billxue/Documents/neural_network/prec.eval.ERA-Int.CRCM5-OUR.day.NAM-22.raw.nc")
time = netd_precip["time"][:]
prec = netd_precip["prec"]
lat_g = np.ma.getdata(netd_precip["lat"][:])
lon_g = np.ma.getdata(netd_precip["lon"][:])
unit = netd_precip["time"].units


# # Mapping

# ### global variables for coordinates

# In[15]:


## dans ce cas les donnés sont pour le québec
## definir les coordonnées relatives dans la matrice (indices)
r_min_lat =120
r_max_lat =180
r_min_lon =200
r_max_lon =260
qc_lat = lat_g[r_min_lat:r_max_lat, r_min_lon:r_max_lon]
qc_lon = lon_g[r_min_lat:r_max_lat, r_min_lon:r_max_lon]

## faire attention, pour la carte, il faut donner des coordonnées absolues
map_lat = [37, 65]
map_lon = [-85, -50]

## créer l'objet de carte, définie avec les coordonnés map_lat et map_lon
map = Basemap(projection='merc',llcrnrlon=map_lon[0],llcrnrlat=map_lat[0],urcrnrlon=map_lon[1],urcrnrlat=map_lat[1],resolution='i')

## transformer les données latitudes, longitudes, en coordonnées cartésiennes
x,y = map(qc_lon,qc_lat)


# # Data analysis and useful functions

# In[46]:


## faire une opération sur une saison, sur toute la base de données
def season_analysis(season_index, dataset, time, unit, options, years = None):
    ## definir les dates de solstices et de equinoxes(?)
    seasons = [(12,21,12), (3,21,12), (6,21,12), (9,21,12), (12,21,12)]
    season = seasons[season_index]
    season_next = seasons[season_index + 1]
        
    ## j'extrait rigoureusement le nombre d'années valides
    ## la librairie num2date sort la date en fonction de la base de données
    ## vous pouvez ajuster les dates et les index au besoin  
    if years is None:
        start_date = net.num2date(time[0], unit)
        end_date = net.num2date(time[-1], unit)
    
    ## le nombre de jours dans la saison peut varier d'année en année
    years = end_date.year - (start_date.year + 1)
    total_deltas = 0
    total_temps = np.zeros((qc_lat.shape[0], qc_lat.shape[1]))

    if season_index == 0:
        for elements in range(years):
            date_1 = int(net.date2num(datetime(start_date.year + elements, season[0], season[1], hour = season[2]), unit)-0.5)
            date_2 = int(net.date2num(datetime(start_date.year + 1 + elements, season_next[0], season_next[1], hour = season_next[2]), unit)-0.5)
            sum_current = options([date_1,date_2], dataset)
            delta = date_2 - date_1
            total_deltas += delta
            total_temps += sum_current  
            print(elements, net.num2date(date_1 + 0.5, unit),"----->", net.num2date(date_2 + 0.5, unit))
                
    else:
        for elements in range(years):
            date_1 = int(net.date2num(datetime(start_date.year + 1 + elements, season[0], season[1], hour = season[2]), unit)-0.5)
            date_2 = int(net.date2num(datetime(start_date.year + 1 + elements, season_next[0], season_next[1], hour = season_next[2]), unit)-0.5)
            sum_current = options([date_1,date_2], dataset)
            delta = date_2 - date_1
            total_deltas += delta
            total_temps += sum_current  
            print(elements, net.num2date(date_1 + 0.5, unit),"----->", net.num2date(date_2 + 0.5, unit))
        
    return total_temps / years


# ### Fonction pour faire, et sauvegarder une carte

# In[52]:


def mapper(x,y, data, title_name, label = "temperature (*C)"):
    parallels = np.arange(map_lat[0],map_lat[1],5) # make latitude lines ever 5 degrees from 30N-50N
    meridians = np.arange(map_lon[0],map_lon[1],10) # make longitude lines every 5 degrees from 95W to 70W
    contour_data = map.contourf(x,y,data)
    cb = map.colorbar(contour_data, "right", size = "5%", pad = "2%")
    cb.set_label(label)
    plt.title(title_name)
    
    map.drawcoastlines()
    map.drawstates()
    map.drawparallels(parallels,labels=[1,0,0,0],fontsize=10)
    map.drawmeridians(meridians,labels=[0,1,0,0],fontsize=10)
    
    
    plt.savefig(f"/Users/billxue/Documents/neural_network/maps/{'_'.join(title_name.split(' '))}.png")
    plt.show()


# In[ ]:





# In[44]:


r_min_lat =120
r_max_lat =180
r_min_lon =200
r_max_lon =260

def maxer(dates, dataset):
    start = dates[0]
    end = dates[1]
    data_final = dataset[start:end, r_min_lat:r_max_lat, r_min_lon:r_max_lon]
    
    return np.max(data_final, axis = 0)


# In[48]:


winter = winter_average_tmax = season_analysis(0,prec, time, unit,maxer)


# ## Precipitation basic data

# In[30]:


data_raw_prec = prec[:,r_min_lat: r_max_lat, r_min_lon:r_max_lon]


# ### min, max, quartiles and average

# In[ ]:


max_prec = np.amax(data_raw_prec_adjusted)
min_prec = np.amin(data_raw_prec_adjusted)
print("max:", max_prec)
print("min:", min_prec)
print("25%: ", np.percentile(data_raw_prec_adjusted, 25))
print("50%: ", np.percentile(data_raw_prec_adjusted, 50))
print("75%: ", np.percentile(data_raw_prec_adjusted, 75))
print(np.average(data_raw_prec_adjusted))


# ### Save and show

# In[54]:


mapper(x,y, winter, "winter")


# In[ ]:




