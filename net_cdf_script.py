#!/usr/bin/env python
# coding: utf-8

# In[1]:
### librairies necessaires, chercher la documentation au besoin
import netCDF4 as net
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
from datetime import datetime, timedelta


# ## Defining the variables and geo-parameters

# In[3]:


#default values for QUEBEC : change depending on use case
## relative grid index, which translates to the right coordinates
r_min_lat =120
r_max_lat =180
r_min_lon =200
r_max_lon =260
target_data = [r_min_lat, r_max_lat, 
               r_min_lon, r_max_lon]


## the map uses a different coordinate system : this one is alse 
### calibrated to center on Quebec
map_lat_min= 37
map_lat_max = 65
map_lon_min = -85
map_lon_max = -50
target_map = [map_lat_min, map_lat_max, 
              map_lon_min, map_lon_max]




# In[7]:


## the path that I am going to use
path = "/Users/billxue/tmax.eval.ERA-Int.CRCM5-OUR.day.NAM-22.raw.nc"

name_of_data = "tmax"
# In[10]:


## we extract every needed variab le from the netcdf file
## sources : https://unidata.github.io/netcdf4-python/

dataset = net.Dataset(path)
time = dataset["time"][:]
data = dataset[name_of_data]
lat_g = np.ma.getdata(dataset["lat"][:])
lon_g = np.ma.getdata(dataset["lon"][:])
unit = dataset["time"].units
coor_grid_lat = lat_g[target_data[0]:target_data[1], 
                    target_data[2] : target_data[3]]
coor_grid_lon = lon_g[target_data[0]:target_data[1], 
                    target_data[2] : target_data[3]]





# ## Mapping and grid mapping

# Mapping is an important part of handling netcfd files. You have many options to customize the map to your liking with the mapper, and map_grid function. Try out the features to adapt it to your liking. Obviously, there are limits to these functions, and for more complex adaptations you will need to rewrite them
## sources :https://www2.atmos.umd.edu/~cmartin/python/examples/netcdf_example1.html

def mapper(data, title_name, fig_size = (20,6), 
           label = "",coor_grid_lon = coor_grid_lon,
           coor_grid_lat = coor_grid_lat, 
           target_data = target_data,
           target_map = target_map, path = "./", 
           int_lat = 5, int_lon = 10, save = True):
    
    plt.figure(figsize=fig_size)
    parallels = np.arange(target_map[0],target_map[1],int_lat)
    meridians = np.arange(target_map[2],target_map[3],int_lon)

    map = Basemap(projection='merc',
                  llcrnrlon=target_map[2],
                  llcrnrlat=target_map[0],
                  urcrnrlon=target_map[3],
                  urcrnrlat=target_map[1],
                  resolution='i')

    x,y = map(coor_grid_lon ,coor_grid_lat)

    assoc = map.contourf(x,y,data)
    cb = map.colorbar(assoc, "right", size = "5%", pad = "2%")
    cb.set_label(label)
    plt.title(title_name)

    map.drawcoastlines()
    map.drawstates()
    map.drawparallels(parallels,labels=[1,0,0,0],fontsize=10)
    map.drawmeridians(meridians,labels=[0,1,0,0],fontsize=10)

    if save:
        plt.savefig(f"{path}{'_'.join(title_name.split(' '))}.png")
    plt.show()



# Making single maps is interesting, but we can take it to another level of automation by making map grids, or plots with subplots. 
# The map_grid function allows you to do this with a lot of flexibility for the data of your liking. The data that you input should be coherent with the grid_shape argument, or greater (the function will just do the necessary iterations)

def map_grid(data, grid_shape,title_name, label = "",
             coor_grid_lon = coor_grid_lon,
             coor_grid_lat = coor_grid_lat, 
             target_data = target_data,
             target_map = target_map, path = "./", 
             int_lat = 5, int_lon = 10, save = True):
    
    parallels = np.arange(target_map[0],target_map[1],int_lat)
    meridians = np.arange(target_map[2],target_map[3],int_lon)

    fig, axes = plt.subplots(nrows=grid_shape[0], ncols=grid_shape[1], squeeze=False,figsize=(20,10))
   
    i = 0
    for ax in axes.flat:
        map = Basemap(projection='merc',llcrnrlon=target_map[2],llcrnrlat=target_map[0],urcrnrlon=target_map[3],urcrnrlat=target_map[1],resolution='i', ax = ax)
        x,y = map(coor_grid_lon ,coor_grid_lat)


        assoc = map.contourf(x,y,data[i,:,:])

        cb = map.colorbar(assoc, "right", size = "5%", pad = "2%")
        map.drawcoastlines()
        map.drawstates()
        map.drawparallels(parallels,labels=[1,0,0,0],fontsize=10)
        map.drawmeridians(meridians,labels=[0,1,0,0],fontsize=10)
        i+= 1

    if save:
        plt.savefig(f"{path}{'_'.join(title_name.split(' '))}.png")
    plt.show()



# as with the mapper function, you can customize each minigraph to your own liking

# ## Analysis

# The season analysis function was created with flexibility in mind. It returns, by means only of the input data, a coherent NP array with the processed data, with the total years and days so that you may subsequently choose how to transform the resulting data


def operation(dates, data_coor, data):
    start = dates[0]
    end = dates[1]
    data_final = data[start:end, data_coor[0]:data_coor[1], data_coor[2]:data_coor[3]]
    
    return np.max(data_final, axis = 0).reshape(1,60,60)



def season_analysis(season_index, data = data,time = time, unit = unit, 
                    options = operation, target_data = target_data,
                    years = None, mode = "operation"):
    
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
        
    else:
        start_date = years[0]
        end_date = years[1]
        
    ## le nombre de jours dans la saison peut varier d'année en année
    years = end_date.year - (start_date.year + 1)
    total_deltas = 0
    
    ## initiation the right matrix
    data_trans = np.zeros((1,abs(target_data[0]- target_data[1]), 
                           abs(target_data[2] - target_data[3])))
    
    if season_index == 0:
        for elements in range(years):
            date_1 = int(net.date2num(datetime(start_date.year + elements, season[0], 
                                               season[1], hour = season[2]), unit)-0.5)
            date_2 = int(net.date2num(datetime(start_date.year + 1 + elements, 
                                               season_next[0], season_next[1], 
                                               hour = season_next[2]), unit)-0.5)
            
            delta = date_2 - date_1
            total_deltas += delta
            
            if mode == "operation":
                current = options([date_1,date_2],
                                  target_data, data)
            elif mode == "data":
                current = data[date_1 : date_2, 
                                  target_data[0]:target_data[1], 
                                  target_data[2] :target_data[3]]
                
            data_trans = np.append(data_trans, current, axis = 0)
            print(elements, net.num2date(date_1 + 0.5, unit),"----->", net.num2date(date_2 + 0.5, unit))
                
    else:
        for elements in range(years):
            date_1 = int(net.date2num(datetime(start_date.year + 1 + elements, 
                                               season[0], season[1], hour = season[2]), 
                                               unit)-0.5)
            date_2 = int(net.date2num(datetime(start_date.year + 1 + elements, 
                                               season_next[0], season_next[1], 
                                               hour = season_next[2]), unit)-0.5)

            delta = date_2 - date_1
            total_deltas += delta
            
            if mode == "operation":
                current = options([date_1,date_2],target_data, data)
            elif mode == "data":
                current = data[date_1 : date_2, 
                                  target_data[0]:target_data[1], 
                                  target_data[2] :target_data[3]]
                
            data_trans = np.append(data_trans, current, axis = 0)
            print(elements, net.num2date(date_1 + 0.5, unit),"----->", net.num2date(date_2 + 0.5, unit))
    
    ## return this tuple, and use the information as appropriate for the next steps
    return data_trans[1:, :, :], years, total_deltas


def operation2(dates, data_coor, dataset):
    
    ## do not change this 
    start = dates[0]
    end = dates[1]
    data_final = dataset[start:end, data_coor[0]:data_coor[1], data_coor[2]:data_coor[3]]
    
    ## change this
    return np.min(data_final, axis = 0).reshape(1,60,60)


def basic_stats(data, time = time, data_coor = target_data):
    time_index = time_index - 0.5
    
    data = data[int(time_index[0]): int(time_index[-1]), 
                   data_coor[0]:data_coor[1], 
                   data_coor[2]:data_coor[3]]
    
    basic_data = {"max": np.amax(data), 
                  "min": np.amin(data),
                  "q1":np.percentile(data, 25) , 
                  "q2" : np.percentile(data, 50),
                  "q3": np.percentile(data, 75), 
                  "average": np.average(data)}
    
    return basic_data




