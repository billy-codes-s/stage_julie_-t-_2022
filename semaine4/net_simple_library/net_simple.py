import netCDF4 as net
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
from datetime import datetime, timedelta

class onev_analysis:
    ## the vision is that you can get lavish outputs just by inserting the path of the netcdf
    def __init__ (self, dataset, time, data_rc, map_rc, lat, lon):

        ### loading the dataset and extracting variables of time, lat and long
        self.dataset = dataset
        self.time = time[:]
        self.unit = time.units

        self.datalat = data_rc[0:2]
        self.datalon = data_rc[2:]

        self.maplat = map_rc[0:2]
        self.maplon = map_rc[2:]

        self.lat = lat[:]
        self.lon = lon[:]
        self.arr_lat = lat[self.datalat[0]:self.datalat[1], self.datalon[0]:self.datalon[1]]
        self.arr_lon = lon[self.datalat[0]:self.datalat[1], self.datalon[0]:self.datalon[1]]

        print(self.datalat)
        print(self.datalon)
        print(dataset.shape)

    def basic_ops(self, dates, time = "dates", operation = "average"):
        dates.sort()
        if time == "dates":
            start = net.date2num(dates[0], self.unit)
            end = net.date2num(dates[1], self.unit)
        else:
            start = dates[0]
            end = dates[1]

        data_final = self.dataset[start:end, self.datalat[0]:self.datalat[1], self.datalon[0]:self.datalon[1]]
        
        if operation == "average":
            return np.average(data_final, axis = 0)
        elif operation == "sum":
            return np.sum(data_final, axis = 0)
        elif operation == "max":
            return np.max(data_final, axis = 0)
        elif operation == "min":
            return np.min(data_final, axis = 0)

    def mapper(self, data, title_name, label, path = "../", int_lat = 5, int_lon = 10, save = True):
        parallels = np.arange(self.maplat[0],self.maplat[1],int_lat)
        meridians = np.arange(self.maplon[0],self.maplon[1],int_lon)

        map = Basemap(projection='merc',llcrnrlon=self.maplon[0],llcrnrlat=self.maplat[0],urcrnrlon=self.maplon[1],urcrnrlat=self.maplat[1],resolution='i')
        x,y = map(self.arr_lon ,self.arr_lat)

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

    def basic_stats(self, data = None):
        if data is None: 
            data = self.dataset[:, self.datalat[0]:self.datalat[1], self.datalon[0]:self.datalon[1]]

        basic_data = {"max": np.amax(data), "min": np.amin(data),"q1":np.percentile(data, 25) , "q2" : np.percentile(data, 50),"q3": np.percentile(data, 75), "average": np.average(data)}
        data = 0
        return basic_data
        
    def season_analysis(self,season_index, mode = "max", years = None):
        ## solstice 
        seasons = [(12,21,12), (3,21,12), (6,21,12), (9,21,12), (12,21,12)]
        season = seasons[season_index]
        season_next = seasons[season_index + 1]
        
        ## figuring out time    
        if years is None:
            start_date = net.num2date(self.time[0], self.unit)
            end_date = net.num2date(self.time[-1], self.unit)
        
        years = end_date.year - (start_date.year + 1)
        total_deltas = 0
        total_temps = np.zeros((self.arr_lat.shape[0], self.arr_lat.shape[1]))

        if season_index == 0:
            for elements in range(years):
                date_1 = int(net.date2num(datetime(start_date.year + elements, season[0], season[1], hour = season[2]), self.unit)-0.5)
                date_2 = int(net.date2num(datetime(start_date.year + 1 + elements, season_next[0], season_next[1], hour = season_next[2]), self.unit)-0.5)
                sum_current = self.basic_ops([date_1,date_2], time = "", operation = mode)
                delta = date_2 - date_1
                total_deltas += delta
                total_temps += sum_current  
                print(elements, net.num2date(date_1 + 0.5, self.unit),"----->", net.num2date(date_2 + 0.5, self.unit))
                
        else:
            for elements in range(years):
                date_1 = int(net.date2num(datetime(start_date.year + 1 + elements, season[0], season[1], hour = season[2]), self.unit)-0.5)
                date_2 = int(net.date2num(datetime(start_date.year + 1 + elements, season_next[0], season_next[1], hour = season_next[2]), self.unit)-0.5)
                sum_current = self.basic_ops([date_1,date_2], time = "", operation = mode)
                delta = date_2 - date_1
                total_deltas += delta
                total_temps += sum_current  
                print(elements, net.num2date(date_1 + 0.5, self.unit),"----->", net.num2date(date_2 + 0.5, self.unit))
        
        return total_temps / years


