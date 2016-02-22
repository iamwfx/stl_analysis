import pandas as pd
import math as m
import numpy as np
import scipy.stats as stats
import time
import csv

##### Matplotlib
import matplotlib.pyplot as plt
from matplotlib import cm
plt.style.use('ggplot')

#####SQL Stuff
from sqlalchemy import *
from sqlalchemy import create_engine
engine = create_engine('postgresql://wenfei:xj32MYNt3q4jaH@52.23.144.14:5432/stl')


##### Regression stuff
from sklearn import cross_validation
from sklearn.ensemble import RandomForestRegressor
RF= RandomForestRegressor()

##### Our data
CD_Grids = pd.read_sql_query('SELECT * FROM cd_residentialgrids_selected',con=engine)
amenities_names = ['atm','bank','convenience','ktv','mall','medical','salon','supermarket']
amenities_k = [5,2,5,2,2,2,10,5]
landscan_Grids = pd.read_sql_query('SELECT * FROM cd_residentialgrids_landscan_overlap',con=engine)

relative_density_tablenames = ['fang_listings','weibo_poi']
relative_density_k= [5,10,25,100]
relative_density = [pd.read_sql_query('SELECT * FROM cd_relative_density_%s_k_%s'%zip(i,j),con=engine), for (i,j) in zip (relative_density_tablenames,relative_density_k)]


def main():

	allamenities = [pd.read_sql_query('SELECT gid,avg_distance FROM cd_residentialgrids_nearest_%s_baidu_%s_poi_distance'\
		%(i,j),con=engine) for (i,j) in zip(amenities_k,amenities_names)]
	# allamenities = [each'gid',"avg_distance_%s"%i] for (i,each) in zip(amenities_names,allamenities)]
	
	for i,each in enumerate(allamenities):
		allamenities[i].columns = ['gid','avg_distance_%s'%amenities_names[i]]

	allamenities = pd.concat(allamenities,axis=1)

	# print allamenities.head()
	for each in amenities_names:	
		plt.hist(sorted(allamenities['avg_distance_%s'%each]),bins=100)
		plt.ylabel("Count")
		plt.title("Distance Histogram for %s"%each)
		plt.show()







if __name__ == "__main__":
    # execute only if run as a script
    main()
            