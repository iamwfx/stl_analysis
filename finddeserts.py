import pandas as pd
import math as m
import numpy as np
import scipy.stats as stats
import time
import csv
import warnings
import datetime
		
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
amenities_names = ['atm','bank','convenience','ktv','mall','medical','salon','supermarket','restaurant','educational']
amenities_k = [5,2,5,2,2,2,10,5,10,2]
landscan_Grids_selected = pd.read_sql_query('SELECT * FROM cd_residentialgrids_landscan_overlap',con=engine)
landscan_Grids = pd.read_sql_query('SELECT * FROM chengdu_landscan_2014_corrected_gcj02',con=engine)
weibo_poi = pd.read_sql_query('Select * from cd_residentialgrids_weibocount', con=engine)

relatxive_density_tablenames = ['fang_listings','weibo_poi']
relative_density_k= [5,10,25,100]
relative_density = [pd.read_sql_query('SELECT * FROM cd_relative_density_%s_k_%s'%zip(i,j),con=engine), for (i,j) in zip (relative_density_tablenames,relative_density_k)]

def initialplots():
	# print allamenities.head()
	for each in amenities_names:	
		plt.hist(sorted(allamenities['avg_distance_%s'%each]),bins=100)
		plt.ylabel("Count")
		plt.title("Distance Histogram for %s"%each)
		plt.show()
	print landscan_Grids.head()
	

	# Plot the histogram for population distributions for the entire region when the population isn't 0.
	plt.hist(landscan_Grids[landscan_Grids.population!=0].population.values, bins=200)
	plt.plot([landscan_mean, landscan_mean], [0, 2000], 'k-', lw=.5)
	for x in xrange(0,25,2):
		plt.plot([landscan_mean+x*m.sqrt(landscan_var), landscan_mean+x*m.sqrt(landscan_var)], [0, 2000], '--', lw=.5)
	plt.ylabel('Count')
	plt.title('Distribution of Landscan Populations in Our Grids')
	plt.show()

def calculations: 
	## Now create thresholds for grids that fall under each of our variable bins
	allamenities['neighborhood'] = [int(x/(2*landscan_mean))+1 if x/landscan_mean > 1 else int(x/landscan_mean) for x in allamenities.prop_pop]
	
	## Reindex and sort by population
	allamenities = allamenities.sort(['prop_pop']).reset_index()

	## Create a lower-bound population threshold
	# allamenities['pop_thres'] = [1 if for x in allamenities.sort(['prop_pop'])[:0.1*len(allamenities)] ]
 	allamenities['pop_thres']= 0
 	cutoff = int(len(allamenities)/6.66) ## Cutting off the bottom 15%
	allamenities['pop_thres'][:cutoff] = 1

	## Create FIDs from GIDs
	allamenities['FID'] = allamenities['gid']-1
 	## Control for neighborhoods with extremely sparse counts (towards the center of the city )
	allamenities.loc[allamenities['neighborhood']>=9,'neighborhood'] =9 

	for each in amenities_names:
		allamenities['threshold_distance_%s'%each] = 0
		for i in set(allamenities.neighborhood):
			mylist = allamenities[allamenities.neighborhood ==i]['avg_distance_%s'%each].values
			
			#Calculations
			print "Neighborhood is ",i
			try: 
				
				A= np.mean(mylist)
				# df.ix[df.A==0, 'B']
				allamenities.ix[allamenities.neighborhood ==i,'threshold_distance_%s'%each] =A
			except RuntimeWarning: 
				break

def allthegraphs():

	for each in amenities_names:
		allamenities['threshold_distance_%s'%each] = 0
		for i in set(allamenities.neighborhood):
			#### Graphs ####
			fig_size = plt.rcParams["figure.figsize"]
			fig_size[0] = 12
			fig_size[1] = 9
			plt.rcParams["figure.figsize"] = fig_size

			## Plot the distances from lowest to highest ##
			plt.subplot(121)
			plt.scatter(range(0,len(mylist)),sorted(mylist),marker = "x",s=20,facecolors='none',edgecolors='b',alpha = 0.5)
			plt.plot([0,len(mylist)],[np.mean(mylist),np.mean(mylist)],'--')
			plt.annotate(int(np.mean(mylist)),xy=(len(mylist)/2,np.mean(mylist)),xytext = (len(mylist)/2+5,np.mean(mylist)+5))
			plt.title('Plot of %s in Landscan Category %s'%(each,i),fontsize = 10)
			plt.ylabel('Distance')
			

			# Plot the distribution of distances
			plt.subplot(122)
			plt.hist(mylist,bins =80,normed=1,histtype='step',linewidth=1.5)

			## Fit the distances to a gamma distribution by estimating alpha and beta of the distribution
			fit_alpha,fit_loc,fit_beta=stats.gamma.fit(mylist, floc=0)
			rv = stats.gamma(fit_alpha, fit_loc,fit_beta)
			print (fit_alpha,fit_loc,fit_beta)

			## Predicted Gamma Distribution PDF
			plt.scatter(mylist, rv.pdf(mylist),facecolors='none',edgecolors='red',alpha=0.5)
			## Plot E[x]
			plt.plot([fit_alpha*fit_beta,fit_alpha*fit_beta],[0,1.2*max(rv.pdf(mylist))],'--')

			# # # plt.plot([0,len(mylist)],[np.mean(mylist),np.mean(mylist)],'--')
			plt.annotate(int(fit_beta*fit_alpha),xy=(fit_alpha*fit_beta,max(rv.pdf(mylist))/2), xytext = (fit_alpha*fit_beta*1.2,1.2*(max(rv.pdf(mylist))/2)))
			plt.title('Distribution of %s in Landscan Category %s'%(each,i), fontsize = 10)
			plt.ylabel('Count')
			##Save Figure
			plt.savefig('%s_%s'%(each,i))
			plt.show()
			plt.close()
			


def main():
	datenow= datetime.datetime.now().strftime ("%Y%m%d")

	## Create our table of amenities
	allamenities = [pd.read_sql_query('SELECT gid,avg_distance FROM cd_residentialgrids_nearest_%s_baidu_%s_poi_distance'\
		%(i,j),con=engine) for (i,j) in zip(amenities_k,amenities_names)]

	for i,each in enumerate(allamenities):
		allamenities[i].columns = ['gid','avg_distance_%s'%amenities_names[i]]

	allamenities = reduce(lambda left,right: pd.merge(left,right,on='gid'), allamenities)

	allamenities=pd.merge(allamenities, landscan_Grids_selected,on='gid')
	print allamenities.head()

	landscan_mean =  np.mean(landscan_Grids[landscan_Grids.population!=0].population)
	landscan_var = landscan_mean**2
			

	## Create a CSV from this info: 
	allamenities.to_csv('allamenities_%s.csv'%datenow)
if __name__ == "__main__":
    # execute only if run as a script
    main()
            