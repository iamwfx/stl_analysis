import pandas as pd
import math as m
import numpy as np
import scipy.stats as stats
import time
import csv
import warnings
import datetime

		
##### Matplotlib and Seaborn
import matplotlib.pyplot as plt
from matplotlib import cm
plt.style.use('ggplot')

# import seaborn as sns
# sns.set(color_codes=True)


#####SQL Stuff
from sqlalchemy import *
from sqlalchemy import create_engine
engine = create_engine('postgresql://wenfei:xj32MYNt3q4jaH@52.23.144.14:5432/stl')

##### Our data
## Residential cells
CD_Grids = pd.read_sql_query('SELECT * FROM cd_residentialgrids_selected',con=engine)

## Amenities
amenities_names = ['atm','bank','convenience','ktv','mall','medical','salon','supermarket','restaurant','educational']
amenities_k = [5,2,5,2,2,2,10,5,10,2]
allamenities = [pd.read_sql_query('SELECT gid,avg_distance FROM cd_residentialgrids_nearest_%s_baidu_%s_poi_distance'%(i,j),con=engine) for (i,j) in zip(amenities_k,amenities_names)]


## Landscan data 
landscan_Grids_selected = pd.read_sql_query('SELECT * FROM cd_residentialgrids_landscan_overlap_allgrids',con=engine).sort([
	'gid'])
landscan_Grids_selected=landscan_Grids_selected.sort([
	'gid'])
landscan_Grids = pd.read_sql_query('SELECT * FROM chengdu_landscan_2014_corrected_gcj02',con=engine).sort(['gid'])

## Weibo data
weibo_selected = pd.read_sql_query('SELECT * FROM cd_residentialgrids_weibocount', con=engine).sort(['gid'])
weibo_selected.rename(columns={'count':'weibo_count'}, inplace=True)


## Date
datetimenow= datetime.datetime.now().strftime ("%Y%m%d_%I%M%p")

## Relative density
# relative_density_tablenames = ['fang_listings','weibo_poi']
# relative_density_k= [5,10,25,100]
# relative_density = [pd.read_sql_query('SELECT * FROM cd_relative_density_%s_k_%s'%zip(i,j),con=engine), for (i,j) in zip (relative_density_tablenames,relative_density_k)]



#-------------------------------------------------------------------------------------------------------

def createCSV(table,table_name):
	## Create a CSV from this info: 
	table.to_csv('%s_%s.csv'%(table_name,datenow))

	## Create the TSV to get types of each column
	datatype = [table[column].dtype.name for column in table.columns]
	print datatype
	with open('%s_%s.tsv'%(table_name,datenow), "w") as tsvfile:
		writer = csv.writer(tsvfile, delimiter=',')
		writer.writerow(datatype)
	# for open ('%s_%s.tsv'%(table_name,datenow),'w'):
	# datatype.to_csv('%s_%s.tsv'%(table_name,datenow))



class analysis():

	def __init__(self):
		self.amenities_k = amenities_k
		self.amenities_names = amenities_names
		self.allamenities = allamenities
		self.landscan_Grids = landscan_Grids
		self.landscan_Grids_selected = landscan_Grids_selected
		self.weibo_selected = weibo_selected
	
	def edit_amenities(self):
		
		allamenities = reduce(lambda left,right: pd.merge(left,right,on='gid'), self.allamenities)
		allamenities.columns =  ['gid']+map(lambda x: 'dist_%s'%x, self.amenities_names)

		allamenities=pd.merge(allamenities, self.landscan_Grids_selected,on='gid')
		# allamenities=pd.merge(allamenities, self.landscan_Grids,on='gid')

		landscan_mean =  np.mean(self.landscan_Grids[self.landscan_Grids.population!=0].population)
		# landscan_mean =  np.mean(self.landscan_Grids_selected[self.landscan_Grids_selected.prop_pop!=0].prop_pop)
		landscan_var = landscan_mean**2
		return allamenities,landscan_mean,landscan_var

	def edit_basepopulation(self):

		# Get the some intial statistics on the landscan and weibo data
		landscan_weibo = pd.merge(self.landscan_Grids_selected,self.weibo_selected,on="gid")
		# print landscan_weibo.columns()
		landscan_weibo['gid'] = landscan_weibo['gid'].astype(float)
		# landscan_weibo['log_prop_pop'] = [np.log(x+1) for x in landscan_weibo['population'].astype(float).values]
		landscan_weibo['log_prop_pop'] = [np.log(x+1) for x in landscan_weibo['prop_pop'].astype(float).values]

		# landscan_weibo = pd.merge(self.landscan_Grids,self.weibo_selected,on="gid")
		# landscan_weibo['gid'] = landscan_weibo['gid'].astype(float)
		# landscan_weibo['log_prop_pop'] = [np.log(x+1) for x in landscan_weibo['population'].astype(float).values]
		landscan_weibo['log_weibo_count'] = [np.log(x+1) for x in (landscan_weibo['weibo_count'].astype(float))]

		weibo_nonzero= landscan_weibo[landscan_weibo['weibo_count'] !=0]

		weibo_mean = np.mean(np.log(weibo_nonzero.weibo_count))
		weibo_var = np.var(np.log(weibo_nonzero.weibo_count))
		landscan_weibo['weibo_std_devs'] = (landscan_weibo['log_weibo_count']-weibo_mean)/float(weibo_var)

		return landscan_weibo
		
	def initialplots_amenities(self):

		## Plot amenities distributions
		for each in self.amenities_names:	
			plt.hist(sorted(self.allamenities['dist_%s'%each]),bins=100)
			plt.ylabel("Count")
			plt.title("Distance Histogram for %s"%each)
			plt.show()		


	def initialplots_landscan(self,landscan_Grids_selected,landscan_mean,landscan_var):

		# Plot the histogram for population distributions for the entire region when the population isn't 0.
		plt.hist(landscan_Grids_selected.prop_pop.values, bins=200)
		# plt.hist(landscan_Grids_selected.population.values, bins=200)

		plt.plot([landscan_mean, landscan_mean], [0, 2000], 'k-', lw=.5)
		for x in xrange(0,16,1):
			plt.plot([landscan_mean+x*m.sqrt(landscan_var), landscan_mean+x*m.sqrt(landscan_var)], [0, 2000], '--', lw=.5)
		plt.ylabel('Count')
		plt.title('Distribution of Landscan Populations in Our Grids')
		plt.savefig('Landscan_dist')
		plt.show()

	def initialplots_landscanweibo(self):

		# Plot the distribution of the weibo data
		plt.hist(np.exp(self.landscan_weibo.log_weibo_count))
		plt.xlabel('Log weibo_count Counts')
		plt.ylabel('Frequency')
		plt.title('Hist of Log Weibo')
		plt.show()

		## Plot the joint distribution of the landscan and weibo values
		plt.hexbin(self.landscan_weibo.log_prop_pop.values,self.landscan_weibo.log_weibo_count.values, bins='log', cmap=plt.cm.YlOrRd_r)
		cb = plt.colorbar()
		cb.set_label('log10(N)')
		plt.xlabel('Log Population Counts')
		plt.ylabel('Log Weibo Counts')
		plt.title('Log-Log Plot of Landscan and Weibo')
		plt.show()




# def weibo_estimators():

# 	## Create a new column that gives us the number of standard deviations we are away from the mean for the weibo data. 
# 	weibo_nonzero = landscan_weibo[landscan_weibo['weibo_count'] !=0]
# 	weibo_nonzero['log_count'] =np.log(weibo_nonzero.weibo_count)

# 	mean = np.mean(weibo_nonzero.weibo_count)
# 	var = np.var(weibo_nonzero.weibo_count)
# 	landscan_weibo['std_devs'] = (landscan_weibo['log_weibo_count']-mean)/float(var)
# 	return  landscan_weibo



	def neighborhood_calculations(self):
		a= analysis()
		[allamenities,landscan_mean,landscan_var] = a.edit_amenities()
		landscan_weibo = a.edit_basepopulation()

		## Now create neighborhoods for grids that fall under each of our variable bins
		allamenities['neighborhood'] = [int(x/(landscan_mean))+1 if x/landscan_mean > 1 else 1 for x in allamenities.prop_pop]

		
		## Reindex and sort by population
		allamenities = allamenities.sort(['prop_pop']).reset_index()

		## Create a lower-bound population threshold
	 	allamenities['pop_thres']= 0
	 # 	cutoff = int(len(allamenities)/6.66) ## Cutting off the bottom 15%
		# allamenities['pop_thres'][:cutoff] = 1

		## Create FIDs from GIDs
		allamenities['FID'] = allamenities['gid']-1



		## Now tweak our neighborhood groupings based on weight from the weibo data:
		allamenities = pd.merge(allamenities,landscan_weibo,how='inner',on='gid')
		
		## Create two new neighborhood categories ###		
		allamenities['new_nei_1'] = allamenities['neighborhood'].values
		allamenities['new_nei_2'] = allamenities['neighborhood'].values

		nei_mean=np.zeros(len(list(set(allamenities.neighborhood)))+1)
		nei_std=np.zeros(len(list(set(allamenities.neighborhood)))+1)

		for i in list(set(allamenities.neighborhood))[:-1]:
			print 'Neighborhood is %s, '%i
			nei_mean[i] = np.mean(allamenities[allamenities.neighborhood == i]['log_weibo_count'])
			nei_std[i] = np.std(allamenities[allamenities.neighborhood == i]['log_weibo_count'])
			nei_mean[i+1] = np.mean(allamenities[allamenities.neighborhood == i+1]['log_weibo_count'])
			nei_std[i+1] = np.std(allamenities[allamenities.neighborhood == i+1]['log_weibo_count'])
			for idx,row in allamenities[allamenities.neighborhood == i].iterrows():
				## Weibo weighting method 1 - bump lower grids based on average of group above ##
				if i ==1:
					# if row['weibo_std_devs'] > np.mean(allamenities[allamenities.neighborhood == i]['weibo_std_devs']) and row['weibo_std_devs']>0:
					# 	allamenities.new_nei_1[idx] = row.neighborhood+1

					## Another way ##
					if (row['log_weibo_count']-nei_mean[i])/nei_std[i]>1:
						allamenities.new_nei_1[idx] = row.neighborhood+1
						print 'method 1 neighborhood', allamenities.new_nei_1[idx]
					else: 
						pass
				else:	
					# if row['weibo_std_devs'] > np.mean(allamenities[allamenities.neighborhood == i-1]['weibo_std_devs']) and row['weibo_std_devs']>0:
					# 	allamenities.new_nei_1[idx] = row.neighborhood+1
					## Another way ##
					if (row['log_weibo_count']-nei_mean[i+1])/nei_std[i+1]>1:
						allamenities.new_nei_1[idx] = row.neighborhood+1
						print 'method 1 neighborhood', allamenities.new_nei_1[idx]

				## Weibo weighting method 2 - weighted bump lower grids based on average of current group ##
				# if row['weibo_std_devs'] > np.mean(allamenities[allamenities.neighborhood == i]['weibo_std_devs']) and row['weibo_std_devs']>0:
				# 	allamenities.new_nei_2[idx] = int(row['weibo_std_devs']/np.mean(allamenities[allamenities.neighborhood == i]['weibo_std_devs']))+row.neighborhood
				
				### New Method ###
				if (row['log_weibo_count']-nei_mean[i])/nei_std[i]>1:
						allamenities.new_nei_2[idx] = row.neighborhood+int((row['log_weibo_count']-nei_mean[i])/nei_std[i])
						print "Method 2 neighborhood ",allamenities.new_nei_2[idx]
				else: 
					pass

	 	## Control for neighborhoods with extremely sparse counts (towards the center of the city )
	 	max_group = 15
		allamenities.loc[allamenities['neighborhood']>=max_group,'neighborhood'] =max_group
		allamenities.loc[allamenities['new_nei_1']>=max_group,'new_nei_1'] =max_group
		allamenities.loc[allamenities['new_nei_2']>=max_group,'new_nei_2'] =max_group

		return allamenities

		
	def threshold_calculations(self,table):
		# print self.allamenities[2].head()
		neighborhood_types = [0,1,2]
		neighborhood_names = ['neighborhood','new_nei_1','new_nei_2']
		for each in self.amenities_names:
			# print each
			for j,k in zip(neighborhood_types,neighborhood_names):
				table['%s_%s'%(j,each)] = 0
				print k

				for i in set(table[k]):
					mylist = table[table[k] ==i]['dist_%s'%each].values
					
					#Calculations
					print "Neighborhood is ",i
					try:
						A= np.mean(mylist)
						print "Mean for %s"%A

						table.ix[table[k] ==i,'%s_%s'%(j,each)] =A
					except RuntimeWarning: 
						break
		return table

	def plot_amenitiesgraphs(self,table):
		a = analysis()
		table = a.threshold_calculations(table)
		for each in self.amenities_names:
			table['%s'%each] = 0
			for i in set(table.neighborhood):
				mylist = table[table.neighborhood ==i]['dist_%s'%each].values

				#### Graphs ####
				fig_size = plt.rcParams["figure.figsize"]
				fig_size[0] = 12
				fig_size[1] = 9
				plt.rcParams["figure.figsize"] = fig_size

				## Plot the distances from lowest to highest ##
				plt.subplot(121)
				plt.scatter(range(0,len(mylist)),sorted(mylist),marker = "x",s=20,facecolors='none',edgecolors='b',alpha = 0.5)
				# plt.plot([0,len(mylist)],[np.mean(mylist),np.mean(mylist)],'--')
				plt.annotate(int(np.mean(mylist)),xy=(len(mylist)/2,np.mean(mylist)),xytext = (len(mylist)/2+5,np.mean(mylist)+5))
				plt.title('Plot of %s in Landscan Category %s, N = %s'%(each,i,len(mylist)),fontsize = 10)
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

				plt.annotate(int(fit_beta*fit_alpha),xy=(fit_alpha*fit_beta,max(rv.pdf(mylist))/2), xytext = (fit_alpha*fit_beta*1.2,1.2*(max(rv.pdf(mylist))/2)))
				plt.title('Distribution of %s in Landscan Category %s'%(each,i), fontsize = 10)
				plt.ylabel('Count')
				##Save Figure
				plt.savefig('%s_%s'%(each,i))
				plt.show()
				plt.close()
			


def main():
	print "running..."
	a = analysis()
	### Set our variables ###
	[allamenities,landscan_mean,landscan_var] = a.edit_amenities()
	landscan_weibo = a.edit_basepopulation()	

	### Plot the landscan population distribution and our breakdowns
	a.initialplots_landscan(landscan_Grids_selected,landscan_mean,landscan_var)

	### Run the neighborhood sorting calcuations ###
	allamenities = a.neighborhood_calculations()

	### Calculate thresholds based on each neighborhood
	amenities = a.threshold_calculations(allamenities)

	### Look at our results
	a.plot_amenitiesgraphs(amenities)

	### Create a CSV from it
	createCSV(amenities,'amenities_NEW2_')
	# createCSV(allamenities,'amenitiestable')
if __name__ == "__main__":
    # execute only if run as a script
    main()
            