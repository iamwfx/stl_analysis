import os
import sys
import math as m
import pandas as pd
import csv
# import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy import stats

plt.style.use('ggplot')

from sqlalchemy import *
from sqlalchemy import create_engine

def main():
    engine = create_engine('postgresql://wenfei:xj32MYNt3q4jaH@52.23.144.14:5432/stl')
    # engine = create_engine('postgresql://mike:pbkqa0pfjd@52.23.144.14:5432/stl')

    CD_Grids = pd.read_sql_query('SELECT * FROM cd_residentialgrids_all_feb4',con=engine)

    fishnet_CD = CD_Grids

    ## Create a marker for the sphere of influence of the residential grids
    fishnet_CD['resi_neighbor'] = 0
    fishnet_CD['FID_1']=[x-1 for x in fishnet_CD.gid]
    row_count = 256  ## Number of squares per row
    rings = 2 ## Number of "spheres of influence"
    print len(fishnet_CD)


## Find neighbors and get their FIDs
    for index, row in fishnet_CD[fishnet_CD.residential == 1].iterrows():

        if index ==0: ## Top left corner
            for i in range(0,rings+1,1):
                fishnet_CD.ix[index+i*row_count:index+i*row_count+rings,'resi_neighbor'] =1 ##Row below and right
                fishnet_CD.set_value(index,'neighbor_fid', range(index+i*row_count,index+i*row_count+rings+1,1))
        elif index == row_count-1: ## Top right corner
            for i in range(0,rings,1):
                fishnet_CD.ix[index+i*row_count-rings:index+i,'resi_neighbor'] =1 ##Row below and left
                fishnet_CD.set_value(index,'neighbor_fid', range(index+i*row_count-rings,index+i+1,1))
        elif index ==fishnet_CD.FID_1.max()-row_count+1: ## Lower left corner
             for i in range(0,rings,1):
                fishnet_CD.ix[index-i*row_count:index-i*row_count+rings,'resi_neighbor'] =1 ##Row above and right
                fishnet_CD.set_value(index,'neighbor_fid', range(index-i*row_count,index-i*row_count+rings+1,1))
        elif index == fishnet_CD.FID_1.max(): ## Lower right corner
            for i in range(0,rings,1):
                fishnet_CD.ix[index-i*row_count:index-i*row_count-rings,'resi_neighbor'] =1 ##Row above and left
                fishnet_CD.set_value(index,'neighbor_fid', range(index-i*row_count,index-i*row_count-rings+1,1))
        elif 1<= index <= row_count-1: ##Top row excluding corners
            for i in range(0,rings,1):
                fishnet_CD.ix[index+i*row_count-rings:index-i*row_count+rings,'resi_neighbor'] =1 ##Rows below, left and right
                fishnet_CD.set_value(index,'neighbor_fid', range(index+i*row_count-rings,index-i*row_count+rings+1,1))
        elif fishnet_CD.FID_1.max()-row_count+2<= index <= fishnet_CD.FID_1.max()-1:##Bottom row excluding corners
            for i in range(0,rings,1):
                fishnet_CD.ix[index+i*row_count+rings:index+i*row_count+rings,'resi_neighbor'] =1 ##Rows above, left and right
                fishnet_CD.set_value(index,'neighbor_fid', range(index+i*row_count+rings,index+i*row_count+rings+1,1))
        else: 
            nb = []
            for j in [-1,1]:
                for i in range(0,rings+1,1):
                    
                    fishnet_CD.ix[index+j*i*row_count-rings:index+j*i*row_count+rings,'resi_neighbor']=1
                    fishnet_CD.ix[index+i*row_count-rings:index+i*row_count+rings,'resi_neighbor']=1
                    nb.extend(range(index+j*i*row_count-rings,index+j*i*row_count+rings+1,1))
            nb.remove(index)
            # print ','.join([str(x) for x in nb])
            fishnet_CD.set_value(index,'neighbor_fid',', '.join([str(x) for x in nb])) ## Create a list of all the neighbors
            

    fishnet_CD.to_csv('/Users/wenfeixu/Dropbox (MIT)/MIT/CDDL/STL_WeChatProject/0_data/working_wenfei/ghostcities_residentialGrid_Feb4.csv',encoding="utf-8")
    print "done"

if __name__ == "__main__":
    # execute only if run as a script
    main()
            