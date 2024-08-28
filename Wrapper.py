# -*- coding: utf-8 -*-
"""
Created on Mon Feb 13 13:08:48 2023

@author: Siva Kumar Valluri
"""

from Importer import Importer
import pandas as pd
import numpy as np
import time

################################################################################################################################################################################################################
def Looper(address_set, name_set, delays, exposures, SIMX_frames):
    from Run_3 import Run
    import itertools
    dataset_df = pd.DataFrame()
    for addresses, name  in itertools.zip_longest(address_set, name_set):
        run = Run(addresses, name, delays, exposures, SIMX_frames)
        run.crop_choice = 'n'
        particle_data_df, _, _= run.analyze_emission_growth()
        run.stitch_images()
        dataset_df = pd.concat([dataset_df, particle_data_df], ignore_index=True)  
        print("run "+ str(name)+ " complete")
    return dataset_df


################################################################################################################################################################################################################
def bin_and_save(sample_name,name,delays, exposures,Dataset_df, SIMX_frames, stat_choice):
    import itertools
    n=np.arange(1,25,1)
    particle_diameters = (1.3**n)
    indices_of_bins = [i for i, x in enumerate(particle_diameters < float(Dataset_df['diameter_in_pixel'].max())) if x]
    indices_of_bins.append(indices_of_bins[-1]+1)
    particle_diameters = particle_diameters[indices_of_bins].tolist()
    bin_mid_points = [(a + b) / 2 for a, b in zip(particle_diameters[0:-1], particle_diameters[1:])]
    
    # New dataframe for particle size - emission growth relationship
    Data_df = pd.DataFrame() 
    Neighbor_df = pd.DataFrame()
    Data_df.insert(0, "Size_bin", bin_mid_points, True)
    Neighbor_df.insert(0, "Size_bin", bin_mid_points, True)
    titles = []
    for i in range(0,int(SIMX_frames),1):
        title = str(delays[i])+"-"+str(int(delays[i])+int(exposures[i]))
        titles.append(title)
        Data_df[title]=""
        if i == 0:
            Neighbor_df[title]=""    
    
    if stat_choice == '0'  : # Particle area covered by emission
        print('Binning percentage particle area covered by emission based on particle size')   
        for j in range(0,len(particle_diameters)-1,1):
            indices = [i for i in range(0,len(Dataset_df['diameter_in_pixel']),1) if ((Dataset_df['diameter_in_pixel'].iloc[i] >= particle_diameters[j]) and (Dataset_df['diameter_in_pixel'].iloc[i] < particle_diameters[j+1]))]
            for k in range(0,len(indices),1):
                for iterant, title in itertools.zip_longest(np.arange(16,16+int(SIMX_frames),1), titles):
                    A1=pd.Series(Dataset_df[np.array(Dataset_df.columns, dtype='str')[iterant]].iloc[indices[k]])
                    A2=pd.Series(Data_df.at[j,title])
                    A12=pd.concat([A1, A2], ignore_index=True)
                    Data_df.at[j,title]=A12
                    
                    if iterant == 16: 
                        A11=pd.Series(Dataset_df[np.array(Dataset_df.columns, dtype='str')[14]].iloc[indices[k]])
                        A22=pd.Series(Neighbor_df.at[j,title])
                        A1122=pd.concat([A11, A22], ignore_index=True)
                        Neighbor_df.at[j,title]=A1122
                    
    elif stat_choice == '1'  : # Pixel in intensities D50
        print('Binning D50 emission values within a particle based on its size')   
        for j in range(0,len(particle_diameters)-1,1):
            indices = [i for i in range(0,len(Dataset_df['diameter_in_pixel']),1) if ((Dataset_df['diameter_in_pixel'].iloc[i] >= particle_diameters[j]) and (Dataset_df['diameter_in_pixel'].iloc[i] < particle_diameters[j+1]))]
            for k in range(0,len(indices),1):
                for iterant, title in itertools.zip_longest(np.arange(16+2*int(SIMX_frames),16+3*int(SIMX_frames),1), titles):
                    A1=pd.Series(Dataset_df[np.array(Dataset_df.columns, dtype='str')[iterant]].iloc[indices[k]])
                    A2=pd.Series(Data_df.at[j,title])
                    A12=pd.concat([A1, A2], ignore_index=True)
                    Data_df.at[j,title]=A12 
                    
                    if iterant == 16+2*int(SIMX_frames): 
                        A11=pd.Series(Dataset_df[np.array(Dataset_df.columns, dtype='str')[14]].iloc[indices[k]])
                        A22=pd.Series(Neighbor_df.at[j,title])
                        A1122=pd.concat([A11, A22], ignore_index=True)
                        Neighbor_df.at[j,title]=A1122
          
    print('Binning complete')   
    print('Writing into excel file...')    
    import xlsxwriter
    with xlsxwriter.Workbook('Emission analysis of '+ sample_name + "-" + name +'.xlsx') as workbook: 
        for title in titles:
            if stat_choice == '0' :
                worksheet = workbook.add_worksheet('emission cover in '+title)
            else:
                worksheet = workbook.add_worksheet('D50 Pixel I in '+title)
            worksheet.write(0,0, 'av no. of neighbors')
            worksheet.write(1,0, 'mean')
            worksheet.write(2,0, 'stdev')
            for i in range(1,len(bin_mid_points)+1,1):    
                try:
                    worksheet.write(0,i, np.mean(np.array(Neighbor_df.at[i-1,titles[0]][:-1])))
                except TypeError:
                    continue
    
            for j in range(1,len(bin_mid_points)+1,1):
                try:
                    worksheet.write(1,j, np.mean(np.array(Data_df[title].iloc[j-1][:-1])))
                except TypeError:
                    continue   
    
            for k in range(1,len(bin_mid_points)+1,1):    
                try:
                    worksheet.write(2,k, np.std(np.array(Data_df[title].iloc[k-1][:-1])))
                except TypeError:
                    continue        
            for l in range(1,len(bin_mid_points)+1,1):    
                worksheet.write(3,l, 'diameter/pixel')
            for m in range(1,len(bin_mid_points)+1,1):    
                worksheet.write(4,m, bin_mid_points[m-1])
            for n in range(0,len(Data_df["Size_bin"]),1):
                DAM = Data_df.at[n,title]
                for o in range(0,len(DAM),1):
                    worksheet.write(o+5,n+1, DAM[o])
                    
    return Data_df
################################################################################################################################################################################################################        
tic = time.perf_counter()

i = Importer()
address_set, name_set, sample_name, delays, exposures, SIMX_frames = i.organize()
Dataset_df = Looper(address_set, name_set, delays, exposures, SIMX_frames).apply(pd.to_numeric)
dispersed_df = Dataset_df[pd.to_numeric(Dataset_df['No of neighbors']) == 0]
contacted_df = Dataset_df[pd.to_numeric(Dataset_df['No of neighbors']) != 0]

"""
while True:              
    stat_choice=input("What do you want the emission statistics for? (particle area coverage : 0 or pixel intensity D50 : 1): ") 
    try:
        if stat_choice in  ["0", "1"]:
            print("Noted")
        else:
           raise Exception("Invalid input! response can only be '0' or '1'")
    except Exception as e:
        print(e)    
    else:
        break

#Data_df = bin_and_save(sample_name,'raw',delays, exposures,Dataset_df,SIMX_frames, stat_choice)
#Data_df2 = bin_and_save(sample_name,'contacted',delays, exposures,contacted_df,SIMX_frames, stat_choice)
#Data_df3 = bin_and_save(sample_name,'dispersed',delays, exposures,dispersed_df,SIMX_frames, stat_choice)
"""
toc = time.perf_counter()
print(f"Emission growth analyzed in {toc - tic:0.4f} seconds")

############################################################################################################################




