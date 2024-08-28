# -*- coding: utf-8 -*-
"""
Created on Tue Jan 17 19:16:15 2023

@author: Siva Kumar Valluri
"""

class Importer():  
    def __init__(self):
        self.address = input("Enter address of folder with SIMX data (just copy paste address): ")
        while True:
            self.SIMX_frames = input("Number of SIMX frames (default 8) : ") or "8" # So that user input is not case-senitive
            try:
                if self.SIMX_frames in  ["4", "8"]:
                    print("Noted")
                else:
                    raise Exception("Invalid input! Answer can only be '4' or '8' only")
            except Exception as e:
                print(e)    
            else:
                break
        while True:
            self.choice1=input("Is the exposure same for all instances captured? (Y/N): ").lower() # So that user input is not case-senitive
            try:
                if self.choice1.lower() in  ["y","yes","yippee ki yay","alright","alrighty"]:
                    print("Consider changing exposure next time based on sampling instances")
                elif self.choice1.lower() in ["n","no","nope"]:
                    print("Get ready to enter the exposures used for each instance")
                else:
                    raise Exception("Invalid input! Answer can only be 'Yes' or 'No'")
            except Exception as e:
                print(e)    
            else:
                break

        while True:
            self.choice2=input("Is the capture contiguous? (Y/N): ").lower() # So that user input is not case-senitive
            try:
                if self.choice2.lower() in  ["y","yes","yippee ki yay","alright","alrighty"]:
                    print("Noted.")
                elif self.choice2.lower() in ["n","no","nope"]:
                    print("Get ready to enter the delays used for each instance sampling")
                else:
                    raise Exception("Invalid input! Answer can only be 'Yes' or 'No'")
            except Exception as e:
                print(e)    
            else:
                break

        self.exposures=[]
        if self.choice1.lower() in  ["n","no","nope"]:
            for instance in range(0,int(self.SIMX_frames),1):           
                exp=int(input("Enter exposure for instance (ns)"+str(instance+1)+": ") or "50") #default value for exposure is 50ns 
                self.exposures.append(exp)
        else:
            exp = int(input("What is the chosen exposure? (in ns) (Only positive integers): "))
            for instance in range(0,int(self.SIMX_frames),1):           
                self.exposures.append(exp)
         
        self.delays=[]
        if self.choice2.lower() in  ["n","no","nope"]:
            for instance in range(0,int(self.SIMX_frames),1):           
                delay=int(input("Enter delay for instance (ns) "+str(instance+1)+": ") or "50") 
                self.delays.append(delay)
        else:
            delay=int(input("Enter when the instance of capture began (ns) :" or "0"))
            self.delays.append(delay)
            for number in self.exposures:
                delay=delay+number
                self.delays.append(delay)
            self.delays.pop(-1)
                       
    def organize(self):
        import os
        image_set_names = []
        tiff_images_addresses = []
        for root, subfolders, filenames in os.walk(self.address):
            for filename in filenames:
                image_set_names.append(root.rpartition('\\')[2])
                tiff_images_addresses.append(root + "/" + filename)
                      
        address_set = []
        name_set = []
        run_set = []
        previous_name = ''
        i = 0
        for image_name in image_set_names :
            current_name = image_name
            if current_name != previous_name and i != 0:
                address_set.append(run_set)
                name_set.append(current_name)
                run_set = []
                run_set.append(tiff_images_addresses[i])
            elif current_name != previous_name and i == 0:
                name_set.append(current_name)
                run_set = []
                run_set.append(tiff_images_addresses[i])
            else:
                run_set.append(tiff_images_addresses[i])
                
            previous_name = current_name
            i += 1
        #Last cycle data added 
        address_set.append(run_set)
        sample_name = self.address.rpartition('\\')[2] #Using primary folder name as sample name
        delays = self.delays
        exposures = self.exposures
        SIMX_frames = int(self.SIMX_frames)
        return address_set, name_set, sample_name, delays, exposures, SIMX_frames 