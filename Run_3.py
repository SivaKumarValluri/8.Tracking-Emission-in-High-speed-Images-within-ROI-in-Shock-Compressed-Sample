"""
Created on Tue Feb 09 19:38:12 2023

@author: Siva Kumar Valluri
"""

class Run():
    """
    Class Vairables
    """        
    #Emission frame processing
    smoothing_choice = 'y'               # default choice to smoothing emission frame #Smoothing convolutional kernel size
    kernel_size = 20                     # default value for smoothing emission frame  
    
    #Type of f.o.v cropping
    crop_choice = 'n'                    # default is 'n'
    min_area_box_cropping_choice = 'n'   # default is to crop field of view using bounding box not mix area rectangle which rotates the image     
    
    #stitched imageset for evry run
    #scale_percent = 40                   # default for 4 frame
    scale_percent = 15                   # default for 8 frame
    write_choice = 'y'
    write_color = [255,255,255]
    write_size = 0.5
    write_linethickness = 2
    see_image_choice = 'y'
    save_image_choice = 'n'
    
    #Variables for 'particles_sizing'
    #see_particles_identified = 'n'
    #save_particles_identified = 'n'
    color_contour = [0,0,255]
    color_box = [150,105,0]
    line_thickness_contour = 2
    line_thickness_box = 1 
    
    #Binarizing emission images
    threshold = 100   
    
    #Minimum contour length to be identified as a particle
    contour_length = 5
    
    #Emission analysis choice
    emission_analysis_choice = 'y'
    
    #Scale no of pixels that equal one micron
    scale = 1 
    
    #Neighbor search radius
    radius_of_search = 5
   
    def __init__( self, address_set, name, delays, exposures, SIMX_frames):
        self.address_set = address_set 
        self.name = name 
        self.delays = delays
        self.exposures = exposures 
        self.SIMX_frames = SIMX_frames
        
    def import_and_crop(self):
        import cv2
        import numpy as np
        import itertools
        from scipy.spatial import ConvexHull
        
        static_images = []
        emission_images = []
        #Importing static and emission images in run
        for image_number in range(0,int(len(self.address_set)),1):
            if image_number < int(len(self.address_set)/2):
                static_image = cv2.imread(self.address_set[image_number],cv2.IMREAD_GRAYSCALE)
                static_images.append(static_image)
            elif image_number > int(len(self.address_set)/2)-1:
                #Based on the choice the emission frame is smoothed or presented as is
                if self.smoothing_choice in  ["y","yes","yippee ki yay","alright","alrighty"]:
                    emission_fr = cv2.imread(self.address_set[image_number],cv2.IMREAD_GRAYSCALE)
                    kernel = np.ones((self.kernel_size,self.kernel_size),np.float32)/(self.kernel_size**2)        
                    gaus_image = cv2.filter2D(emission_fr,-1,kernel)
                    emission_image = np.zeros(gaus_image.shape).astype(np.uint8)
                    emission_image = cv2.normalize(gaus_image, None, 0, 255, cv2.NORM_MINMAX)
                    emission_images.append(emission_image)
                else:
                    emission_image = cv2.imread(self.address_set[image_number],cv2.IMREAD_GRAYSCALE)
                    emission_images.append(emission_image)
        
        #Cropping field of view if specifically chosen to do so.
        if  self.crop_choice in ["y","yes","yippee ki yay","alright","alrighty"]:
            iterant = np.arange(0,len(static_images))
            for stat_img, emiss_img, i in itertools.zip_longest(static_images,emission_images, iterant):
                
                contours,_ = cv2.findContours(stat_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
                f_o_v = max(contours, key = cv2.contourArea)
                
                f_o_v2 = f_o_v[:,0,:]
                hull=ConvexHull(f_o_v2)
                hull_points = f_o_v2[hull.vertices]
                HULL=[]
                HULL.append(hull_points)
                mask = np.zeros_like(stat_img) #mask with static image dimensions
                mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)
                cv2.drawContours(mask,tuple(HULL),-1,[255,255,255],-1)
                mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
                #flips black and white (so we have white particles) 
                image_useful_to_size = cv2.bitwise_not(stat_img).copy()                            
                #erosion to ensure the 'outline' of the field of view is removed
                kernel = np.ones((3, 3), np.uint8)
                mask = cv2.erode(mask, kernel, iterations=2)
                corrected_stat_image = cv2.bitwise_and(image_useful_to_size,image_useful_to_size,mask = mask)
                corrected_emission_image = cv2.bitwise_and(emiss_img,emiss_img,mask = mask)
                         
                if self.min_area_box_cropping_choice in  ["y","yes","yippee ki yay","alright","alrighty"]:
                    #cropping to min area bounding box fitting f.o.v (this rotates image)
                    rect = cv2.minAreaRect(f_o_v)
                    box = np.int0(cv2.boxPoints(rect))
                    X_size = int(rect[1][0]) #width
                    Y_size = int(rect[1][1]) #height  
                    src_pts = box.astype("float32")
                    dst_pts = np.array([[0, Y_size],
                                        [0, 0],
                                        [X_size, 0],
                                        [X_size, Y_size]], dtype="float32")  
                    M = cv2.getPerspectiveTransform(src_pts, dst_pts) # the perspective transformation matrix
                    #Final cropped processed images
                    cropped_stationary = cv2.warpPerspective(corrected_stat_image, M, (X_size, Y_size))  
                    cropped_emission = cv2.warpPerspective(corrected_emission_image, M, (X_size, Y_size))
                    static_images[i] = cropped_stationary
                    emission_images[i] = cropped_emission
                else:
                    #cropping to simple bounding box (no rotation) - DEFAULT
                    rect = cv2.minAreaRect(f_o_v)
                    box = np.int0(cv2.boxPoints(rect))
                    Xs = [i[0] for i in box]
                    Ys = [i[1] for i in box]    
                    top_left_x = min(Xs)
                    top_left_y = min(Ys)
                    bot_right_x = max(Xs)
                    bot_right_y = max(Ys)    
                    X_size = int(bot_right_x - top_left_x)
                    Y_size = int(bot_right_y - top_left_y)                   
                    #Final cropped processed images
                    cropped_stationary = corrected_stat_image[top_left_y:bot_right_y+1, top_left_x:bot_right_x+1] 
                    cropped_emission = corrected_emission_image[top_left_y:bot_right_y+1, top_left_x:bot_right_x+1]
                    static_images[i] = cropped_stationary
                    emission_images[i] = cropped_emission            
        return static_images, emission_images
    
    def analyze_emission_growth(self):
        import pandas as pd
        import itertools
        import cv2
        import numpy as np
        from scipy.spatial import cKDTree
        import statsmodels.api as sm
        import scipy

  
        static_images = self.__getattribute__('import_and_crop')()[0]
        emission_images = self.__getattribute__('import_and_crop')()[1]
        name = self.name
        SIMX_frames = self.SIMX_frames
        delays = self.delays
        exposures = self.exposures
        
        #Thresholded emission images for emission coverage analysis
        thresholded_emission_images = []
        for emission_image in emission_images:
            _, thresholded_image = cv2.threshold(emission_image,self.threshold,255,cv2.THRESH_BINARY)
            thresholded_emission_images.append(thresholded_image)
        
        #identifies particles and gives contours and descriptors 
        particle_perimeters = []
        all_bounding_boxes = []
        
        
        #Dataframe for Particle information
        Particles_df = pd.DataFrame()
        p_colmns = ['shot detail','particle no','diameter_in_pixel','centroid_x','centroid_y','Area/pixel^2','Perimeter/pixel','circle_center_x','circle_center_y','Radius/pixel','rect_center_x','rect_center_y','Rect_width/pixel','Rect_height/pixel', 'No of neighbors', 'Perimeter coverage, %']
        Particles_df = pd.DataFrame(columns = p_colmns)
        
        #Dataframe for Emission information
        Emission_df = pd.DataFrame()
        Mu_df = pd.DataFrame()
        Sigma_df = pd.DataFrame()
        D25_df = pd.DataFrame()
        D50_df = pd.DataFrame()
        D75_df = pd.DataFrame()
        
        
        iterant = np.arange(0,int(SIMX_frames),1)
        for static_image, emission_image, thresh_image, i in itertools.zip_longest(static_images,emission_images,thresholded_emission_images, iterant):            
            #Only the first static frame will be processed to yield particle descriptors thats why iterant condition of i==0            
            if i == 0:
                #Particle bounding boxes for imaging 
                boundingboxes=[]
                
                #Identifying all particles            
                contours, hierarchy = cv2.findContours(static_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
                
                #Removing very small 'particles'
                contour_new =[]
                cc=list(contours)
                for contournumber in range(0, len(cc),1):
                    M = cv2.moments(cc[contournumber])
                    if (cc[contournumber].shape[0] > self.contour_length and M['m00'] > 0): # Considering particles that are more than 4 pixels and with moment greater than zero (necessary for centroid)
                        contour_new .append(cc[contournumber])       
                contours=tuple(contour_new)  
                
                List_of_particle_perimeters = []
                for contour in contours:
                    particle_perimeter=contour[:,0,:]
                    List_of_particle_perimeters.append(particle_perimeter)

                

                #emission coverage list    
                emissionlist = []
                #cdf fit features
                mu_list = []
                sigma_list = []
                d25_list =[]
                d50_list =[]
                d75_list =[]
                #Getting emission within particle and particle statistics 
                for contour_number in range(0,len(contours),1):
                    cnt = contours[contour_number]
                    #Collecting Particle data 
                    M = cv2.moments(cnt)
                    centroid_x = int(M['m10']/M['m00'])
                    centroid_y = int(M['m01']/M['m00'])
                    area = cv2.contourArea(cnt)
                    perimeter = cv2.arcLength(cnt,True)
                    diameter = 2*(area/3.14)**0.5
                    #Fitting circle
                    (x,y),radius = cv2.minEnclosingCircle(cnt)
                    circle_center_x = int(x)
                    circle_center_y = int(y)
                    radius = int(radius)
                        
                    #Fitting least area rectangle
                    #It returns a Box2D structure which contains following details - ( center (x,y), (width, height), angle of rotation ). But to draw this rectangle, we need 4 corners of the rectangle. It is obtained by the function cv.boxPoints()
                    rect = cv2.minAreaRect(cnt)
                    rect_center_x=rect[0][0]
                    rect_center_y=rect[0][1]
                    rect_width=rect[1][0]
                    rect_height=rect[1][1]
                    box = cv2.boxPoints(rect)
                    box = np.int0(box)
                    
                    #Emission Analysis
                    if  self.emission_analysis_choice in ["y","yes","yippee ki yay","alright","alrighty"]:
                        #Percentage emission coverage of particle area
                        particle_mask = np.zeros_like(static_image)
                        particle_mask = cv2.cvtColor(particle_mask, cv2.COLOR_GRAY2RGB)
                        cv2.drawContours(particle_mask,contours,contour_number,[255,255,255],thickness=cv2.FILLED)
                        particle_mask = cv2.cvtColor(particle_mask, cv2.COLOR_BGR2GRAY)
                        emission_in_particle2 = cv2.bitwise_and(thresh_image,thresh_image, mask = particle_mask)       #Thresholded emission
                        e_contours, _ = cv2.findContours(emission_in_particle2, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
                        emission_a = []
                        for e_contour_number in range(0, len(e_contours),1):
                            e_cnt = e_contours[e_contour_number]
                            e_area = cv2.contourArea(e_cnt)
                            emission_a.append(e_area)
                        emission_area = np.array(emission_a).sum()
                        percentage_area_coverage = (emission_area/area)*100
                        emissionlist.append(percentage_area_coverage)
                        
                        #Collecting cdf of pixel intensities within particle
                        X_size = int(rect[1][0]) #width
                        Y_size = int(rect[1][1]) #height  
                        src_pts = box.astype("float32")
                        dst_pts = np.array([[0, Y_size],
                                            [0, 0],
                                            [X_size, 0],
                                            [X_size, Y_size]], dtype="float32")  
                        M = cv2.getPerspectiveTransform(src_pts, dst_pts) # the perspective transformation matrix
                        #Final cropped processed images
                        cropped_stationary = cv2.warpPerspective(particle_mask, M, (X_size, Y_size))  
                        cropped_emission = cv2.warpPerspective(emission_image, M, (X_size, Y_size)) 
                        stat_frame = cropped_stationary.flatten()
                        emission_frame = cropped_emission.flatten()
                        emission_within = emission_frame[np.where(stat_frame == 255)].reshape(len(emission_frame[np.where(stat_frame == 255)]),1)
                        X = np.arange(-50,300,1)
                        kde = sm.nonparametric.KDEUnivariate(emission_within)
                        kde.fit(bw=5)
                        Y = kde.evaluate(X)
                        Y = (Y/(np.array(Y).sum()))*100 
                        y = (Y[50:306]/(np.array(Y[50:306]).sum()))*100
                        x = X[50:306]
                        cdf = np.cumsum(y)
                        d25 = np.interp(25,cdf,x)
                        d50 = np.interp(50,cdf,x)
                        d75 = np.interp(75,cdf,x)
                        #Fitting observed cdf to a univariate normal curve cdf
                        try:
                            f = lambda x,mu,sigma: scipy.stats.norm(mu,sigma).cdf(x)
                            mu,sigma = scipy.optimize.curve_fit(f,x,cdf)[0]
                        except ValueError:
                            mu = 0
                            sigma = 0
                        except RuntimeError:
                            mu = 0
                            sigma = 0
                        
                        mu_list.append(mu)
                        sigma_list.append(sigma)
                        d25_list.append(d25)
                        d50_list.append(d50)
                        d75_list.append(d75)
                        
  
                    #Seeing if the particle is in touch with other neighboring particles (see Image_Analysis_Module_II for analysis detail)                 
                    particle_perimeter=cnt[:,0,:]
                    current_particle_index = contour_number
                    other_particle_perimeter_indeces = list(np.arange(0,len(List_of_particle_perimeters),1))
                    other_particle_perimeter_indeces.remove(current_particle_index)
                    
                    List_of_contact_points = []
                    List_of_contacting_particle_indices = []   
                    for other_particle_index in other_particle_perimeter_indeces:            
                        kd_tree1 = cKDTree(List_of_particle_perimeters[current_particle_index])
                        kd_tree2 = cKDTree(List_of_particle_perimeters[other_particle_index])
                        number_of_contact_points=kd_tree1.count_neighbors(kd_tree2, r=self.radius_of_search) #Second input term is radius of survey
                        if number_of_contact_points>0:
                            List_of_contacting_particle_indices.append(other_particle_index)
                            List_of_contact_points.append(number_of_contact_points)
                    
                    number_of_contacting_neighbors = int(len(List_of_contacting_particle_indices))
                    percentage_perimeter_in_contact_with_neighbor = (int(len(List_of_contact_points))/int(len(List_of_particle_perimeters[current_particle_index])))*100
                     
                    
                    #Adding Particle data
                    row_values = [name,contour_number, diameter,centroid_x,centroid_y,area,perimeter,circle_center_x,circle_center_y,radius,rect_center_x,rect_center_y,rect_width,rect_height,number_of_contacting_neighbors,percentage_perimeter_in_contact_with_neighbor]
                    Dataset1 = np.column_stack(row_values)    
                    X1 = pd.DataFrame(Dataset1,columns = p_colmns)
                    Particles_df = pd.concat([Particles_df, X1], ignore_index=True)
                    boundingboxes.append(box)
                    
                    
                boundingboxes=tuple(boundingboxes)    
                particle_perimeters.append(contours)
                all_bounding_boxes.append(boundingboxes)
                
                if  self.emission_analysis_choice in ["y","yes","yippee ki yay","alright","alrighty"]:
                    title = str(delays[i])+"-"+str(int(delays[i])+int(exposures[i]))
                    title2 = "mu " + title
                    title3 = "sigma " + title
                    title4 = "D25 "+ title
                    title5 = "D50 "+ title
                    title6 = "D75 "+ title
                    Emission_df[title] = np.array(emissionlist)
                    Mu_df[title2] = np.array(mu_list)
                    Sigma_df[title3] = np.array(sigma_list)
                    D25_df[title4] = np.array(d25_list)
                    D50_df[title5] = np.array(d50_list)
                    D75_df[title6] = np.array(d75_list)
                           
                
            else:
                contours, _ = cv2.findContours(static_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
                contour_new =[]
                boundingboxes=[]
                
                #Removing very small 'particles'
                cc=list(contours)
                for contournumber in range(0, len(cc), 1):
                    M = cv2.moments(cc[contournumber])
                    if (cc[contournumber].shape[0] > self.contour_length and M['m00'] > 0): # Considering particles that are more than 4 pixels and with moment greater than zero (necessary for centroid)
                        contour_new .append(cc[contournumber])
        
                contours=tuple(contour_new)
                
                #emission coverage list    
                emissionlist = []
                #cdf fit features
                mu_list = []
                sigma_list = []
                d25_list =[]
                d50_list =[]
                d75_list =[]
                for contour_number in range(0,len(contours),1):
                    cnt = contours[contour_number]
                    area = cv2.contourArea(cnt)
                    rect = cv2.minAreaRect(cnt)
                    box = cv2.boxPoints(rect)
                    box = np.int0(box)
                    boundingboxes.append(box)

                    
                    #Emission analysis
                    if  self.emission_analysis_choice in ["y","yes","yippee ki yay","alright","alrighty"]:                   
                        #Percentage emission coverage of particle area
                        particle_mask = np.zeros_like(static_image)
                        particle_mask = cv2.cvtColor(particle_mask, cv2.COLOR_GRAY2RGB)
                        cv2.drawContours(particle_mask,contours,contour_number,[255,255,255],thickness=cv2.FILLED)
                        particle_mask = cv2.cvtColor(particle_mask, cv2.COLOR_BGR2GRAY)
                        emission_in_particle2 = cv2.bitwise_and(thresh_image,thresh_image, mask = particle_mask) #Thresholded emission
                        
                        """
                        #seeing emission within all particles in last frame
                        if i == iterant[-1]:
                            M = cv2.moments(cnt)
                            centroid_x = int(M['m10']/M['m00'])
                            centroid_y = int(M['m01']/M['m00'])
                            #Fitting circle
                            _,radius = cv2.minEnclosingCircle(cnt)
                            radius = int(radius)
                            test = cv2.cvtColor(emission_in_particle2, cv2.COLOR_GRAY2RGB)
                            cv2.drawContours(test,contours,-1,(0,0,255),1)
                            cv2.drawContours(test,cnt,-1,(0,255,0),1)
                            font = cv2.FONT_HERSHEY_SIMPLEX #Other options cv2.FONT_HERSHEY_COMPLEX,cv2.FONT_HERSHEY_TRIPLEX etc
                            text = str(contour_number)
                            #To center the text we find its length and then use it to give position value
                            textX = int(centroid_x+radius+1)
                            textY = int(centroid_y+radius+1) 
                            test = cv2.putText(test, text, (textX, textY ), font, int(self.write_size/2), [0,255,0], int(self.write_linethickness/2), lineType=cv2.LINE_AA)
                            cv2.imshow('Particles identified', test)
                            cv2.waitKey(0)
                        """
                        e_contours, _ = cv2.findContours(emission_in_particle2, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
                        emission_a = []
                        for e_contour_number in range(0, len(e_contours),1):
                            e_cnt = e_contours[e_contour_number]
                            e_area = cv2.contourArea(e_cnt)
                            emission_a.append(e_area)
                        emission_area = np.array(emission_a).sum()
                        percentage_area_coverage = (emission_area/area)*100
                        emissionlist.append(percentage_area_coverage)
                        
                        #Collecting cdf of pixel intensities within particle
                        X_size = int(rect[1][0]) #width
                        Y_size = int(rect[1][1]) #height  
                        src_pts = box.astype("float32")
                        dst_pts = np.array([[0, Y_size],
                                            [0, 0],
                                            [X_size, 0],
                                            [X_size, Y_size]], dtype="float32")  
                        M = cv2.getPerspectiveTransform(src_pts, dst_pts) # the perspective transformation matrix
                        #Final cropped processed images
                        cropped_stationary = cv2.warpPerspective(particle_mask, M, (X_size, Y_size))  
                        cropped_emission = cv2.warpPerspective(emission_image, M, (X_size, Y_size)) 
                        stat_frame = cropped_stationary.flatten()
                        emission_frame = cropped_emission.flatten()
                        emission_within = emission_frame[np.where(stat_frame == 255)].reshape(len(emission_frame[np.where(stat_frame == 255)]),1)
                        X = np.arange(-50,300,1)
                        kde = sm.nonparametric.KDEUnivariate(emission_within)
                        kde.fit(bw=5)
                        Y = kde.evaluate(X)
                        Y = (Y/(np.array(Y).sum()))*100 
                        y = (Y[50:306]/(np.array(Y[50:306]).sum()))*100
                        x = X[50:306]
                        cdf = np.cumsum(y)
                        d25 = np.interp(25,cdf,x)
                        d50 = np.interp(50,cdf,x)
                        d75 = np.interp(75,cdf,x)
                        #Fitting observed cdf to a univariate normal curve cdf
                        try:
                            f = lambda x,mu,sigma: scipy.stats.norm(mu,sigma).cdf(x)
                            mu,sigma = scipy.optimize.curve_fit(f,x,cdf)[0]
                        except ValueError:
                            mu = 0
                            sigma = 0
                        except RuntimeError:
                            mu = 0
                            sigma = 0
                        
                        mu_list.append(mu)
                        sigma_list.append(sigma)
                        d25_list.append(d25)
                        d50_list.append(d50)
                        d75_list.append(d75)
                    
                boundingboxes=tuple(boundingboxes) 
                particle_perimeters.append(contours)
                all_bounding_boxes.append(boundingboxes)
                
                if  self.emission_analysis_choice in ["y","yes","yippee ki yay","alright","alrighty"]:
                    title = str(delays[i])+"-"+str(int(delays[i])+int(exposures[i]))
                    title2 = "mu "+ title
                    title3 = "sigma "+ title
                    title4 = "D25 "+ title
                    title5 = "D50 "+ title
                    title6 = "D75 "+ title
                    Emission_df[title] = np.array(emissionlist)
                    Mu_df[title2] = np.array(mu_list)
                    Sigma_df[title3] = np.array(sigma_list)
                    D25_df[title4] = np.array(d25_list)
                    D50_df[title5] = np.array(d50_list)
                    D75_df[title6] = np.array(d75_list)
     
        if  self.emission_analysis_choice in ["y","yes","yippee ki yay","alright","alrighty"]:
            for column_name in Emission_df.columns:
                Particles_df[column_name] = Emission_df[column_name]
            for column_name4 in D25_df.columns:
                Particles_df[column_name4] = D25_df[column_name4]
            for column_name5 in D50_df.columns:
                Particles_df[column_name5] = D50_df[column_name5]
            for column_name6 in D75_df.columns:
                Particles_df[column_name6] = D75_df[column_name6]
            for column_name2 in Mu_df.columns:
                Particles_df[column_name2] = Mu_df[column_name2]
            for column_name3 in Sigma_df.columns:
                Particles_df[column_name3] = Sigma_df[column_name3]               


                
        return Particles_df, particle_perimeters, all_bounding_boxes      

    def stitch_images(self):
        import cv2
        import numpy as np 
        import itertools
           
        #Gives a stitched image showing all static frames and corresponding emission frames
        static_images = self.__getattribute__('import_and_crop')()[0]
        emission_images = self.__getattribute__('import_and_crop')()[1]
        particle_perimeters = self.__getattribute__('analyze_emission_growth')()[1]
        all_bounding_boxes = self.__getattribute__('analyze_emission_growth')()[2]
        delays = self.delays
        exposures = self.exposures
        imagename =  "run " + self.name
        
        resized_images = []
        for static_image, contours, delay, exposure  in itertools.zip_longest(static_images, particle_perimeters, delays, exposures):
            image =  cv2.cvtColor(static_image, cv2.COLOR_GRAY2RGB)
            cv2.drawContours(image=image, contours=contours, contourIdx=-1, color=[0,0,255], thickness=1, lineType=cv2.LINE_AA)
            wscale = int(image.shape[1] * self.scale_percent / 100)
            hscale = int(image.shape[0] * self.scale_percent / 100)    
            resized_image = cv2.resize(image, (wscale, hscale), interpolation = cv2.cv2.INTER_LINEAR)
            if self.write_choice in  ["y","yes","yippee ki yay","alright","alrighty"]:
                h = resized_image.shape[0]
                w = resized_image.shape[1]
                font = cv2.FONT_HERSHEY_SIMPLEX #Other options cv2.FONT_HERSHEY_COMPLEX,cv2.FONT_HERSHEY_TRIPLEX etc
                text = str(delay)+"-"+str(int(delay)+int(exposure))+" ns"
                #To center the text we find its length and then use it to give position value
                textsize = cv2.getTextSize(text, font, self.write_size, self.write_linethickness)[0]
                textX = int((resized_image.shape[1] - textsize[0])*0.5)
                textY = int((resized_image.shape[0] + textsize[1])*0.1) 
                resized_image = cv2.putText(resized_image, text, (textX, textY ), font, self.write_size, self.write_color, self.write_linethickness)
            resized_images.append(resized_image)
        for static_image, contours, boxes in itertools.zip_longest(emission_images,particle_perimeters, all_bounding_boxes):
            image =  cv2.cvtColor(static_image, cv2.COLOR_GRAY2RGB)
            cv2.drawContours(image=image, contours=contours, contourIdx=-1, color=[0,0,255], thickness=2, lineType=cv2.LINE_AA)
            cv2.drawContours(image=image, contours=boxes, contourIdx=-1, color=[0,255,0], thickness=2, lineType=cv2.LINE_AA)
            wscale = int(image.shape[1] * self.scale_percent / 100)
            hscale = int(image.shape[0] * self.scale_percent / 100)    
            resized_image = cv2.resize(image, (wscale, hscale), interpolation = cv2.cv2.INTER_LINEAR)
            resized_images.append(resized_image)
             
        image = resized_images[0]
        height = (image.shape[0])*2
        width = (image.shape[1])*int(self.SIMX_frames)
        if len(image.shape) == 3:
            stitched = np.zeros((height,width,3), np.uint8)
        else:
            stitched = np.zeros((height,width), np.uint8)
        
        #If images are color
        if len(image.shape) == 3:
            for x in range(1,2*int(self.SIMX_frames)+1,1):
                if x < (int(self.SIMX_frames)+1):
                    h,w,c = resized_images[(x-1)].shape
                    stitched[0:h,(x-1)*w:x*w] = resized_images[(x-1)]
                else:
                    h,w,c = resized_images[(x-1)].shape
                    stitched[h:2*h,(x-int(self.SIMX_frames)-1)*w:(x-int(self.SIMX_frames))*w] = resized_images[(x-1)]
        #If images are grayscale
        else:
            for x in range(1,2*(int(self.SIMX_frames))+1,1):
                if x < (int(self.SIMX_frames)+1):
                    h,w = resized_images[(x-1)].shape
                    stitched[0:h,(x-1)*w:x*w] = resized_images[(x-1)]
                else:
                    h,w = resized_images[(x-1)].shape
                    stitched[h:2*h,(x-int(self.SIMX_frames)-1)*w:(x-int(self.SIMX_frames))*w] = resized_images[(x-1)]
            
      
        if self.see_image_choice in ["y","yes","yippee ki yay","alright","alrighty"]:          
            cv2.imshow(str(imagename), stitched)
            cv2.waitKey(0)
        if self.save_image_choice in ["y","yes","yippee ki yay","alright","alrighty"]:
            cv2.imwrite(str(self.address)+'\\'+ str(imagename)+' .tif', stitched)
