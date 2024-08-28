# Shock Microscope Image-Analysis to Track Growth of Emission
## General Description: ##    
-The code in current form is set to automate image processing of the images from the SIMX8 camera used in Laser driven shock compression experiments conducted in Dlott Lab. To associate reactions to spatial regions and track their growth quantitatively.

-The high-speed videography in the Laser driven shock compression experiment enables 4/8 frames of observation during/post shock compression and an additional initial stationary image.

-The program takes segmeneted versions of initial stationary images captured before the experiment (The Regions of interest R.O.I) and tracks emissions growth within these defined R.O.I using frames captrued during experiment.

-Since each frame during experiment is not backlit, the bright pixels observed are 'emission' which are then sequestered into the R.O.I for each frame. 

-Based on the exposure and delay settings used to capture each frame, we can get temporal variation of emission within these R.O.I.

-The program when run will process the images and yield a workbook collecting the output which is either the percentage of R.O.I covered by emission or pixel intensity distributions within R.O.I as a function of R.O.I size. Further the program is able to identify and separate R.O.I in 'contact' and isolated R.O.I.

## What to expect when you're running the code ##
You can use the example data provided to understand how to use the code. 
This dataset shown has been published in 

### Folder structure for input data ###
-Program asks for address of master folder (usually name of project) containing subfolders (labelled based on experimental run they are acquired from in the example dataset provided).

-Each subfolder has 'x' number of segmented/binarized images with foregorund (R.O.I) in white and background in black and same number (x) of emission frames that are untouched/unmodified. 

### Output file ###
-The program when run shows yields a excel workbook as output. (Check for it in the main directory not user supplied data directory) 
_______
**The functions defined can be used for other type of images based on user discretion.** 
 

