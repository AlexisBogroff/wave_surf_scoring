import cv2
import time
import numpy as np
import matplotlib.pyplot as plt
from high_detection import * 
#from class_video import *


net = cv2.dnn.readNetFromDarknet('yolov4.cfg', 'yolov4.weights')
#%%
'''
surf=ImageWave("wave2.png")
surf.detect_surfer_size(net)
surf.detect_wave_height()
surf.show_image("image", surf.image_segmented)
surf.show_image("image", surf.image_with_lines)
'''


#%%
Vague=[]
i=0
for i in range(8):
    
    
    
    Vague.append(ImageWave("wave"+str(i)+".png"))
    start_time = time.time()
    Vague[i].detect_surfer_size(net)
    Vague[i].detect_wave_height()
    end_time = time.time()
    execution_time = end_time - start_time
    print("Temps d'exécution pour wave"+str(i)+f": {execution_time} secondes")

    Vague[i].show_image("wave" +str(i), Vague[i].image_with_lines)
    i=i+1

# %%
