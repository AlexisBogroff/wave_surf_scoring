import cv2
import time
import numpy as np
import matplotlib.pyplot as plt
from sanstitre0 import * 
#from class_video import *


net = cv2.dnn.readNetFromDarknet('yolov4.cfg', 'yolov4.weights')
#%%

surf=image_vague("wave2.png")
surf.Taille_surfeur(net)
surf.detection_haut_de_vague()
surf.affichage_image("image", surf.image_segmented)
surf.affichage_image("image", surf.image_with_lines)

#%%
Vague=[]
i=0
for i in range(8):
    
    
    
    Vague.append(image_vague("wave"+str(i)+".png"))
    start_time = time.time()
    Vague[i].Taille_surfeur(net)
    Vague[i].detection_haut_de_vague()
    end_time = time.time()
    execution_time = end_time - start_time
    print("Temps d'exécution pour wave"+str(i)+f": {execution_time} secondes")

    Vague[i].affichage_image("wave" +str(i), Vague[i].image_with_lines)
    i=i+1

# %%
