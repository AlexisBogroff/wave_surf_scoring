from class_surf import *
import cv2
import numpy as np
import matplotlib.pyplot as plt
import time


#%%
def show_video(video) :
    while(video.isOpened()):
      # Capture frame-by-frame
      ret, frame = video.read()
      if ret == True:
        # Display the resulting frame
        cv2.imshow('Frame',frame)
     
        # Press Q on keyboard to  exit
        if cv2.waitKey(25) & 0xFF == ord('q'):
          break
     
      # Break the loop
      else: 
        break
    video.release()
    cv2.destroyAllWindows()
    
def reduce_image_via_svd(image, k):
    # Split the image into its RGB components
    channels = cv2.split(image)
    
    # Reconstruct each channel with reduced SVD
    recon_channels = []
    for channel in channels:
        # Compute SVD
        U, S, Vt = np.linalg.svd(channel, full_matrices=False)
        
        # Keep only the top-K components
        U_k = U[:, :k]
        S_k = np.diag(S[:k])
        Vt_k = Vt[:k, :]
        
        # Reconstruct the channel
        recon_channel = np.dot(U_k, np.dot(S_k, Vt_k))
        recon_channels.append(recon_channel)

    # Merge the channels back into an image
    recon_image = cv2.merge(recon_channels).astype(np.uint8)
    
    return recon_image
    

class VideoSurferDetector:
    def __init__(self, video_path):
        self.video_capture = cv2.VideoCapture(video_path)
        self.net = cv2.dnn.readNetFromDarknet('yolov4.cfg', 'yolov4.weights')
        
    def process_video(self):
        try:
            last_process_time = time.time()
            while True:
                ret, frame = self.video_capture.read()
                if not ret:
                    break  # No more frames or error
                current_time = time.time()
                
                
                if current_time - last_process_time >= 0.5:
                    #frame = cv2.resize(frame, (800, 500))
                    #frame=reduce_image_via_svd(frame,70)
                    image_vague_instance = image_vague('.', frame)
                    image_vague_instance.Taille_surfeur(self.net)
                    self.display_frame('Processed Frame', image_vague_instance.image_box)
                    last_process_time = current_time
                else :
                    self.display_frame('Processed Frame', frame)
                
               
               
        except Exception as e:
            print(f"An error occurred: {e}")
        finally:
            self.video_capture.release()
            cv2.destroyAllWindows()       
    
        
    @staticmethod
    def display_frame(window_name, frame):
        cv2.imshow(window_name, frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to quit
            cv2.destroyAllWindows()
            return True
        return False
      
'''
video_path = 'surf.mp4'  
video_surfer_detector = VideoSurferDetector(video_path)
video_surfer_detector.process_video()
'''