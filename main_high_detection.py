import cv2
import numpy as np
import matplotlib.pyplot as plt



class image_vague:
    def __init__(self,lien,frame=None):
        if frame is not None:
            self.image_original=frame
            self.image_box=frame.copy()
            self.image_ligne=frame.copy()
            self.height=None
            self.width=None
            self.x=None
            self.y=800
            self.lines_above_surfer = None
            self.hauteur_vague = None
            self.image_segmented = np.zeros_like(self.image_original)
            
            
        else :
            self.image_original=cv2.imread(lien)
            self.image_box=cv2.imread(lien)
            self.image_ligne=cv2.imread(lien)
            self.height=None
            self.width=None
            self.x=None
            self.y=None
            self.lines_above_surfer = None
            self.hauteur_vague = None
            self.image_segmented = np.zeros_like(self.image_original)
            
        with open('coco.names', 'r') as f:
            self.classes = f.read().splitlines()
            
       

        
        
    def affichage_image(self,name,image):
        cv2.imshow(name, image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        #print("Taille de la vague (en pixels): ",self.hauteur_vague)

        
    def Taille_surfeur(self, net, conf_threshold=0.5, nms_threshold=0.4):
        model = cv2.dnn_DetectionModel(net)
        model.setInputParams(scale=1 / 255, size=(416, 416), swapRB=True)
        classIds, scores, boxes = model.detect(self.image_box, confThreshold=conf_threshold, nmsThreshold=nms_threshold)
    
        # Identifier l'index de la classe "surfeur"
        surfeur_class_id = self.classes.index('person')  # Remplacez 'surfer' par le nom exact dans votre fichier coco.names
    
        for (classId, score, box) in zip(classIds, scores, boxes):
            if classId == surfeur_class_id and score > conf_threshold:
                x, y, width, height = box
                self.height = height
                self.width = width
                self.x = x
                self.y = y
                
                self.surfeur_haut = y
                self.surfeur_bas = y + height
    
                #cv2.rectangle(self.image_box, (x, y), (x + width, y + height), color=(0, 255, 0), thickness=2)
    
        #self.affichage_image("Image", self.image_box)
            
        
        
        
        #self.ligne_surfeur()

    def ligne_surfeur(self):
        height, width, channels = self.image_original.shape
        cv2.line(self.image_ligne,(0,self.surfeur_haut),(width,self.surfeur_haut),(0,0,255),1)
        cv2.line(self.image_ligne,(0,self.surfeur_bas),(width,self.surfeur_bas),(0,0,255),1)
        self.affichage_image("Ligne",self.image_ligne)
    
    def detecter_contours(self,threshold1=0,threshold2=30):
        gray = cv2.cvtColor(self.image_box, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (7, 7), 0)
        canny = cv2.Canny(blurred, threshold1, threshold2)
 
        contours, _ = cv2.findContours(canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
 
        image_with_contours = np.zeros_like(self.image_original)
        cv2.drawContours(image_with_contours, contours, -1, (0, 255, 0), 2)
        cv2.imshow("Image Segmentee", self.image_segmented)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        self.image_segmented = image_with_contours
            

    def calculate_line_percentage(self, sub_rect):
        return np.count_nonzero(sub_rect) / (sub_rect.shape[0] * sub_rect.shape[1])

    def draw_line_and_text(self, output_img, y):
        cv2.line(output_img, (0, y), (self.image_box.shape[1], y), (0, 255, 0), 5)
        cv2.putText(output_img, str(y), (10, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    def process_detected_line(self, output_img, lines_above_surfer, last_y_center, y):
        lines_above_surfer += 1
        if lines_above_surfer == 1:
            self.hauteur_vague = self.y
            self.draw_line_and_text(output_img, y)
        last_y_center = y
        return output_img, last_y_center, lines_above_surfer

    def detect_lines(self, binary_image, step, stepX, threshold_low, threshold_high):
        lines_above_surfer = 0
        last_y_center = -1
        output_img = self.image_original.copy()

        for y in range(self.y, 100, -step):
            if lines_above_surfer == 4:
            
                break
                
            
            # Update the range for x based on surfer's coordinates
            quarter_width = self.image_original.shape[1] // 4
            x_start = max(0, self.x - quarter_width // 2)
            x_end = min(binary_image.shape[1], self.x + quarter_width // 2)
            x_start = max(x_start, 0)
            x_end = min(x_end, binary_image.shape[1])


            # Modify the line to use x_start and x_end
            #sub_rect = binary_image[int(y - step / 2):int(y + step / 2), x_start:x_end]
            sub_rect = binary_image[int(y - step / 2):int(y + step / 2), 0:binary_image.shape[1]]
            line_percentage = self.calculate_line_percentage(sub_rect)
            threshold = -5 / 1000 * stepX + 0.25

            if line_percentage > threshold:
                if abs(y - last_y_center) > 100:
                    output_img, last_y_center, lines_above_surfer = self.process_detected_line(
                        output_img, lines_above_surfer, last_y_center, y
                    )
        cv2.imshow("Lignes Detectees", output_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return output_img, last_y_center, lines_above_surfer


    def detection_haut_de_vague(self, step=10, threshold_min=0, threshold_max=100):
        for stepX in range(step, 40, 5):
            for threshold_high in range(threshold_min + 10, threshold_max, 10):
                for threshold_low in range(threshold_min, threshold_high, 10):
                    # reinitialisation des variables
                    lines_above_surfer = 0
                    self.hauteur_vague = None
                    last_y_center = -1
                    self.detecter_contours(threshold_low, threshold_high)
                    gray_image = cv2.cvtColor(self.image_segmented, cv2.COLOR_BGR2GRAY)
                    _, binary_image = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY)

                    # début du test après avoir fixé step et threshold
                    output_img, last_y_center, lines_above_surfer = self.detect_lines(
                        binary_image, step, stepX, threshold_low, threshold_high
                    )

                    if lines_above_surfer <= 2 and lines_above_surfer >= 1:
                        break
                if lines_above_surfer <= 2 and lines_above_surfer >= 1:
                    break
            if lines_above_surfer <= 2 and lines_above_surfer >= 1:
                break

        print(last_y_center)
        print(lines_above_surfer)
        self.image_with_lines = output_img  
