import cv2
import numpy as np

class ImageWave:
    """
    Classe pour le traitement d'images de vagues et détection de surfeurs.
    """

    def __init__(self, path=None, frame=None):
        """
        Initialise l'objet ImageWave.
        """
        self.image_original = frame if frame is not None else cv2.imread(path)
        self.image_box = self.image_original.copy()
        self.image_ligne = self.image_original.copy()
        self.height = self.width = self.x = self.y = None
        self.lines_above_surfer = self.hauteur_vague = None
        self.image_segmented = np.zeros_like(self.image_original)
        self.classes = self._load_classes()
        self.surfeur_haut = self.surfeur_bas = None
        self.image_with_lines = None
        #self.binary=None

    def _load_classes(self):
        """
        Charge les noms de classes depuis un fichier.
        """
        with open('coco.names', 'r') as file:
            return file.read().splitlines()

    def show_image(self, name, image):
        """
        Affiche une image.
        """
        if image is None or image.size == 0:
            print(f"Aucune image à afficher pour '{name}'.")
            return
        cv2.imshow(name, image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def detect_surfer_size(self, net, conf_threshold=0.5, nms_threshold=0.4):
        """
        Détecte la taille du surfeur dans l'image.
        """
        model = cv2.dnn_DetectionModel(net)
        model.setInputParams(scale=1/255, size=(416, 416), swapRB=True)
        class_ids, scores, boxes = model.detect(self.image_box, confThreshold=conf_threshold, nmsThreshold=nms_threshold)

        surfeur_class_id = self.classes.index('person')

        for (class_id, score, box) in zip(class_ids, scores, boxes):
            if class_id == surfeur_class_id and score > conf_threshold:
                x, y, width, height = box
                self.height = height
                self.width = width
                self.x = x
                self.y = y
                self.surfeur_haut = y
                self.surfeur_bas = y + height
                cv2.rectangle(self.image_box, (x, y), (x + width, y + height), color=(0, 255, 0), thickness=2)
    
        #self.show_image("Image", self.image_box)
            
        
        
        
        #self.ligne_surfeur()

    def draw_surfer_line(self):
        """
        Dessine des lignes horizontales aux positions du surfeur.
        """
        height, width, _ = self.image_box.shape
        cv2.line(self.image_ligne, (0, self.surfeur_haut), (width, self.surfeur_haut), (0, 0, 255), 1)
        cv2.line(self.image_ligne, (0, self.surfeur_bas), (width, self.surfeur_bas), (0, 0, 255), 1)
        #self.show_image("Ligne surfeur", self.image_ligne)

    def detect_contours(self, threshold1=0, threshold2=30):
        """
        Détecte les contours dans l'image.
        """
        gray = cv2.cvtColor(self.image_original, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (7, 7), 0)
        canny = cv2.Canny(blurred, threshold1, threshold2)
        contours, _ = cv2.findContours(canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(self.image_segmented, contours, -1, (0, 255, 0), 2)
        #self.show_image("Contours", self.image_segmented)


    def detect_wave_height(self, step=10, threshold_min=0, threshold_max=100):
        """
        Détecte la hauteur de la vague en analysant l'image.
        """
        for stepX in range(step, 40, 5):
            if self._find_suitable_lines(step,stepX, threshold_min, threshold_max):
                break

        #print("Y center of the last line:", self.last_y_center)
        #print("Number of lines above surfer:", self.lines_above_surfer)

    def reset_detection_variables(self):
        """
        Réinitialise les variables pour la détection.
        """
        self.lines_above_surfer = 0
        self.hauteur_vague = None
        self.last_y_center = -1

    def calculate_line_percentage(self, sub_rect):
        """
        Calcule le pourcentage de ligne détectée dans une région spécifique de l'image.
        """
        return np.count_nonzero(sub_rect) / (sub_rect.shape[0] * sub_rect.shape[1])

    def draw_line_and_text_green(self, output_img, y):
        """
        Dessine une ligne et du texte sur l'image à une hauteur spécifique.
        """
        cv2.line(output_img, (0, y), (output_img.shape[1], y), (0, 255, 0), 5)
        cv2.putText(output_img, str(y), (10, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        
    def draw_line_and_text_red(self, output_img, y):
        """
        Dessine une ligne et du texte sur l'image à une hauteur spécifique.
        """
        cv2.line(output_img, (0, y), (output_img.shape[1], y), (0, 0, 255), 5)
        cv2.putText(output_img, str(y), (10, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    def detect_lines(self, binary_image, step, stepX):
        """
        Détecte des lignes dans l'image binaire.
        """
        if binary_image is None:
            print("Aucune image binaire fournie pour la détection des lignes.")
            return None

        #self.lines_above_surfer = 0
        #last_y_center = -1
        output_img = self.image_original.copy()
        image_width = self.image_original.shape[1]
        
        surfer_center_x = self.x + self.width // 2
        detection_width = image_width // 4  # Largeur du rectangle de détection = 1/4 de la largeur de l'image
        start_x = max(surfer_center_x - detection_width // 2, 0)  # S'assurer que le rectangle ne dépasse pas à gauche
        end_x = min(surfer_center_x + detection_width // 2, image_width)

        for y in range(self.y, 100, -step):
            if self.lines_above_surfer == 4:
                break
            
            #Cas de detction full largeur
            #sub_rect = binary_image[int(y - step / 2):int(y + step / 2), :]
            
            #cas de detection 1/4 largeur
            sub_rect = binary_image[int(y - step / 2):int(y + step / 2), start_x:end_x]
            line_percentage = self.calculate_line_percentage(sub_rect)
            #Cas de detction full largeur
            threshold = -5.5 / 1000 * stepX + 0.5
            
            #Cas de detction 1/4 largeur
            threshold = -5.5 / 1000 * stepX + 0.55

            if line_percentage > threshold and abs(y - self.last_y_center) > 100:
                self.lines_above_surfer += 1
                if self.lines_above_surfer == 1:
                    self.draw_line_and_text_green(output_img, y)
                else : self.draw_line_and_text_red(output_img, y)
                self.last_y_center = y

        #self.show_image("Lignes détectées", output_img)
        return output_img

    def _find_suitable_lines(self, step, stepX, threshold_min, threshold_max):
        """
        Trouve des lignes convenables en ajustant les seuils.
        """
        for threshold_high in range(threshold_min + 10, threshold_max, 10):
            for threshold_low in range(threshold_min, threshold_high, 10):
                self.reset_detection_variables()
                self.detect_contours(threshold_low, threshold_high)
                binary=self.process_detection(stepX)
                self.image_with_lines = self.detect_lines(binary,step,stepX )
                print("Tentative avec threshold_low = {}, threshold_high = {}, stepX = {}".format(threshold_low, threshold_high, stepX))
                print("Nombre de lignes détectées (après detect_lines):", self.lines_above_surfer)
                if  self.lines_above_surfer == 1:
                    print("Nombre de lignes detectees", self.lines_above_surfer)
                    return True
        return False

    def process_detection(self, stepX):
        """
        Prépare l'image pour la détection des lignes.
        """
        gray_image = cv2.cvtColor(self.image_segmented, cv2.COLOR_BGR2GRAY)
        _, binary_image = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY)

        return binary_image
        
        
        
# probleme : detect_wave_height n'est paés appelé dans detect line