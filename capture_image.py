import cv2
import os
import face_recognition

# Chemin vers la vidéo
video_path = 'video.mp4'

# Créer un dossier pour enregistrer les images
output_folder = 'photos'
os.makedirs(output_folder, exist_ok=True)

# Créer un objet VideoCapture pour la vidéo
cap = cv2.VideoCapture(video_path)

# Vérifier si la vidéo est ouverte correctement
if not cap.isOpened():
    print("Erreur lors de l'ouverture de la vidéo.")
    exit()

# Boucle pour lire chaque trame de la vidéo
while True:
    # Capturez le cadre par cadre
    ret, frame = cap.read()
    if not ret:
        break
    
    # Convertir le cadre en RGB (car face_recognition fonctionne avec des images RGB)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Trouver tous les visages dans le cadre
    face_locations = face_recognition.face_locations(rgb_frame)
    
    # Si des visages sont détectés, enregistrez les images
    if face_locations:
        for i, face_location in enumerate(face_locations):
            # Décomposez la boîte englobante en coins (haut, droite, bas, gauche)
            top, right, bottom, left = face_location
            
            # Région d'intérêt (ROI) : extrayez le visage
            face_roi = frame[top:bottom, left:right]

            # Enregistrez l'image du visage dans le dossier de photos
            img_name = f'photo_{i}.jpg'
            img_path = os.path.join(output_folder, img_name)
            cv2.imwrite(img_path, face_roi)

    # Afficher le cadre résultant
    cv2.imshow('Video', frame)
    
    # Quitter la boucle si 'q' est pressé
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Libérer la capture vidéo et fermer les fenêtres
cap.release()
cv2.destroyAllWindows()
