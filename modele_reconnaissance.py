import dlib
import glob
import os

# Chemin vers le dossier contenant les images annotées
dataset_folder = 'images'

# Création de l'objet de type dlib pour la détection des visages
detector = dlib.get_frontal_face_detector()

# Création de l'objet de type dlib pour l'extraction des caractéristiques faciales
shape_predictor = dlib.shape_predictor()

# Liste pour stocker les emplacements des visages et les étiquettes
face_locations = []
face_labels = []

# Parcourir les fichiers d'images dans le dossier dataset
for file_path in glob.glob(os.path.join(dataset_folder, '*.jpg')):
    # Charger l'image
    img = dlib.load_rgb_image(file_path)

    # Détecter les visages dans l'image
    dets = detector(img, 1)

    # Pour chaque visage détecté dans l'image
    for det in dets:
        # Extraire les coordonnées du visage
        left, top, right, bottom = det.left(), det.top(), det.right(), det.bottom()

        # Extraire les caractéristiques faciales (68 points)
        shape = shape_predictor(img, det)

        # Stocker les emplacements des visages et les étiquettes
        face_locations.append(shape)
        face_labels.append(os.path.basename(file_path).split('.')[0])  # Utiliser le nom du fichier comme étiquette

# Entraîner le modèle de reconnaissance faciale dlib
recognizer = dlib.face_recognition_model_v1(face_locations, face_labels)

# Enregistrer le modèle entraîné
recognizer.save('modele_reconnaissance_dlib.dat')
