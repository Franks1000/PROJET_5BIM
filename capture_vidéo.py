import cv2

# Chemin vers la vidéo
video_path = 'video.mp4'

# Créer un objet VideoCapture pour la vidéo
cap = cv2.VideoCapture(video_path)

# Vérifier si la vidéo est ouverte correctement
if not cap.isOpened():
    print("Erreur lors de l'ouverture de la vidéo.")
    exit()

# Boucle pour lire chaque trame de la vidéo
while True:
    # Lire la trame suivante
    ret, frame = cap.read()

    # Vérifier si la trame a été lue correctement
    if not ret:
        break  # Sortir de la boucle si la trame n'est pas lue correctement

    # Afficher la trame
    cv2.imshow('Frame', frame)

    # Attendre 25 millisecondes et vérifier si la touche 'q' est pressée pour quitter
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

# Libérer la capture vidéo et fermer la fenêtre
cap.release()
cv2.destroyAllWindows()
