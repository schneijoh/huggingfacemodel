from ultralytics import YOLO
import cv2

# YOLOv8 Modell laden (wird automatisch heruntergeladen)
model = YOLO("yolov8n.pt")

# Bildpfad (HIER anpassen!)
image_path = "bild.jpg"

# Bild laden
image = cv2.imread(image_path)

if image is None:
    print("❌ Bild nicht gefunden. Prüfe den Pfad!")
    exit()

# Objekterkennung durchführen
results = model(image)

# Ergebnis speichern
for r in results:
    annotated_frame = r.plot()  # Bounding Boxes einzeichnen
    cv2.imwrite("output.jpg", annotated_frame)

print("✅ Fertig! Ergebnis wurde als 'output.jpg' gespeichert.")
