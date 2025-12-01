# -*- coding: utf-8 -*-


import numpy as np
import math
import csv
import os
import cv2
import mediapipe as mp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd 
from sklearn import tree
from sklearn.tree import export_graphviz
from six import StringIO
from IPython.display import Image  
#import pydotplus
#import graphviz 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from imblearn.over_sampling import RandomOverSampler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization, Bidirectional, Input
from tensorflow.keras import Input
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


fila_rostro = [] #para guardar los datos de el rostro
ang_bocas = [None]*4  # Lista temporal
etiqueta = 'k'

datos = []
secuencia = []
longitud_secuencia = 30  # Por ejemplo, 30 frames = ~1 segundo
letra_mostrada = ""
letra_anterior = ""


# === Función para calcular ángulo entre tres puntos
def calcular_angulo(p1, p2, p3):
    v1 = np.array([p1[0] - p2[0], p1[1] - p2[1]])
    v2 = np.array([p3[0] - p2[0], p3[1] - p2[1]])
    v1_norm = np.linalg.norm(v1)
    v2_norm = np.linalg.norm(v2)
    if v1_norm == 0 or v2_norm == 0:
        return 0
    cos_theta = np.dot(v1, v2) / (v1_norm * v2_norm)
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    return math.degrees(np.arccos(cos_theta))

# === Inicializar soluciones MediaPipe
mp_face_mesh = mp.solutions.face_mesh
mp_hands = mp.solutions.hands
mp_pose = mp.solutions.pose

# === Inicializar archivo CSV
#df = pd.read_csv('angulos_movimiento.csv')

# === Puntos del rostro
puntos_emocionales = [
    70, 107, 336, 300,   # Cejas
    33, 133, 362, 263,   # Ojos
    4, 14, 291, 78       # Boca y otros
]
connections_cejas = [(70, 107), (336, 300), (107,133), (362,336)]
connections_ojos = [(133, 33), (362, 263)]
connections_boca = [(78, 14), (14, 291), (78, 4), (4, 291)]

# === Pose (sin puntos de muñeca)
puntos_pose = [11, 12, 13, 14]
conexiones_pose = [(11, 13), (12, 14), (11, 12)]

# === Conexiones visuales de mano
conexiones_mano = [(4, 8), (8, 12), (12, 16), (16, 20), (0, 4), (0, 20)]


#======LSTM(yahir)=============

# ======================
# 1️⃣ Cargar datos
# ======================
data = pd.read_csv("angulos_movimiento.csv")

# Etiqueta al inicio
y = data.iloc[:, 0]
X = data.iloc[:, 1:]

# Codificar etiquetas
encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)
y_categorical = to_categorical(y_encoded)

# Reshape a (muestras, 30 frames, 26 features)
X = X.values.reshape(len(X), 30, 26)

# Dividir datos
X_train, X_test, y_train, y_test = train_test_split(X, y_categorical, test_size=0.2, stratify=y_encoded)

# ======================
# 2️⃣ Modelo LSTM
# ======================
model = Sequential([
    Input(shape=(30, 26)),
    Bidirectional(LSTM(128, return_sequences=True)),
    BatchNormalization(),
    Dropout(0.3),
    Bidirectional(LSTM(64)),
    BatchNormalization(),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(len(np.unique(y)), activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# Entrenamiento
history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test))

# ---- Predicciones del modelo ----
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true_classes = np.argmax(y_test, axis=1)

# ---- Cálculo de métricas ----
acc = accuracy_score(y_true_classes, y_pred_classes)
prec = precision_score(y_true_classes, y_pred_classes, average='weighted')
rec = recall_score(y_true_classes, y_pred_classes, average='weighted')
f1 = f1_score(y_true_classes, y_pred_classes, average='weighted')

print("📊 Resultados del modelo LSTM:")
print(f"Accuracy:  {acc:.4f}")
print(f"Precision: {prec:.4f}")
print(f"Recall:    {rec:.4f}")
print(f"F1-score:  {f1:.4f}")

# ---- Reporte por clase ----
print("\n📄 Classification Report:")
print(classification_report(y_true_classes, y_pred_classes, target_names=encoder.classes_))

# ---- Matriz de Confusión ----
cm = confusion_matrix(y_true_classes, y_pred_classes)

plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=encoder.classes_, yticklabels=encoder.classes_)
plt.xlabel("Predicción")
plt.ylabel("Etiqueta Real")
plt.title("Matriz de Confusión - LSTM")
plt.show()

# === Captura de video
cap = cv2.VideoCapture(1)

with mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as face_mesh, \
    mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        model_complexity=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as hands, \
    mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        enable_segmentation=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as pose:

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        height, width, _ = frame.shape
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results_face = face_mesh.process(image_rgb)
        results_hands = hands.process(image_rgb)
        results_pose = pose.process(image_rgb)

        coords_pose = {}
        coords_muñeca = {}
        fila_csv = [None] * 16
        ang_ci = ang_cd = ang_hi = ang_hd = None

        # === ROSTRO ===
        coords_rostro = {}
        if results_face.multi_face_landmarks:
            for face_landmarks in results_face.multi_face_landmarks:
                for punto in puntos_emocionales:
                    lm = face_landmarks.landmark[punto]
                    x, y = int(lm.x * width), int(lm.y * height)
                    coords_rostro[punto] = (x, y)
                    cv2.circle(frame, (x, y), 2, (0, 0, 255), 2)

                # Dibuja cejas, ojos y boca
                for a, b in connections_cejas + connections_ojos + connections_boca:
                    if a in coords_rostro and b in coords_rostro:
                        cv2.line(frame, coords_rostro[a], coords_rostro[b], (255, 255, 0), 1)

                # UNE EL EXTREMO DE LA CEJA CON EL EXTREMO DEL OJO
                if 70 in coords_rostro and 33 in coords_rostro:
                    cv2.line(frame, coords_rostro[70], coords_rostro[33], (255, 255, 0), 1)
                if 300 in coords_rostro and 263 in coords_rostro:
                    cv2.line(frame, coords_rostro[300], coords_rostro[263], (255, 255, 0), 1)

                # === ÁNGULO INTERNO DEL PUNTO 336 ===
                # Vecinos: 362 (ojo derecho extremo), 300 (ceja derecha media)
                if all(p in coords_rostro for p in [362, 336, 300]):
                    ang_336 = calcular_angulo(coords_rostro[362], coords_rostro[336], coords_rostro[300])
                    x_336, y_336 = coords_rostro[336]
                    cv2.putText(
                        frame, f"{int(ang_336)}", (x_336 + 5, y_336 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1
                    )
                    
                if all(p in coords_rostro for p in [70, 107, 133]):
                    ang_107 = calcular_angulo(coords_rostro[70], coords_rostro[107], coords_rostro[133])
                    x_107, y_107 = coords_rostro[107]
                    cv2.putText(
                        frame, f"{int(ang_107)}", (x_107 + 5, y_107 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1
                    )
                    
                if all(p in coords_rostro for p in [107, 70, 33]):
                    ang_70 = calcular_angulo(coords_rostro[107], coords_rostro[70], coords_rostro[33])
                    x_70, y_70 = coords_rostro[70]
                    cv2.putText(
                        frame, f"{int(ang_70)}", (x_70 + 5, y_70 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1
                    )
                    
                # === ÁNGULO INTERNO DEL PUNTO 300 ===
                if all(p in coords_rostro for p in [336, 300, 263]):
                    ang_300 = calcular_angulo(coords_rostro[336], coords_rostro[300], coords_rostro[263])
                    x_300, y_300 = coords_rostro[300]
                    cv2.putText(
                        frame, f"{int(ang_300)}", (x_300 + 5, y_300 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 100, 255), 1
                    )
                    
                if all(p in coords_rostro for p in [133, 33, 70]):
                    ang_33 = calcular_angulo(coords_rostro[133], coords_rostro[33], coords_rostro[70])
                    x_33, y_33 = coords_rostro[33]
                    cv2.putText(
                        frame, f"{int(ang_33)}", (x_33-5, y_33+15 ),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1
                    )
                    
                # === ÁNGULO INTERNO DEL PUNTO 133 ===
                if all(p in coords_rostro for p in [107, 133, 33]):
                   ang_133 = calcular_angulo(coords_rostro[107], coords_rostro[133], coords_rostro[33])
                   x_133, y_133 = coords_rostro[133]
                   cv2.putText(
                       frame, f"{int(ang_133)}", (x_133 + 5, y_133 + 15),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1
                    )
                    
                # === ÁNGULO INTERNO DEL PUNTO 362 ===
                if all(p in coords_rostro for p in [263, 362, 336]):
                    ang_362 = calcular_angulo(coords_rostro[263], coords_rostro[362], coords_rostro[336])
                    x_362, y_362 = coords_rostro[362]
                    cv2.putText(
                        frame, f"{int(ang_362)}", (x_362 + 5, y_362 + 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1
                    )
                    
                # === ÁNGULO INTERNO DEL PUNTO 263 ===
                if all(p in coords_rostro for p in [362, 263, 300]):
                    ang_263 = calcular_angulo(coords_rostro[362], coords_rostro[263], coords_rostro[300])
                    x_263, y_263 = coords_rostro[263]
                    cv2.putText(
                        frame, f"{int(ang_263)}", (x_263 + 5, y_263 + 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1
                    )
                # === Yahir ===
                    try:
                        fila_rostro = [
                            round(ang_70, 2), round(ang_107, 2), round(ang_133, 2), round(ang_33, 2),
                            round(ang_336, 2), round(ang_300, 2), round(ang_362, 2), round(ang_263, 2)
                        ]
                    except:
                        fila_rostro = [None]*8
                    #OJOS

                # === ÁNGULOS INTERNOS DE LA BOCA ===
                tripletas_boca = [
                    (78, 14, 291),    # Ángulo en 14 (centro arriba de la boca)
                    (14, 291, 78),    # Ángulo en 291 (comisura derecha)
                    (291, 78, 4),     # Ángulo en 78 (comisura izquierda)
                    (78, 4, 291),     # Ángulo en 4 (centro abajo de la boca)
                ]
                for i, (a, b, c) in enumerate(tripletas_boca):
                    if a in coords_rostro and b in coords_rostro and c in coords_rostro:
                        ang = calcular_angulo(coords_rostro[a], coords_rostro[b], coords_rostro[c])
                        ang_bocas[i] = round(ang, 2)
                        cx, cy = coords_rostro[b]
                        offset_x, offset_y = 10, -10
                        if i == 1: offset_x, offset_y = 20, 10
                        if i == 2: offset_x, offset_y = -30, 10
                        if i == 3: offset_x, offset_y = -15, 25
                        cv2.putText(
                            frame,
                            f"{int(ang)}",
                            (cx + offset_x, cy + offset_y),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            (0, 200, 0),
                            1
                        )
                fila_rostro += [ang_bocas[0], ang_bocas[3]]    

        # === POSE
        if results_pose.pose_landmarks:
            for i in puntos_pose:
                lm = results_pose.pose_landmarks.landmark[i]
                if lm.visibility > 0.5:
                    x, y = int(lm.x * width), int(lm.y * height)
                    coords_pose[i] = (x, y)
                    cv2.circle(frame, (x, y), 5, (255, 255, 0), -1)
            for a, b in conexiones_pose:
                if a in coords_pose and b in coords_pose:
                    cv2.line(frame, coords_pose[a], coords_pose[b], (0, 255, 255), 2)

        # === MANOS
        if results_hands.multi_hand_landmarks and results_hands.multi_handedness:
            for i, hand_landmarks in enumerate(results_hands.multi_hand_landmarks):
                lado = results_hands.multi_handedness[i].classification[0].label
                puntos = [0, 4, 8, 12, 16, 20]
                letras = ['A', 'B', 'C', 'D', 'E', 'F'] if lado == 'Right' else ['G', 'H', 'I', 'J', 'K', 'L']
                offset = 0 if lado == 'Right' else 6
                coords_mano = {}
                for punto in puntos:
                    lm = hand_landmarks.landmark[punto]
                    x, y = int(lm.x * width), int(lm.y * height)
                    coords_mano[punto] = (x, y)
                    if punto == 0:
                        coords_muñeca[lado] = (x, y)
                for a, b in conexiones_mano:
                    if a in coords_mano and b in coords_mano:
                        cv2.line(frame, coords_mano[a], coords_mano[b], (0, 255, 0), 2)
                tripletas = [
                    (0, 4, 8),
                    (4, 8, 12),
                    (8, 12, 16),
                    (12, 16, 20),
                    (16, 20, 0),
                    (20, 0, 4)
                ]
                for j, (a, b, c) in enumerate(tripletas):
                    if a in coords_mano and b in coords_mano and c in coords_mano:
                        ang = calcular_angulo(coords_mano[a], coords_mano[b], coords_mano[c])
                        fila_csv[offset + j] = round(ang, 2)
                        cx, cy = coords_mano[b]
                        cv2.putText(frame, f"{letras[j]}:{int(ang)}", (cx + 5, cy + 15),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

        # === Ángulos y conexiones correctos de codo a muñeca (corregido)
        if 'Left' in coords_muñeca and 14 in coords_pose and 11 in coords_pose:
            cv2.line(frame, coords_muñeca['Left'], coords_pose[14], (255, 255, 0), 2)
            ang_ci = calcular_angulo(coords_muñeca['Left'], coords_pose[14], coords_pose[11])
            cv2.putText(frame, f"EL:{int(ang_ci)}", (coords_pose[14][0] + 5, coords_pose[14][1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)
        if 'Right' in coords_muñeca and 13 in coords_pose and 12 in coords_pose:
            cv2.line(frame, coords_muñeca['Right'], coords_pose[13], (255, 255, 0), 2)
            ang_cd = calcular_angulo(coords_muñeca['Right'], coords_pose[13], coords_pose[12])
            cv2.putText(frame, f"ER:{int(ang_cd)}", (coords_pose[13][0] + 5, coords_pose[13][1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)
        if 13 in coords_pose and 11 in coords_pose and 12 in coords_pose:
            ang_hi = calcular_angulo(coords_pose[13], coords_pose[11], coords_pose[12])
            cv2.putText(frame, f"SR:{int(ang_hi)}", (coords_pose[11][0] + 5, coords_pose[11][1] + 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
        if 14 in coords_pose and 12 in coords_pose and 11 in coords_pose:
            ang_hd = calcular_angulo(coords_pose[14], coords_pose[12], coords_pose[11])
            cv2.putText(frame, f"SL:{int(ang_hd)}", (coords_pose[12][0] + 5, coords_pose[12][1] + 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
        
        if all(isinstance(val, (int, float)) for val in fila_rostro) and \
           all(isinstance(val, (int, float)) for val in fila_csv[:12]) and \
           all(isinstance(val, (int, float)) for val in [ang_ci, ang_cd, ang_hi, ang_hd]):
            
            # Guardar ángulos en la fila
            fila_csv[12:16] = [round(ang_ci, 2), round(ang_cd, 2), round(ang_hi, 2), round(ang_hd, 2)]
            fila_total = fila_rostro + fila_csv
            secuencia.append(fila_total)
        
            # Mantener ventana de 30 frames
            if len(secuencia) > longitud_secuencia:
                secuencia.pop(0)
        
            # 👉 SOLO predecir cuando la secuencia esté completa (30 frames)
            if len(secuencia) == longitud_secuencia:
                # Convertir a arreglo (1, 30, 26)
                entrada = np.array(secuencia).reshape(1, longitud_secuencia, 26)
            
                # Predecir
                prediccion = model.predict(entrada, verbose=0)
                clase_pred = np.argmax(prediccion)
                letra_actual = encoder.inverse_transform([clase_pred])[0]
            
                if letra_actual != letra_anterior:
                    print(f"Letra detectada: {letra_actual}")
                    letra_mostrada = letra_actual
                    letra_anterior = letra_actual
                    
        if letra_mostrada != "":
            # Obtener ancho del texto para colocarlo en la esquina derecha
            font_scale = 1.5
            thickness = 7
            text_size = cv2.getTextSize(letra_mostrada, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0]
            text_x = frame.shape[1] - text_size[0] - 20  # margen de 20 px desde el borde derecho
            text_y = 60  # distancia desde el borde superior
        
            cv2.putText(frame, f"{letra_mostrada}", (text_x, text_y),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 255), thickness)  # rojo, grueso
        
        cv2.imshow("Ángulos Mano y Codo - LSTM", frame)
        if cv2.waitKey(1) == 27:
            break

cap.release()
cv2.destroyAllWindows()


