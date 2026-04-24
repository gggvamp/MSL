
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 21 23:34:26 2023

@author: 52331
"""

import cv2
import mediapipe as mp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd 
from sklearn import tree
from sklearn.tree import export_graphviz
from six import StringIO
from IPython.display import Image  
import pydotplus
import graphviz 

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)


#df=pd.read_excel(r"C:\Users\Gerardo\\OneDrive\Desktop\investigacion\datosO.xlsx", sheet_name='Hoja1',header=None)
df= pd.read_excel(r"C:\Users\Gerardo\datosO.xlsx", sheet_name='Hoja1', header=None)


with mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5) as hands:
    
    while True:
        ret, frame = cap.read()
        if ret == False:
            break
        height, width, _ = frame.shape
     
        frame = cv2.flip(frame, 1)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)
        if results.multi_hand_landmarks is not None:
               
        
              for hand_landmarks in results.multi_hand_landmarks:
                  x1=int(hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].x * width)
                  y1=int(hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].y * height)
               
                  x2=int(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x * width)
                  y2=int(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y * height)
               
                  x3=int(hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].x * width)
                  y3=int(hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y * height)
                  
                  x4=int(hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP].x * width)
                  y4=int(hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP].y * height)
                  
                  x5=int(hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP].x * width)
                  y5=int(hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP].y * height)
                  
                  x6=int(hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].x * width)
                  y6=int(hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].y * height)
                  
                  angulos=[]
                  arregloangulos=[]
                  
                  # print("THUMB_TIP azul-> ",x1,y1)
                  # print("INDEX_FINGER_TIP verde-> ",x2,y2)
                  # print("MIDDLE_FINGER_TIP rojo->",x3,y3)
                  # print("RING_FINGER_TIP amari->",x4,y4)
                  # print("PINKY_TIP negro->",x5,y5)
                  # print("WRIST mage->",x6,y6)
                  cv2.circle(frame,(x1,y1),3,(255,0,0),4)
                  cv2.circle(frame,(x2,y2),3,(0,255,0),4)
                  cv2.circle(frame,(x3,y3),3,(0,0,255),4)
                  cv2.circle(frame,(x4,y4),3,(0,255,255),4)
                  cv2.circle(frame,(x5,y5),3,(0,0,0),4)
                  cv2.circle(frame,(x6,y6),3,(255,0,255),4)
                  cv2.line(frame,[x1,y1],[x2,y2],(255,255,0),thickness = 2)
                  cv2.line(frame,[x2,y2],[x3,y3],(255,255,0),thickness = 2)
                  cv2.line(frame,[x3,y3],[x4,y4],(255,255,0),thickness = 2)
                  cv2.line(frame,[x4,y4],[x5,y5],(255,255,0),thickness = 2)
                  cv2.line(frame,[x5,y5],[x6,y6],(255,255,0),thickness = 2)
                  cv2.line(frame,[x6,y6],[x1,y1],(255,255,0),thickness = 2)
                  
                  v=[x2,y2]
                  a=[x1,y1]
                  m=[x6,y6]
                  n=[x5,y5]
                  am=[x4,y4]
                  r=[x3,y3]
                  
                  v0=np.array(v)-np.array(a)
                  v1=np.array(m)-np.array(a)
                  
                  v2=np.array(a)-np.array(m)
                  v3=np.array(n)-np.array(m)
                  
                  v4=np.array(m)-np.array(n)
                  v5=np.array(am)-np.array(n)
                  
                  v6=np.array(n)-np.array(am)
                  v7=np.array(r)-np.array(am)
                  
                  v8=np.array(am)-np.array(r)
                  v9=np.array(v)-np.array(r)
                  
                  v10=np.array(r)-np.array(v)
                  v11=np.array(a)-np.array(v)
                  
                
                  
                  angle1=np.math.atan2(np.linalg.det([v0,v1]),np.dot(v0,v1))#rad
                  grado1= np.degrees(angle1)
                  grado1=round(grado1)
                  #print(grado1)
                  cv2.putText(frame,str(grado1),(x1+15,y1),cv2.FONT_HERSHEY_SIMPLEX,0.3,(255,0,0),1)
                  
                  angle2=np.math.atan2(np.linalg.det([v2,v3]),np.dot(v2,v3))#rad
                  grado2= np.degrees(angle2)
                  grado2=round(grado2)
                  #print(grado2)
                  cv2.putText(frame,str(grado2),(x6+15,y6),cv2.FONT_HERSHEY_SIMPLEX,0.3,(255,0,255),1)
                  
                  angle3=np.math.atan2(np.linalg.det([v4,v5]),np.dot(v4,v5))#rad
                  grado3= np.degrees(angle3)
                  grado3=round(grado3)
                  #print(grado3)
                  cv2.putText(frame,str(grado3),(x5+15,y5),cv2.FONT_HERSHEY_SIMPLEX,0.3,(0,0,0),1)
                  
                  angle4=np.math.atan2(np.linalg.det([v6,v7]),np.dot(v6,v7))#rad
                  grado4= np.degrees(angle4)
                  grado4=round(grado4)
                  #print(grado4)
                  cv2.putText(frame,str(grado4),(x4+10,y4),cv2.FONT_HERSHEY_SIMPLEX,0.3,(0,255,255),1)
                  
                  angle5=np.math.atan2(np.linalg.det([v8,v9]),np.dot(v8,v9))#rad
                  grado5= np.degrees(angle5)
                  grado5=round(grado5)
                  #print(grado5)
                  cv2.putText(frame,str(grado5),(x3+5,y3-10),cv2.FONT_HERSHEY_SIMPLEX,0.3,(0,0,255),1)
                  
                  angle6=np.math.atan2(np.linalg.det([v10,v11]),np.dot(v10,v11))#rad
                  grado6= np.degrees(angle6)
                  grado6=round(grado6)   #(grado,2)
                  #print(grado6)
                  cv2.putText(frame,str(grado6),(x2,y2-10),cv2.FONT_HERSHEY_SIMPLEX,0.3,(0,255,0),1)
             
                                              
                  angulos.append(grado1)
                  angulos.append(grado2)
                  angulos.append(grado3)
                  angulos.append(grado4)
                  angulos.append(grado5)
                  angulos.append(grado6)
                  
                  # cont=cont+1
                  
                  #print(angulos)
                 
                  # ########################################################################################
             
                
             
                
             
                  letra = []
                   # print(df)
                  data_1 = df.iloc[:,0:6]
                  #  #segunda forma
                  #  #print(data_1, "\n")
                                     

                  clf=tree.DecisionTreeClassifier()
                  X=data_1
                  Y = df.iloc[:,6]
                
                  #Y=letra
                  clf=clf.fit(X, Y)
                 
                  tree.plot_tree(clf)
                  dato=angulos
                  
            

                  #  #dato=[93,8,97,-94,-135,31]
                  #  #dato=[86,7,113,-97,-135,27]

                                                  
                  prediccion=clf.predict([dato])
                  #  print(prediccion)
                    
                  if prediccion =="A":
                   texto="A"
                   cv2.putText(frame,str("A"),(500,200),cv2.FONT_HERSHEY_SIMPLEX,3,(0,255,0),5)
                   print(texto)
                   
                  elif prediccion =="B":
                   texto="B"
                   cv2.putText(frame,str("B"),(500,200),cv2.FONT_HERSHEY_SIMPLEX,3,(0,255,0),5)
                   print(texto)
                   
                  elif prediccion =="C":
                    texto="C"
                    cv2.putText(frame,str("C"),(500,200),cv2.FONT_HERSHEY_SIMPLEX,3,(0,255,0),5)
                    print(texto)
                    
                  elif prediccion =="D":
                     texto="D"
                     cv2.putText(frame,str("D"),(500,200),cv2.FONT_HERSHEY_SIMPLEX,3,(0,255,0),5)
                     print(texto)
                                         
                  elif prediccion =="E":
                   texto="E"
                   cv2.putText(frame,str("E"),(500,200),cv2.FONT_HERSHEY_SIMPLEX,3,(0,255,0),5)
                   print(texto)
                   
                  elif prediccion =="F":
                    texto="F"
                    cv2.putText(frame,str("F"),(500,200),cv2.FONT_HERSHEY_SIMPLEX,3,(0,255,0),5)
                    print(texto)
                    
                  elif prediccion =="G":
                      texto="G"
                      cv2.putText(frame,str("G"),(500,200),cv2.FONT_HERSHEY_SIMPLEX,3,(0,255,0),5)
                      print(texto)
                   
                  elif prediccion=="I":
                   texto="I"
                   cv2.putText(frame,str("I"),(500,200),cv2.FONT_HERSHEY_SIMPLEX,3,(0,255,0),5)
                   print(texto)
                   
                  elif prediccion=="O":
                   texto="O"
                   cv2.putText(frame,str("O"),(500,200),cv2.FONT_HERSHEY_SIMPLEX,3,(0,255,0),5)
                   print(texto)
                   
                  elif prediccion=="P":
                    texto="P"
                    cv2.putText(frame,str("P"),(500,200),cv2.FONT_HERSHEY_SIMPLEX,3,(0,255,0),5)
                    print(texto)
                   
                  elif prediccion=="R":
                   texto="R"
                   cv2.putText(frame,str("R"),(500,200),cv2.FONT_HERSHEY_SIMPLEX,3,(0,255,0),5)
                   print(texto) 
                   
                  elif prediccion=="U":
                   texto="U"
                   cv2.putText(frame,str("U"),(500,200),cv2.FONT_HERSHEY_SIMPLEX,3,(0,255,0),5)
                   print(texto)
                   
                  elif prediccion=="V":
                   texto="V"
                   cv2.putText(frame,str("V"),(500,200),cv2.FONT_HERSHEY_SIMPLEX,3,(0,255,0),5)
                   print(texto)
                   
                  elif prediccion=="H":
                    texto="H"
                    cv2.putText(frame,str("H"),(500,200),cv2.FONT_HERSHEY_SIMPLEX,3,(0,255,0),5)
                    print(texto)
                       
                  elif prediccion=="L":
                    texto="L"
                    cv2.putText(frame,str("L"),(500,200),cv2.FONT_HERSHEY_SIMPLEX,3,(0,255,0),5)
                    print(texto)
                    
                  elif prediccion=="N":
                      texto="N"
                      cv2.putText(frame,str("N"),(500,200),cv2.FONT_HERSHEY_SIMPLEX,3,(0,255,0),5)
                      print(texto)
                      
                  elif prediccion=="M":
                      texto="M"
                      cv2.putText(frame,str("M"),(500,200),cv2.FONT_HERSHEY_SIMPLEX,3,(0,255,0),5)
                      print(texto)
                      
                  elif prediccion=="W":
                      texto="W"
                      cv2.putText(frame,str("W"),(500,200),cv2.FONT_HERSHEY_SIMPLEX,3,(0,255,0),5)
                      print(texto)
                  elif prediccion=="Y":
                      texto="Y"
                      cv2.putText(frame,str("Y"),(500,200),cv2.FONT_HERSHEY_SIMPLEX,3,(0,255,0),5)
                      print(texto)
                  elif prediccion=="T":
                      texto="T"
                      cv2.putText(frame,str("T"),(500,200),cv2.FONT_HERSHEY_SIMPLEX,3,(0,255,0),5)
                      print(texto)
                      
                      
                  elif prediccion=="S":
                      texto="S"
                      cv2.putText(frame,str("S"),(500,200),cv2.FONT_HERSHEY_SIMPLEX,3,(0,255,0),5)
                      print(texto)
              
                   
                  else:
                   texto=" "
                   cv2.putText(frame,str(" "),(500,200),cv2.FONT_HERSHEY_SIMPLEX,3,(0,255,0),5)
                   print(texto)
                      
                     
                 # ######################################################################################
                  
             
                # mp_drawing.draw_landmarks(
                #     frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                #     mp_drawing.DrawingSpec(color=(0,0,255), thickness=3, circle_radius=3),
                #     mp_drawing.DrawingSpec(color=(0,255,0), thickness=4, circle_radius=5))
        cv2.namedWindow("image", cv2.WINDOW_NORMAL)
        cv2.setWindowProperty("image", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        cv2.imshow('Frame',frame)
        
       
        
        if cv2.waitKey(1) & 0xFF == 27:
            break
cap.release()
cv2.destroyAllWindows()




# cap = cv2.VideoCapture(0)


# # Códec:
# fourcc = cv2.VideoWriter_fourcc('D','I','V','3')

# out = cv2.VideoWriter('C:/Users/Gerardo/Desktop/investigacion/output.avi', fourcc, 20.0, (640, 480))
# #out = cv2.VideoWriter('C:\Users\Gerardo\Desktop\investigacion\output.avi', fourcc, 20.0, (int(cap.get(3)),int(cap.get(4))))


# # loop runs if capturing has been initialized.
# while True:
#     # reads frames from a camera
#     # ret checks return at each frame
#     ret, frame = cap.read()
#     frame= cv2.flip(frame, 1)
#     # The original input frame is shown in the window
#     cv2.imshow('Original', frame)

#     out.write(frame)

#     if cv2.waitKey(1) & 0xFF == ord('a'):
#         break

# # Close the window / Release webcam
# cap.release()

# # After we release our webcam, we also release the output
# out.release()

# # De-allocate any associated memory usage
# cv2.destroyAllWindows()

###########################################



dot_data = StringIO()
export_graphviz(clf, out_file=dot_data,  
                filled=True, rounded=True,
                special_characters=True,class_names=['A','B','C','D','E','F','G','H','I','L','M','N','O','P','R','S','T','U','V','W','Y'])#,max_depth=2)
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
graph.write_png('letras3.png')



ArithmeticError


Image(graph.create_png())