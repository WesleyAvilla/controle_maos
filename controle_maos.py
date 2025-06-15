import cv2
import mediapipe as mp
import pyautogui
import math
import numpy as np

# Tamanho da tela
screen_width, screen_height = pyautogui.size()

# Tamanho do vídeo
frame_width, frame_height = 640, 480

# Inicializações
cap = cv2.VideoCapture(0)
cap.set(3, frame_width)
cap.set(4, frame_height)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)
mp_draw = mp.solutions.drawing_utils

# Suavização do movimento
prev_x, prev_y = 0, 0
smoothening = 5  # quanto maior, mais suave

while True:
    success, frame = cap.read()
    frame = cv2.flip(frame, 1)  # espelha a imagem
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            lm_list = []
            for id, lm in enumerate(hand_landmarks.landmark):
                cx, cy = int(lm.x * frame_width), int(lm.y * frame_height)
                lm_list.append((cx, cy))

            if lm_list:
                # Coordenadas do indicador (8) e polegar (4)
                x1, y1 = lm_list[8]  # dedo indicador
                x2, y2 = lm_list[4]  # polegar
                x3, y3 = lm_list[12] # dedo médio

                # Movimentar mouse
                screen_x = np.interp(x1, (100, frame_width - 100), (0, screen_width))
                screen_y = np.interp(y1, (100, frame_height - 100), (0, screen_height))

                # Suavização
                curr_x = prev_x + (screen_x - prev_x) / smoothening
                curr_y = prev_y + (screen_y - prev_y) / smoothening
                pyautogui.moveTo(curr_x, curr_y)
                prev_x, prev_y = curr_x, curr_y

                # Desenhar pontos na tela
                cv2.circle(frame, (x1, y1), 10, (255, 0, 255), cv2.FILLED)
                cv2.circle(frame, (x2, y2), 10, (0, 255, 0), cv2.FILLED)
                cv2.line(frame, (x1, y1), (x2, y2), (255, 255, 0), 2)

                # Distância entre polegar e indicador (para clique)
                dist = math.hypot(x2 - x1, y2 - y1)
                if dist < 40:
                    pyautogui.click()
                    cv2.circle(frame, ((x1 + x2) // 2, (y1 + y2) // 2), 15, (0, 255, 0), cv2.FILLED)

                # Scroll com o dedo médio
                if y3 < y1 - 50:  # dedo médio acima do indicador
                    pyautogui.scroll(20)
                elif y3 > y1 + 50:  # dedo médio abaixo
                    pyautogui.scroll(-20)

            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    cv2.imshow("Controle com a Mão", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
