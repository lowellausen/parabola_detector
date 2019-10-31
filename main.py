#UNIVERSIDADE FEDERAL DO RIO GRANDE DO SUL
#INSTITUTO DE INFORMÁTICA
#TRABALHO DA DISCIPLINA INF01030 - FUNDAMENTOS DE VISÃO COMPUTACIONAL
#Implementa̧c̃ao de algoritmo de calibra̧c̃ao de câmera
#Alunos:  Leonardo Oliveira Wellausen
#Matheus Fernandes Kovaleski
#Orientadore:  Prof.  Dr.  Cĺaudio Rosito Jung

import cv2   # biblioteca opencv utilizada para funções de imagem
import numpy as np  # biblioteca numpy para funções matemáticas, como a SDV
import os


def draw_square_at(pos):
    global img
    img[int(pos[1]) - 3:int(pos[1]) + 3, int(pos[0]) - 3:int(pos[0]) + 3] = [255, 0, 0]

# carregamos a imagem, dimensionamos uma janela para exibí-la
img = cv2.imread('exemplo1.jpg')
size = (img.shape[1], img.shape[0])

nvotes = 350
threshold = 120
ekernel_size = (3, 3)

#converte imagem bgr pra grey scale
grey_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

'''hist = [0 for i in range(255)]
avg = 0
for i in range(size[0]):
    for j in range(size[1]):
        hist[grey_img[j, i]] += 1
        avg += grey_img[j, i]

avg /= size[0]*size[1]'''

for i in range(size[0]):
    for j in range(size[1]):
        if grey_img[j, i] < threshold:
            grey_img[j, i] = 255
        else:
            grey_img[j, i] = 0

#tentar fazer uma erosão
kernel = np.ones(ekernel_size, np.uint8)
grey_img = cv2.erode(grey_img, kernel, iterations=1)


#  hough aqui
'''lines = cv2.HoughLinesP(grey_img, 1, np.pi/180, 200)
for line in lines:
    x1, y1, x2, y2 = line[0]
    cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 3)'''

lines = cv2.HoughLines(grey_img, 1, np.pi/180, nvotes)
t1 = 0
for line in lines:
    for rho, theta in line:
        ori_sin = np.sin(theta)
        ori_cos = np.cos(theta)
        ori_rho = rho
        if t1 != 0:
            theta = t1 - np.pi/2
        a = np.cos(theta)
        b = np.sin(theta)
        rho = rho - (b*rho - ori_sin*rho)
        ori_x0 = ori_cos * rho
        ori_y0 = ori_sin * rho
        ori_x1 = int(ori_x0 + 10000 * (-ori_sin))
        ori_y1 = int(ori_y0 + 10000 * ori_cos)
        ori_x2 = int(ori_x0 - 10000 * (-ori_sin))
        ori_y2 = int(ori_y0 - 10000 * ori_cos)
        x0 = a*rho
        y0 = b*rho
        x1 = int(x0 + 10000 * (-b))
        y1 = int(y0 + 10000 * a)
        x2 = int(x0 - 10000 * (-b))
        y2 = int(y0 - 10000 * a)
        t1 = theta

        cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 3)
        cv2.line(grey_img, (ori_x1, ori_y1), (ori_x2, ori_y2), (0, 0, 0), 40)


parab = [(j, i) for i in range(size[0]) for j in range(size[1]) if grey_img[j, i] == 255]

a_size = (len(parab), 3)
b_size = (len(parab), 1)
x_size = (3, 1)

a_matrix = np.zeros(a_size)
b_matrix = np.zeros(b_size)
x_matrix = np.zeros(x_size)

for i in range(a_size[0]):
    a_matrix[i, 0] = parab[i][1] ** 2
    a_matrix[i, 1] = parab[i][1]
    a_matrix[i, 2] = 1

for i in range(b_size[0]):
    b_matrix[i] = parab[i][0]

ata = np.matmul(np.transpose(a_matrix), a_matrix)
atb = np.matmul(np.transpose(a_matrix), b_matrix)

sol = np.linalg.solve(ata, atb)

print(parab[2000])
print((parab[2000][1]**2)*sol[0] + parab[2000][1]*sol[1] + sol[2])

# exibimos a imagem por último para não receber cliques antes de tudo devidamente calculado
cv2.namedWindow('image', cv2.WINDOW_NORMAL)
cv2.resizeWindow('image', size[0], size[1])
cv2.imshow('image', img)

cv2.namedWindow('grey image', cv2.WINDOW_NORMAL)
cv2.resizeWindow('grey image', size[0], size[1])
cv2.imshow('grey image', grey_img)

# ficamos em laço esperando o usuário ou fechar a janela ou clicar na imagem (botão esquerdo) para adicionar um jogador
while 1:
    k = cv2.waitKey(0)

    saida = cv2.destroyAllWindows()
    if (saida == None):
        break
