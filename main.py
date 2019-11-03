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


def draw_square_at(pos, color):
    global img
    img[int(pos[1]) - 3:int(pos[1]) + 3, int(pos[0]) - 3:int(pos[0]) + 3] = color


def parab_x(x, s):
    return(x ** 2) * s[0] + x * s[1] + s[2]


def intersec(p1, p2, p3, p4):
    x = 0
    y = 1
    numx = (p1[x]*p2[y] - p1[y]*p2[x])*(p3[x] - p4[x]) - (p1[x] - p2[x])*(p3[x]*p4[y] - p3[y]*p4[x])
    denx = (p1[x] - p2[x])*(p3[y] - p4[y]) - (p1[y] - p2[y])*(p3[x] - p4[x])

    numy = (p1[x]*p2[y] - p1[y]*p2[x])*(p3[y] - p4[y]) - (p1[y] - p2[y])*(p3[x]*p4[y] - p3[y]*p4[x])
    deny = (p1[x] - p2[x])*(p3[y] - p4[y]) - (p1[y] - p2[y])*(p3[x] - p4[x])

    return [(numx/denx), (numy/deny)]


def find_endpoints(t):
    rho = t[0]
    theta = t[1]
    print(np.degrees(theta), theta)
    a = np.cos(theta)
    b = np.sin(theta)
    x0 = a * rho
    y0 = b * rho
    x1 = int(x0 + 10000 * (-b))
    y1 = int(y0 + 10000 * a)
    x2 = int(x0 - 10000 * (-b))
    y2 = int(y0 - 10000 * a)
    t1 = theta

    return (x1, y1), (x2, y2)


def rotate(parab, theta):
    minparaby = min(parab, key=lambda t: t[1])[1]
    maxparaby = max(parab, key=lambda t: t[1])[1]
    minparabx = min(parab, key=lambda t: t[0])[0]
    maxparabx = max(parab, key=lambda t: t[0])[0]

    center_parab = ((minparabx + maxparabx) / 2, (minparaby + maxparaby) / 2)

    shifted_parab = []
    for p in parab:
        shiftp = (p[0] - center_parab[0], p[1] - center_parab[1])
        shifted_parab.append(shiftp)
    parab = shifted_parab

    rot_matrix = np.array(([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta), np.cos(theta)]
    ]))

    rotated_parab = []
    for p in parab:
        rot_par = np.matmul(rot_matrix, p)
        rotated_parab.append(rot_par)

    parab = rotated_parab

    shifted_parab = []
    for p in parab:
        shiftp = (p[0] + center_parab[0], p[1] + center_parab[1])
        shifted_parab.append(shiftp)
    parab = shifted_parab

    return parab


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

cv2.namedWindow('sdsds image', cv2.WINDOW_NORMAL)
cv2.resizeWindow('sdsds image', size[0], size[1])
cv2.imshow('sdsds image', grey_img)


#tentar fazer uma erosão
kernel = np.ones(ekernel_size, np.uint8)
grey_img = cv2.erode(grey_img, kernel, iterations=1)

cv2.namedWindow('sdsds a', cv2.WINDOW_NORMAL)
cv2.resizeWindow('sdsds a', size[0], size[1])
cv2.imshow('sdsds a', grey_img)



#  hough aqui
'''lines = cv2.HoughLinesP(grey_img, 1, np.pi/180, 200)
for line in lines:
    x1, y1, x2, y2 = line[0]
    cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 3)'''

lines = cv2.HoughLines(grey_img, 1, np.pi/180, nvotes)

for line in lines:
    x1, x2 = find_endpoints(line[0])
    cv2.line(img, x1, x2, (0, 255, 255), 3)

lines = sorted(lines, key=lambda t: t[0][1])

x1, x2 = find_endpoints(lines[-1][0])
x3, x4 = find_endpoints(lines[0][0])

t1 = lines[-1][0][1]
t2 = lines[0][0][1]

theta_diff = np.abs(t2 - np.pi/2)
print('Teta diffffff: ', np.degrees(theta_diff))

sub = np.subtract(x2, x1)

v1 = sub/np.linalg.norm(sub)

inter = intersec(x1, x2, x3, x4)

v2 = [v1[1], -v1[0]]

print("NORMA DO V2 AQUI OIE", np.linalg.norm(v2))


new_x3 = np.add(inter, [v2[0]*-10000, v2[1]*-10000])
new_x3 = tuple([int(i) for i in new_x3])
new_x4 = np.add(inter, [v2[0]* 10000, v2[1]* 10000])
new_x4 = tuple([int(i) for i in new_x4])


cv2.line(img, x1, x2, (0, 0, 255), 3)
cv2.line(img, new_x3, new_x4, (255, 0, 255), 3)
cv2.line(grey_img, x1, x2, (0, 0, 0), 40)
cv2.line(grey_img, x3, x4, (0, 0, 0), 40)

draw_square_at(inter, [255, 0, 0])


parab = [np.array((i, j)) for i in range(size[0]) for j in range(size[1]) if grey_img[j, i] == 255]

parab = rotate(parab, theta_diff)

'''projected_parab =[]
matriz_braba = np.array(([v2[1], v1[1]],
                          [v2[0], v1[0]
                          ]))
matriz_braba_inv = np.linalg.inv(matriz_braba)
for p in parab:
    projected_point = np.matmul(matriz_braba_inv, np.array([p[1], p[0]]))
    projected_parab.append((int(projected_point[1]), int(projected_point[0])))

parab = projected_parab'''



'''thin_parab = []
prev_x = parab[0][1]
pysum = 0
pycount = 0
for p in parab:
    if p[1] == prev_x:
        pysum += p[0]
        pycount += 1
    else:
        pymean = pysum // pycount
        thin_parab.append((pymean, prev_x))
        prev_x = p[1]
        pysum = p[0]
        pycount = 1

parab = thin_parab'''

#for p in parab:
#    draw_square_at(p, [255, 0, 255])

a_size = (len(parab), 3)
b_size = (len(parab), 1)
x_size = (3, 1)

a_matrix = np.zeros(a_size)
b_matrix = np.zeros(b_size)
x_matrix = np.zeros(x_size)

for i in range(a_size[0]):
    a_matrix[i, 0] = parab[i][0] ** 2
    a_matrix[i, 1] = parab[i][0]
    a_matrix[i, 2] = 1

for i in range(b_size[0]):
    b_matrix[i] = parab[i][1]

ata = np.matmul(np.transpose(a_matrix), a_matrix)
atb = np.matmul(np.transpose(a_matrix), b_matrix)

sol = np.linalg.solve(ata, atb)

print(parab[0])
print((parab[0][0]**2)*sol[0] + parab[0][0]*sol[1] + sol[2])

stip_parab = []
for i in range(0, size[0]):
    res = parab_x(i, sol)
    if res < 0:
        continue
    stip_parab.append((i, res))

parab = rotate(stip_parab, -theta_diff)

for p in parab:
    draw_square_at(p, [0, 255, 0])

'''for i in range(minparab, maxparab):
    res = parab_x(i, sol)
    res_screen = np.matmul(matriz_braba, np.array([res, i]))
    if res_screen[1] < 0:
        continue
    draw_square_at(res_screen, [0, 255, 0])'''

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
