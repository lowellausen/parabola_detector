#UNIVERSIDADE FEDERAL DO RIO GRANDE DO SUL
#INSTITUTO DE INFORMÁTICA
#TRABALHO DA DISCIPLINA INF01030 - FUNDAMENTOS DE VISÃO COMPUTACIONAL
#Implementa̧c̃ao de algoritmo de calibra̧c̃ao de câmera
#Alunos:  Leonardo Oliveira Wellausen
#Matheus Fernandes Kovaleski
#Orientadore:  Prof.  Dr.  Cĺaudio Rosito Jung

import cv2   # biblioteca opencv utilizada para funções de imagem
import numpy as np  # biblioteca numpy para funções matemáticas, como a SDV
from include.parabola import Parabola
import random


def draw_square_at(img, pos, color):
    img[int(pos[1]) - 3:int(pos[1]) + 3, int(pos[0]) - 3:int(pos[0]) + 3] = color


def color_dist(v1, v2):
    return (v1 - v2)**2


def kmeans(img):
    c1, c2 = 0, 255
    prev_c1 = prev_c2 = -7
    g1 = []
    g2 = []
    while c1 != prev_c1 and c2 != prev_c2:
        g1 = []
        sum1 = 0
        count1 = 0
        g2 = []
        sum2 = 0
        count2 = 0

        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                val = img[i, j]
                dists = [color_dist(val, c) for c in [c1, c2]]
                if dists[0] < dists[1]:
                    g1.append((i, j))
                    sum1 += val
                    count1 += 1
                else:
                    g2.append((i, j))
                    sum2 += val
                    count2 += 1

        prev_c1 = c1
        prev_c2 = c2
        c1 = sum1 / count1
        c2 = sum2 / count2

        print(c1, c2)

    if c1 < c2:
        return g1, g2
    else:
        return g2, g1


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


def main(threshold, nvotes):
    # carregamos a imagem, dimensionamos uma janela para exibí-la
    img = cv2.imread('exemplo1 - Copy.jpg')
    size = (img.shape[1], img.shape[0])

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

    '''for i in range(size[0]):
        for j in range(size[1]):
            if grey_img[j, i] < threshold:
                grey_img[j, i] = 255
            else:
                grey_img[j, i] = 0'''


    cv2.namedWindow('thresholding', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('thresholding', size[0], size[1])
    cv2.imshow('thresholding', grey_img)

    b, p = kmeans(grey_img)

    for p in p:
        grey_img[p] = 0
    for p in b:
        grey_img[p] = 255

    cv2.namedWindow('k means', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('k means', size[0], size[1])
    cv2.imshow('k means', grey_img)

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

    sub = np.subtract(x2, x1)

    v1 = sub/np.linalg.norm(sub)

    inter = intersec(x1, x2, x3, x4)

    v2 = [v1[1], -v1[0]]

    new_x3 = np.add(inter, [v2[0]*-10000, v2[1]*-10000])
    new_x3 = tuple([int(i) for i in new_x3])
    new_x4 = np.add(inter, [v2[0]* 10000, v2[1]* 10000])
    new_x4 = tuple([int(i) for i in new_x4])

    cv2.line(img, x1, x2, (0, 0, 255), 3)
    cv2.line(img, new_x3, new_x4, (255, 0, 255), 3)
    cv2.line(grey_img, x1, x2, 150, 50)
    cv2.line(grey_img, x3, x4, 150, 50)

    draw_square_at(img, inter, [255, 0, 0])

    cv2.namedWindow('hough', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('hough', size[0], size[1])
    cv2.imshow('hough', img)

    cv2.namedWindow('houghg', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('houghg', size[0], size[1])
    cv2.imshow('houghg', grey_img)

    #return

    # tentar fazer uma erosão
    kernel = np.ones(ekernel_size, np.uint8)
    #grey_img = cv2.erode(grey_img, kernel, iterations=1)

    points = [np.array((i, j)) for i in range(size[0]) for j in range(size[1]) if grey_img[j, i] == 255]
    parab = Parabola(points, theta_diff)

    parab.draw(img)

    # exibimos a imagem por último para não receber cliques antes de tudo devidamente calculado
    cv2.namedWindow('image', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('image', size[0], size[1])
    cv2.imshow('image', img)

    cv2.namedWindow('grey image', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('grey image', size[0], size[1])
    cv2.imshow('grey image', grey_img)


if __name__ == '__main__':
    nvotes = 550
    threshold = 120
    main(threshold, nvotes)
    # ficamos em laço esperando o usuário ou fechar a janela ou clicar na imagem (botão esquerdo) para adicionar um jogador
    while 1:
        k = cv2.waitKey(0)

        print(k)
        #saida = cv2.destroyAllWindows()
        if k == 27:
            cv2.destroyAllWindows()
            break
        elif k == 110:
            nvotes = int(input('Insert new nvotes value:'))
            main(threshold, nvotes)
        elif k == 116:
            threshold = int(input('Insert new threshold value:'))
            main(threshold, nvotes)

