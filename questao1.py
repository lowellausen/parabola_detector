#UNIVERSIDADE FEDERAL DO RIO GRANDE DO SUL
#INSTITUTO DE INFORMÁTICA
#TRABALHO DA DISCIPLINA INF01030 - FUNDAMENTOS DE VISÃO COMPUTACIONAL
#Implementa̧c̃ao de algoritmo de detecção de parábolas
#Alunos:  Leonardo Oliveira Wellausen
#Matheus Fernandes Kovaleski
#Orientadore:  Prof.  Dr.  Cĺaudio Rosito Jung

import cv2   # biblioteca opencv utilizada para funções de imagem
import numpy as np  # biblioteca numpy para funções matemáticas, como a SDV
import sys
import random
from math import floor


# classe representando uma parábola
class Parabola:
    def __init__(self, points, angle, imgsize):
        # os pontos representando a parábola inicialmente
        self.points = points

        self.imgsize = imgsize

        # dimensóes para translação
        self.miny = min(points, key=lambda t: t[1])[1]
        self.maxy = max(points, key=lambda t: t[1])[1]
        self.minx = min(points, key=lambda t: t[0])[0]
        self.maxx = max(points, key=lambda t: t[0])[0]
        self.center = ((self.minx + self.maxx) / 2, (self.miny + self.maxy) / 2)

        # rotamos para poder estimar a parábola com sistema linear
        self.rotate(angle)
        # afinamente para estimar melhor (meme)
        #self.thin()
        # desce um ransac aí
        self.equation = self.ransac()
        #self.equation = self.stipulate_equation(self.points)
        # atualizamos os pontos que agora são calculaods pela equação encontrada
        self.points = self.stipulate_parabola()
        # desfaz a rotação
        self.rotate(-angle)

    # função f(x) para a paábola, usa a equação definida em equation e a aplica em um ponto x
    def f_x(self, x):
        s = self.equation
        return (x ** 2) * s[0] + x * s[1] + s[2]

    # função que desenha na imagem os pontos points
    def draw(self, img):
        for p in self.points:
            draw_square_at(img, p, [0, 255, 0])

    # calcula pontos para a parábola baseandose na equação que tá lá em equation
    def stipulate_parabola(self):
        stip_parab = []
        for x in range(0 , self.imgsize[1]):
            y = self.f_x(x)
            if 0 > y or y >= self.imgsize[0]:
                continue
            stip_parab.append((x, y))

        return stip_parab

    # ransac da massa
    def ransac(self):
        print('Começando ransac!')
        # por enquanto sem nenhum melhor fit
        best_fit = None
        best_error = np.inf

        # fazemos o laço 100 vezes!! arbitrário
        for indice in range(100):
            # pegamos 3 pontos aleatórios e estimamos uma equação com eles
            maybe_ins = random.choices(self.points, k=3)
            equa = self.equation = self.stipulate_equation(maybe_ins)
            # dependendo dos pontos a matriz é singular ):
            while equa is 'Singular':
                maybe_ins = random.choices(self.points, k=3)
                equa = self.equation = self.stipulate_equation(maybe_ins)

            # achamos os inliers para a equação estimada
            ins = []
            for p in self.points:
                y = self.f_x(p[0])
                if np.abs(y - p[1]) < 10:
                    ins.append(p)

            # se mais de 20% dos pontos são inliers (arbitrário, mas melhor prevenir do que remediar. calculamos uma nova esqualão com os inliers
            if len(ins) >= len(self.points)*0.2:
                better_model = self.equation = self.stipulate_equation(ins)
                err = 0
                for p in self.points:
                    y = self.f_x(p[0])
                    err += np.abs(y - p[1])
                # se o erro desse melhor modelo é melhor que o melho so far, aualizamos
                if err < best_error:
                    best_error = err
                    best_fit = better_model
                    print('Novo melhor erro: ', err)

        # retorna o melhor dos modelos
        return best_fit

    # modela e resolve um sistema linear mdefinidos por points, deveria ser estática mas whatever olhaesse código
    def stipulate_equation(self, points):
        a_size = (len(points), 3)
        b_size = (len(points), 1)

        a_matrix = np.zeros(a_size)
        b_matrix = np.zeros(b_size)

       # monta a matriz A
        for i in range(a_size[0]):
            a_matrix[i, 0] = points[i][0] ** 2
            a_matrix[i, 1] = points[i][0]
            a_matrix[i, 2] = 1

        # monta a matriz
        for i in range(b_size[0]):
            b_matrix[i] = points[i][1]

        # montamos o sistema do minímo erro quadrado
        ata = np.matmul(np.transpose(a_matrix), a_matrix)
        atb = np.matmul(np.transpose(a_matrix), b_matrix)

        # resolve o sistema se ele não for singular e retorna a equação encontrada
        try:
            return np.linalg.solve(ata, atb)
        except np.linalg.LinAlgError:
            return 'Singular'

    # afina a parábola, para cada X define seu Y como a média de todos o Y daquele X. antes a parábola era gorda, com ransac não precisa disso não
    def thin(self):
        thin_parab = []
        prev_x = self.points[0][0]
        pysum = 0
        pycount = 0
        for p in self.points:
            if p[0] == prev_x:
                pysum += p[1]
                pycount += 1
            else:
                pymean = pysum // pycount
                thin_parab.append((prev_x, pymean))
                prev_x = p[0]
                pysum = p[1]
                pycount = 1

        self.points = thin_parab

    # rotaciona seus pontos pelo ângulo theta
    def rotate(self, theta):

        # primeiro shiftamos os pontos para a origem
        shifted_parab = []
        for p in self.points:
            shiftp = (p[0] - self.center[0], p[1] - self.center[1])
            shifted_parab.append(shiftp)
        self.points = shifted_parab

        # monta a matriz de rotação
        rot_matrix = np.array(([
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta), np.cos(theta)]
        ]))

        # roda roda roda
        rotated_parab = []
        for p in self.points:
            rot_par = np.matmul(rot_matrix, p)
            rotated_parab.append(rot_par)

        self.points = rotated_parab

        # desshifta
        shifted_parab = []
        for p in self.points:
            shiftp = (floor(p[0] + self.center[0]), floor(p[1] + self.center[1]))
            shifted_parab.append(shiftp)
        self.points = shifted_parab


# desenha um quadrado na img em pos com color
def draw_square_at(img, pos, color):
    y = pos[1]
    if 0 > y or y >= img.shape[0]:
        return
    img[int(pos[1]) - 3:int(pos[1]) + 3, int(pos[0]) - 3:int(pos[0]) + 3] = color


# calcula a distância entre as cores v1 e v2. wtf v e não c???
def color_dist(v1, v2):
    return (v1 - v2)**2


# kmeans da loucura
def kmeans(img):
    print('Começando K-means')
    # centróides iniciados como preto e branco porque aleatório tava um lixo
    c1, c2 = 0, 255
    prev_c1 = prev_c2 = -7
    # grupos definidos a partir dos centróides
    g1 = []
    g2 = []
    # ajusta os centróides até convergir o que acontece incrivelmente rápido
    while c1 != prev_c1 and c2 != prev_c2:
        g1 = []
        sum1 = 0
        count1 = 0
        g2 = []
        sum2 = 0
        count2 = 0

        # percorremos a imagem atribuindo cada pixel a um grupo de acordo com sua cor apartheid???
        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                val = img[i, j]
                dists = [color_dist(val, c) for c in [c1, c2]]
                # lasca no grupo 1
                if dists[0] < dists[1]:
                    g1.append((i, j))
                    sum1 += val
                    count1 += 1
                # toca no grupo 2
                else:
                    g2.append((i, j))
                    sum2 += val
                    count2 += 1

        print('Centroide 1: ', c1, 'Centroide 2: ', c2)
        # atualizamos os centróides de acordo com a média de cor em cada grupo!
        prev_c1 = c1
        prev_c2 = c2
        c1 = sum1 / count1
        c2 = sum2 / count2

    # retorna sempre preto depois branco
    if c1 < c2:
        return g1, g2
    else:
        return g2, g1


# calcula a intersecção entre as retas definidas por (p1, p2) e (p3, p4)
def intersec(p1, p2, p3, p4):
    x = 0
    y = 1
    numx = (p1[x]*p2[y] - p1[y]*p2[x])*(p3[x] - p4[x]) - (p1[x] - p2[x])*(p3[x]*p4[y] - p3[y]*p4[x])
    denx = (p1[x] - p2[x])*(p3[y] - p4[y]) - (p1[y] - p2[y])*(p3[x] - p4[x])

    numy = (p1[x]*p2[y] - p1[y]*p2[x])*(p3[y] - p4[y]) - (p1[y] - p2[y])*(p3[x]*p4[y] - p3[y]*p4[x])
    deny = (p1[x] - p2[x])*(p3[y] - p4[y]) - (p1[y] - p2[y])*(p3[x] - p4[x])

    return [(numx/denx), (numy/deny)]


# dada uma reta em coordendas polares (cuspida por hough) calcula dois endpoints cartesianos
def find_endpoints(t):
    rho = t[0]
    theta = t[1]
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


def main(nvotes, name, debug):
    # carregamos a imagem, dimensionamos uma janela para exibí-la
    print('Abrindo imagem ', name)
    img = cv2.imread(name)
    size = (img.shape[1], img.shape[0])

    #kerel da dilatação perdido aqui bem doido
    ekernel_size = (3, 3)

    #converte imagem bgr pra grey scale
    grey_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


    # aborto de anaálise de histograma pra thresholding
    '''hist = [0 for i in range(255)]
    avg = 0
    for i in range(size[0]):
        for j in range(size[1]):
            hist[grey_img[j, i]] += 1
            avg += grey_img[j, i]
    
    avg /= size[0]*size[1]'''

    # aborto de limiarização
    '''for i in range(size[0]):
        for j in range(size[1]):
            if grey_img[j, i] < threshold:
                grey_img[j, i] = 255
            else:
                grey_img[j, i] = 0

    if debug:
        cv2.namedWindow('thresholding', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('thresholding', size[0], size[1])
        cv2.imshow('thresholding', grey_img)'''

    # recebe os clusters preto e branco do ká médias
    b, p = kmeans(grey_img)

    # binarização! printa o branco (quadro) de preto e o preto (caneta) de branco, para mandar pro hough
    for p in p:
        grey_img[p] = 0
    for p in b:
        grey_img[p] = 255

    # dilata
    kernel = np.ones(ekernel_size, np.uint8)
    grey_img = cv2.dilate(grey_img, kernel, iterations=1)

    if debug:
        cv2.namedWindow('k means', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('k means', size[0], size[1])
        cv2.imshow('k means', grey_img)

    # aplica a transformada de hough para encontrar os eixos cartesianos e muito mais
    lines = cv2.HoughLines(grey_img, 1, np.pi/180, nvotes)

    if debug:
        for line in lines:
            x1, x2 = find_endpoints(line[0])
            cv2.line(img, x1, x2, (0, 255, 255), 3)

    # ordena por theta
    lines = sorted(lines, key=lambda t: t[0][1], reverse=True)

    # pega a primeira reta como eixo 1. e então acha a reta que te angulo mais próximo de 90 com esta
    x1, x2 = find_endpoints(lines[0][0])
    t1 = lines[0][0][1]
    best_dif = np.inf
    best_t2 = lines[0][0][1]
    #while not(80 <= np.abs(np.degrees(t1) - np.degrees(t2)) <= 100):
    for line in lines:
        t2 = line[0][1]
        diff = np.abs(np.degrees(t1) - np.degrees(t2))
        if 80 <= diff <= 100:
            if np.abs(diff - 90) <= best_dif:
                best_dif = diff
                best_t2 = t2
                x3, x4 = find_endpoints(line[0])

    # theta diff pra rotar a parábola
    t2 = best_t2
    theta_diff = np.abs(t2 - np.pi/2)

    print("Ângulo theta do eixo 1: ", t1, 'Ângulo theta do eixo 2: ', t2)

    # calcula intersecção dos dois eixos
    inter = intersec(x1, x2, x3, x4)

    # gera um vetor represetando o primeiro eixo
    sub = np.subtract(x2, x1)
    v1 = sub/np.linalg.norm(sub)
    # rotaciona o primeiro eixo em 90 graus!
    v2 = [v1[1], -v1[0]]

    # translada o ponto de intersecção na direção do cara perpendcular ao eixo 1!
    new_x3 = np.add(inter, [v2[0]*-10000, v2[1]*-10000])
    new_x3 = tuple([int(i) for i in new_x3])
    new_x4 = np.add(inter, [v2[0]* 10000, v2[1]* 10000])
    new_x4 = tuple([int(i) for i in new_x4])

    # apaga eixos na imagem binária desenha exos na imagem colorida
    cv2.line(img, x1, x2, (0, 0, 255), 3)
    cv2.line(img, new_x3, new_x4, (255, 0, 255), 3)
    if debug:
        cv2.line(img, x3, x4, (255, 0, 0), 3)
    cv2.line(grey_img, x1, x2, 0, 50)
    cv2.line(grey_img, x3, x4, 0, 50)

    # desenha o ponto de interseção perdido aqui
    draw_square_at(img, inter, [255, 0, 0])

    if debug:
        cv2.namedWindow('hough', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('hough', size[0], size[1])
        cv2.imshow('hough', img)

        cv2.namedWindow('houghg', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('houghg', size[0], size[1])
        cv2.imshow('houghg', grey_img)

    #return

    # coleta os pontos restantes que devem pertencer à parábola ou lixo!
    points = [np.array((i, j)) for i in range(size[0]) for j in range(size[1]) if grey_img[j, i] == 255]
    # gera um objeto parábola que magicamente faz tudo já, programação orientada à loucura
    parab = Parabola(points, theta_diff, img.shape)

    # desenha a par[abola mágica na imagem
    parab.draw(img)

    # exibimos a imagem final
    cv2.namedWindow('image', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('image', size[0], size[1])
    cv2.imshow('image', img)

    if debug:
        cv2.namedWindow('grey image', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('grey image', size[0], size[1])
        cv2.imshow('grey image', grey_img)


if __name__ == '__main__':
    nvotes = 500
    # lê os parâmetros orientado à gambiarra
    try:
        if sys.argv[2] == 'debug':
            debug = True
        else:
            debug = False
    except IndexError:
        debug = False
    main(nvotes, sys.argv[1], debug)
    # ficamos em laço esperando o usuário ou fechar a janela ou clicar na imagem (botão esquerdo) para adicionar um jogador
    while 1:
        k = cv2.waitKey(0)

        saida = cv2.destroyAllWindows()
        if (saida == None):
            break


