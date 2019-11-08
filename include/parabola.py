import numpy as np
import random
from math import floor


def draw_square_at(img, pos, color):
    img[int(pos[1]) - 3:int(pos[1]) + 3, int(pos[0]) - 3:int(pos[0]) + 3] = color


class Parabola:
    def __init__(self, points, angle):
        self.points = points

        self.miny = min(points, key=lambda t: t[1])[1]
        self.maxy = max(points, key=lambda t: t[1])[1]
        self.minx = min(points, key=lambda t: t[0])[0]
        self.maxx = max(points, key=lambda t: t[0])[0]
        self.center = ((self.minx + self.maxx) / 2, (self.miny + self.maxy) / 2)

        self.rotate(angle)
        #self.thin()
        self.equation = self.ransac()
        #self.equation = self.stipulate_equation(self.points)
        self.points = self.stipulate_parabola()
        self.rotate(-angle)

    def f_x(self, x):
        s = self.equation
        return (x ** 2) * s[0] + x * s[1] + s[2]

    def draw(self, img):
        for p in self.points:
            draw_square_at(img, p, [0, 255, 0])

    def stipulate_parabola(self):
        stip_parab = []
        for x in range(self.minx, self.maxx):
            y = self.f_x(x)
            if y < 0:
                continue
            stip_parab.append((x, y))

        return stip_parab

    def ransac(self):
        best_fit = None
        best_error = np.inf

        for indice in range(100):
            maybe_ins = random.choices(self.points, k=3)
            equa = self.equation = self.stipulate_equation(maybe_ins)
            while equa is 'Singular':
                maybe_ins = random.choices(self.points, k=3)
                equa = self.equation = self.stipulate_equation(maybe_ins)

            ins = []
            for p in self.points:
                y = self.f_x(p[0])
                if np.abs(y - p[1]) < 10:
                    ins.append(p)

            if len(ins) >= len(self.points)*0.2:
                better_model = self.equation = self.stipulate_equation(ins)
                err = 0
                for p in self.points:
                    y = self.f_x(p[0])
                    err += np.abs(y - p[1])
                if err < best_error:
                    best_error = err
                    best_fit = better_model
                    print(len(self.points), len(ins), err)

        return best_fit

    def stipulate_equation(self, points):
        a_size = (len(points), 3)
        b_size = (len(points), 1)

        a_matrix = np.zeros(a_size)
        b_matrix = np.zeros(b_size)

        for i in range(a_size[0]):
            a_matrix[i, 0] = points[i][0] ** 2
            a_matrix[i, 1] = points[i][0]
            a_matrix[i, 2] = 1

        for i in range(b_size[0]):
            b_matrix[i] = points[i][1]

        ata = np.matmul(np.transpose(a_matrix), a_matrix)
        atb = np.matmul(np.transpose(a_matrix), b_matrix)

        try:
            return np.linalg.solve(ata, atb)
        except np.linalg.LinAlgError:
            return 'Singular'

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

    def rotate(self, theta):

        shifted_parab = []
        for p in self.points:
            shiftp = (p[0] - self.center[0], p[1] - self.center[1])
            shifted_parab.append(shiftp)
        self.points = shifted_parab

        rot_matrix = np.array(([
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta), np.cos(theta)]
        ]))

        rotated_parab = []
        for p in self.points:
            rot_par = np.matmul(rot_matrix, p)
            rotated_parab.append(rot_par)

        self.points = rotated_parab

        shifted_parab = []
        for p in self.points:
            shiftp = (floor(p[0] + self.center[0]), floor(p[1] + self.center[1]))
            shifted_parab.append(shiftp)
        self.points = shifted_parab
