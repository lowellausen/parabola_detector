import numpy as np


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
        self.thin()
        self.equation = self.stipulate_equation()
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

    def stipulate_equation(self):
        a_size = (len(self.points), 3)
        b_size = (len(self.points), 1)

        a_matrix = np.zeros(a_size)
        b_matrix = np.zeros(b_size)

        for i in range(a_size[0]):
            a_matrix[i, 0] = self.points[i][0] ** 2
            a_matrix[i, 1] = self.points[i][0]
            a_matrix[i, 2] = 1

        for i in range(b_size[0]):
            b_matrix[i] = self.points[i][1]

        ata = np.matmul(np.transpose(a_matrix), a_matrix)
        atb = np.matmul(np.transpose(a_matrix), b_matrix)

        return np.linalg.solve(ata, atb)

    def thin(self):
        thin_parab = []
        prev_x = self.points[0][1]
        pysum = 0
        pycount = 0
        for p in self.points:
            if p[1] == prev_x:
                pysum += p[0]
                pycount += 1
            else:
                pymean = pysum // pycount
                thin_parab.append((pymean, prev_x))
                prev_x = p[1]
                pysum = p[0]
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
            shiftp = (p[0] + self.center[0], p[1] + self.center[1])
            shifted_parab.append(shiftp)
        self.points = shifted_parab
