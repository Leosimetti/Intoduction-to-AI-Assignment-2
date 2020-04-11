import random
import numpy as np
from PIL import Image


def mse(imageA, imageB):
    # the 'Mean Squared Error' between the two images is the
    # sum of the squared difference between the two images;
    # NOTE: the two images must have the same dimension

    # print(imageA)
    # print(imageB)

    # print("ATTEMPT")

    imageA = np.delete(imageA, 3, 2)
    imageB = np.delete(imageB, 3, 2)

    t1 = imageA.astype("float")
    # print("LOL")
    t2 = imageB.astype("float")

    # imageA.shape = (128,128,3)

    # before = imageA[0]
    # after =

    # print(f'BEFORE:{before}  AFTER:{after}')

    # print("Converted")

    err = np.sum((t1 - t2) ** 2)
    # print("Squared")

    shape0 = imageA.shape[0]
    shape1 = imageA.shape[1]
    # print("Created shapes")
    err /= float(shape0 * shape1)
    # print("WORKED")

    # return the MSE, the lower the error, the more "similar"
    # the two images are
    return err


def random_point(WIDTH, HEIGHT):
    return [random.randint(0, WIDTH - 1), random.randint(0, HEIGHT - 1)]


def random_pallet():
    sas = [random.randint(1, 100) / 100 for i in range(4)]
    sas[3] = 1.0
    return sas


def legal(val, WIDTH, HEIGHT):
    if (val[0] >= 0 and val[0] <= WIDTH) and (val[1] >= 0 and val[1] <= HEIGHT):
        return True
    else:
        return False


def random_points_3(WIDTH, HEIGHT):
    center = random_point(WIDTH, HEIGHT)
    rad = random.randint(1, 15)

    # Varying X
    t1 = center.copy()
    t2 = center.copy()

    t1[0] = center[0] + rad
    t2[0] = center[0] - rad

    if legal(t1, WIDTH, HEIGHT):
        p1 = t1.copy()
    else:
        p1 = t2.copy()

    # Varying Y
    t1 = center.copy()
    t2 = center.copy()

    t1[1] = center[1] + rad
    t2[1] = center[1] - rad

    if legal(t1, WIDTH, HEIGHT):
        p2 = t1.copy()
    else:
        p2 = t2.copy()

    return [center, p1, p2]


def remove_alpha(image, color=(255, 255, 255)):
    image.load()  # needed for split()
    background = Image.new('RGB', image.size, color)
    background.paste(image, mask=image.split()[3])  # 3 is the alpha channel
    return background


def convert(img_path):
    png = Image.open(img_path)
    png.load()  # required for png.split()

    background = Image.new("RGB", png.size, (0, 0, 0))
    background.paste(png, mask=png.split()[3])  # 3 is the alpha channel

    background.save(img_path, 'PNG', quality=100)


def remove_grid(path):
    pic = Image.open(path)
    pxls = pic.load()
    WIDTH, HEIGHT = pic.size
    for i in range(0, HEIGHT):
        pxls[WIDTH - 1, i] = pxls[WIDTH - 2, i]
        pxls[i, WIDTH - 1] = pxls[i, WIDTH - 2]
    pic.save(path)
