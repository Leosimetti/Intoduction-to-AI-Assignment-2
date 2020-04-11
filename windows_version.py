from tkinter import filedialog, Tk
import cairo
import cv2
from PIL import ImageOps
from deap import base, creator
import os
from additional.genetic_things import *

root = Tk()
root.withdraw()

# os.chdir(r'C:\tmp\Genetic')

IMAGE_TO_WORK_ON = filedialog.askopenfilename()  # r'kesya.jpg'
print(IMAGE_TO_WORK_ON)
PIXEL_SCALE = 100
NUMBER_OF_FIGURES = 50  # round(WIDTH*HEIGHT/2)

ATTR_MUTATION = 0.2
MUTATION = 0.1

GENERATIONS = 1
POPULATION = 5


## GOOD ones
# NUMBER_OF_FIGURES = 5000#round(WIDTH*HEIGHT/2)
#
# ATTR_MUTATION = 0.2
# MUTATION = 0.1
#
# GENERATIONS = 10
# POPULATION = 50

def draw_triangle(cr, points, pallet):
    r, g, b = [pallet[i] for i in range(3)]
    p1, p2, p3 = [points[i] for i in range(3)]

    cr.set_source_rgb(r, g, b)
    cr.move_to(p1[0], p1[1])
    cr.line_to(p2[0], p2[1])
    cr.line_to(p3[0], p3[1])
    cr.fill()


def draw_individual(individual):
    ims = cairo.ImageSurface(cairo.FORMAT_ARGB32, WIDTH, HEIGHT)
    cr = cairo.Context(ims)
    for figure in individual:
        draw_triangle(cr, figure[0], figure[1])

    buf = ims.get_data()
    array = np.ndarray(shape=(WIDTH, HEIGHT, 4), dtype=np.uint8, buffer=buf)

    ims.finish()

    return array


def show_ind(individual, path):
    ims = cairo.ImageSurface(cairo.FORMAT_ARGB32, WIDTH, HEIGHT)
    cr = cairo.Context(ims)
    for figure in individual:
        draw_triangle(cr, figure[0], figure[1])

    ims.write_to_png(path)
    convert(path)


def err(individual):
    attempt = draw_individual(individual)
    return mse(attempt, source),


def rand_attribute():
    sas = random_points_3(WIDTH, HEIGHT)

    origin = sas[0]
    Ox = origin[0]
    Oy = origin[1]
    src_color = list(source[Ox, Oy])
    kek = [c / 255 for c in src_color]

    # kek = random_pallet()
    return [sas, kek]


def mutate(individual, indpb):
    for i in range(len(individual)):
        if random.random() <= indpb:
            individual[i] = rand_attribute()
            # (individual[i])[1] = random_pallet()

            # points = (individual[i])[0]
            # Xs = [j[0] for j in points]
            # Ys = [j[1] for j in points]
            # Ox = round((sum(Xs)) / 3)
            # Oy = round((sum(Ys)) / 3)
            # src_color = list(source[Ox, Oy])
            # (individual[i])[1] = [c / 255 for c in src_color]
            # else:
            # Find the centroid
            # print("\nSTART",individual[i])

            # print(individual[i], "END\n")
    return individual,


def initialize_deap():
    toolbox = base.Toolbox()

    toolbox.register("mate", cross_two)
    toolbox.register("mutate", mutate, indpb=ATTR_MUTATION)
    toolbox.register("attr_bool", rand_attribute)
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bool, NUMBER_OF_FIGURES)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    return toolbox


def do_the_thing(ind):
    toolbox = initialize_deap()
    pop = toolbox.population(n=POPULATION)

    # if __name__ == "__main__":

    # pool = multiprocessing.Pool()
    # toolbox.register("map", pool.map)

    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    # stats.register("avg", np.mean)
    # stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)

    # pop, log = \
    do_the_darvin_thing(pop, toolbox, 0.5, MUTATION, GENERATIONS, err, stats, hof)
    #

    path = f'part{ind}.png'
    show_ind(hof[0], path)
    remove_grid(path)


############################################  START  #################################################################
# Some deap setup
creator.create("FitnessMix", base.Fitness, weights=(-1, -1))
creator.create("Individual", list, fitness=creator.FitnessMix)

img = cv2.imread(IMAGE_TO_WORK_ON)

# Add alpha layer with OpenCV
bgra = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)

# Set alpha layer semi-transparent with Numpy indexing, B=0, G=1, R=2, A=3
bgra[..., 3] = 255

# Save result and rotate the picture so that there are no problems
cv2.imwrite('KOSTIL.png', bgra)
pic = Image.open(f"KOSTIL.png")
pic = ImageOps.mirror(pic)
pic = pic.rotate(90)
im = np.asarray(pic)

# Split the picture into sections
M = im.shape[0] // 4
N = im.shape[1] // 4
tiles = [im[x:x + M, y:y + N] for x in range(0, im.shape[0], M) for y in range(0, im.shape[1], N)]

for i in range(0, len(tiles)):
    print(f"\n\nWORKING ON PART {i + 1}\n\n")
    source = tiles[i]
    HEIGHT, WIDTH = source.shape[0], source.shape[1]
    do_the_thing(i)

# Put parts in the same list
parts = []
for i in range(0, len(tiles)):
    pic = Image.open(f"part{i}.png")
    im = np.asarray(pic)
    parts.append(im)

# Create a big picture from parts
columns = []
i = 0
while (i < len(tiles) - 1):
    if i % 2 == 0:
        vert1 = np.concatenate((parts[i], parts[i + 1]), axis=0)
        vert2 = np.concatenate((parts[i + 2], parts[i + 3]), axis=0)
        vert = np.concatenate((vert1, vert2), axis=0)
        columns.append(vert)
    i += 4
complete = columns[0]
for i in range(1, len(columns)):
    complete = np.concatenate((complete, columns[i]), axis=1)

image = Image.fromarray(complete)
image.save('result.png')
image.show()

for i in range(0, 16):
    os.remove(f'part{i}.png')
os.remove(f'KOSTIL.png')

############################################  END  #################################################################


# img = cv2.imread('result.png')
# # Add alpha layer with OpenCV
# bgra = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
# # Set alpha layer semi-transparent with Numpy indexing, B=0, G=1, R=2, A=3
# bgra[...,3] = 255
# # Save result
# cv2.imwrite('KOSTIL.png',bgra)
# pic = Image.open(f"KOSTIL.png")
# im = np.asarray(pic)
# source = im
# HEIGHT, WIDTH = source.shape[0], source.shape[1]
# NUMBER_OF_FIGURES = HEIGHT*HEIGHT*2
# do_the_thing(20)
#
#
# img = cv2.imread('sas/part20.png')
# # Add alpha layer with OpenCV
# bgra = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
# # Set alpha layer semi-transparent with Numpy indexing, B=0, G=1, R=2, A=3
# bgra[...,3] = 255
# # Save result
# cv2.imwrite('KOSTIL.png',bgra)
# pic = Image.open(f"KOSTIL.png")
# im = np.asarray(pic)
# source = im
# HEIGHT, WIDTH = source.shape[0], source.shape[1]
# NUMBER_OF_FIGURES = HEIGHT*HEIGHT*2
# do_the_thing(20)
