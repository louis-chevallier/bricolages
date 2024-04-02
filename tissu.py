"""
un code pour placer des coupons de tissus sur un grand morceau de tissu
"""
# pip install rectpack
import random
from rectpack import float2dec, newPacker
from utillc import *
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle
import matplotlib.colors as mcolors
from rectpack import newPacker
from guizero import App, Text, PushButton, Drawing


random.seed(5)


# taille du grand morceau de tissu - en cm
tissu = (155*2, 140*2)

# avec cette taille ça passe juste juste ...
#tissu = (130*2, 136*2)
#tissu = (100*2, 236*2)

# les 3 blocs de mousse, w x h
blocs = [ (63, 96), (90, 45), (85+90-96, 63)]

# l'epaisseur commune des blocs
epaisseur = 14

def faces(bloc) :
    """
    donne les 6 faces d'un bloc
    un bloc == w, h
    chaque bloc a 6 faces
    j'ajoute un ourlet
    """
    ourlet = 5
    p1 = lambda x : x + ourlet*2
    w, h, e = map(p1, list(bloc) + [ epaisseur])
    r = [(w, h)] * 2 + [(e, w)] * 2 + [ (e, h)] * 2
    return r

rectangles = [ f for b in blocs for f in faces(b) ]
bins = [ tissu]

EKOX(rectangles)
EKOX(bins)


def compute(bins) :
    packer = newPacker() #pack_algo=Skyline)

    # Add the rectangles to packing queue
    for r in rectangles:
            packer.add_rect(*r)

    # Add the bins where the rectangles will be placed
    for b in bins:
            packer.add_bin(*b)

    # Start packing
    packer.pack()

    # Obtain number of bins used for packing
    nbins = len(packer)
    # Index first bin
    abin = packer[0]
    # Bin dimmensions (bins can be reordered during packing)
    width, height = abin.width, abin.height
    # Number of rectangles packed into first bin
    nrect = len(packer[0])
    if (nrect == len(rectangles)) :
        return packer
    else :
        return None



packer = compute(bins)

if True :
    fig, ax = plt.subplots(1)

    ax.set_xlim([0, bins[0][0]])
    ax.set_ylim([0, bins[0][1]])

    colors = [ color for  (_, color) in mcolors.CSS4_COLORS.items()]
    random.shuffle(colors)
    for i, rect in enumerate(packer[0]) :
        EKOX(rect)
        boxes = [Rectangle((rect.x, rect.y), rect.width, rect.height)]
        pc = PatchCollection(boxes, facecolor=colors[i % len(colors)], alpha=0.5,  edgecolor='b')
        ax.add_collection(pc)

    plt.show()

if True :

    app = App(title="guizero")

    #intro = Text(app, text="Have a go with guizero and see what you can create.")
    #ok = PushButton(app, text="Ok")
    drawing = Drawing(app, width='fill', height='fill')

    sw, sh = 0, 0

    def repeat() :
        global sw, sh
        w,h = app.width, app.height
        if (w,h) != (sw, sh) :
            random.seed(5)        
            #w, h = drawing.width, drawing.height
            sw, sh = w, h
            drawing.clear()
            p =compute([(w, h)])
            def r() : return random.randint(0, 255)

            if p is not None :
                 for i, rect in enumerate(p[0]) :
                     drawing.rectangle(rect.x, rect.y, rect.x + rect.width, rect.y + rect.height, color=(r(), r(), r()))



    app.repeat(10, repeat)



    app.display()
