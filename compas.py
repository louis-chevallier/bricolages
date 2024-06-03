import sympy
from spb import plot, plot_geometry, MB, BB, geometry, graphics, surface, PB
import torch
import numpy as no
from utillc import *
import sympy.abc as X
from sympy import symbols, solve, cos, sin, lambdify, exp, pi
import os, sys
import random
from sympy import symbols, sqrt
import spb
from sympy.plotting import plot, plot_implicit

    

"""
voir mon projet compas ( user = louis.chevallier@googlemail.com )
https://www.geogebra.org/calculator

A, B, J = Point
O = Point2D(100, A.y)
C' dans le cercle(A, a)
angle(Line(O, A), Line(A, C)) = aa
F dans le cercle(B, b)
F dans le cercle(C, o)
K dans cercle(C', d)
K dans le cercle(J, n)
N = JK * e
H = C'F * k
O dans cercle(N, l)
O dans cercel(H, m)

longueurs des joints
a = 5.5
b = 4.5
d = 2.3
n = 1.6
(J, N) = 8.2
(C', H) = 8.1
l = 6
m = 4.3


"""


    
        
l_symbols = list(range(10000))
a, r, ra, rb = symbols("a r ra rb")

def S(n="") :
    s = symbols("%s_%d" % (n, l_symbols.pop(0)))
    return s

def Pf(n="") :
    if len(n.split(" ")) > 1 :
        return [ Pf(e) for e in n.split(" ")]
    else :
        x = symbols("x%s_%d" % (n, l_symbols.pop(0)))
        y = symbols("y%s_%d" % (n, l_symbols.pop(0)))
        return Point2D(x, y)

def intersection(_c1, _c2) :
    f, h = _c1.radius, _c2.radius
    xa, ya = _c1.center.x, _c1.center.y
    xc, yc = _c2.center.x, _c2.center.y

    return [Point2D(-(f**2 - h**2 - xa**2 + xc**2 - ya**2 + yc**2 + (2*ya - 2*yc)*(-sqrt(-(f**2 - 2*f*h + h**2 - xa**2 + 2*xa*xc - xc**2 - ya**2 + 2*ya*yc - yc**2)*(f**2 + 2*f*h + h**2 - xa**2 + 2*xa*xc - xc**2 - ya**2 + 2*ya*yc - yc**2))*(xa - xc)/(2*(xa**2 - 2*xa*xc + xc**2 + ya**2 - 2*ya*yc + yc**2)) - (f**2*ya - f**2*yc - h**2*ya + h**2*yc - xa**2*ya - xa**2*yc + 2*xa*xc*ya + 2*xa*xc*yc - xc**2*ya - xc**2*yc - ya**3 + ya**2*yc + ya*yc**2 - yc**3)/(2*(xa**2 - 2*xa*xc + xc**2 + ya**2 - 2*ya*yc + yc**2))))/(2*(xa - xc)), -sqrt(-(f**2 - 2*f*h + h**2 - xa**2 + 2*xa*xc - xc**2 - ya**2 + 2*ya*yc - yc**2)*(f**2 + 2*f*h + h**2 - xa**2 + 2*xa*xc - xc**2 - ya**2 + 2*ya*yc - yc**2))*(xa - xc)/(2*(xa**2 - 2*xa*xc + xc**2 + ya**2 - 2*ya*yc + yc**2)) - (f**2*ya - f**2*yc - h**2*ya + h**2*yc - xa**2*ya - xa**2*yc + 2*xa*xc*ya + 2*xa*xc*yc - xc**2*ya - xc**2*yc - ya**3 + ya**2*yc + ya*yc**2 - yc**3)/(2*(xa**2 - 2*xa*xc + xc**2 + ya**2 - 2*ya*yc + yc**2))),
        Point2D(-(f**2 - h**2 - xa**2 + xc**2 - ya**2 + yc**2 + (2*ya - 2*yc)*(sqrt(-(f**2 - 2*f*h + h**2 - xa**2 + 2*xa*xc - xc**2 - ya**2 + 2*ya*yc - yc**2)*(f**2 + 2*f*h + h**2 - xa**2 + 2*xa*xc - xc**2 - ya**2 + 2*ya*yc - yc**2))*(xa - xc)/(2*(xa**2 - 2*xa*xc + xc**2 + ya**2 - 2*ya*yc + yc**2)) - (f**2*ya - f**2*yc - h**2*ya + h**2*yc - xa**2*ya - xa**2*yc + 2*xa*xc*ya + 2*xa*xc*yc - xc**2*ya - xc**2*yc - ya**3 + ya**2*yc + ya*yc**2 - yc**3)/(2*(xa**2 - 2*xa*xc + xc**2 + ya**2 - 2*ya*yc + yc**2))))/(2*(xa - xc)), sqrt(-(f**2 - 2*f*h + h**2 - xa**2 + 2*xa*xc - xc**2 - ya**2 + 2*ya*yc - yc**2)*(f**2 + 2*f*h + h**2 - xa**2 + 2*xa*xc - xc**2 - ya**2 + 2*ya*yc - yc**2))*(xa - xc)/(2*(xa**2 - 2*xa*xc + xc**2 + ya**2 - 2*ya*yc + yc**2)) - (f**2*ya - f**2*yc - h**2*ya + h**2*yc - xa**2*ya - xa**2*yc + 2*xa*xc*ya + 2*xa*xc*yc - xc**2*ya - xc**2*yc - ya**3 + ya**2*yc + ya*yc**2 - yc**3)/(2*(xa**2 - 2*xa*xc + xc**2 + ya**2 - 2*ya*yc + yc**2)))]


def distance(A, B) :
    xa, ya = A.x, A.y
    xb, yb = B.x, B.y
    return sqrt( (xa - xb) **2 + (yb - ya) **2)


# les classes suivantes pour remplacer celles de sympy
#  car elles appellent automatiquement symplify qui prend un temps infini

class Point2D :
    def __init__(self, x, y) :
        self.x, self.y = x, y
    def __add__(self, p) :
        return Point2D(self.x + p.x, self.y + p.y)
    def __sub__(self, p) :
        return Point2D(self.x - p.x, self.y - p.y)

    # scalar
    def __truediv__(self, f) :
        return Point2D(self.x / f, self.y / f)
    
    def __mul__(self, f) :
        return Point2D(self.x * f, self.y * f)
    
    def rotate(self, angle, A) :
        xc, yc = self.x, self.y
        xa, ya = A.x, A.y
        Cp=Point2D(xa + (-xa + xc)*cos(angle) + (ya - yc)*sin(angle), ya + (-xa + xc)*sin(angle) + (-ya + yc)*cos(angle))
        return Cp
    def __str__(self) :
        return "Point2D(" + str(self.x) + ", " + str(self.y) + ")"

    def subs(self, su) :
        return sympy.Point2D(self.x.subs(su),
                             self.y.subs(su))
                             
class Circle :
    def __init__(self, C, radius_) :
        self.center, self.radius = C, radius_
    def __str__(self) :
        return "Circle(" + str(self.center) + ", " + str(self.radius) + ")"
    def subs(self, su) :
        return sympy.Circle(self.center.subs(su),
                            self.radius.subs(su))
class Line :
    def __init__(self, A, B) :
        self.A, self.B = A, B
    def __str__(self) :
        return "Line(" + str(self.A) + ", " + str(self.B) + ")"
    def subs(self, su) :
        return sympy.Line(self.A.subs(su),
                          self.B.subs(su))


    
def yyy() :

    x, y, z = symbols("x, y, z")
    r = sqrt(x**2 + y**2)
    d = symbols('d')
    expr = 10 * cos(r) * exp(-r * d)
    graphics(
        surface(
            expr, (x, -10, 10), (y, -10, 10), label="z-range",
            params={d: (0.15, 0, 1)}, n=51, use_cm=True,
            wireframe = True, wf_n1=15, wf_n2=15,
            wf_rendering_kw={"line_color": "#003428", "line_width": 0.75}
        ),
        title = "My Title",
        xlabel = "x axis",
        ylabel = "y axis",
        zlabel = "z axis",
        backend = PB
    )
    
    
def xxx() :

    
    x, a, b, c = symbols("x, a, b, c")

    




    
    xa, ya, xc, yc, xq, yq, f, g, h, o, b , angle = symbols("xa ya, xc yc xq yq f g h o b angle", real=True)
    xb, yb = symbols("xb yb")
    f, g, h = symbols("f g h", Positive=True, real=True)
    A, C, Q, B = Point2D(xa, ya), Point2D(xc, yc), Point2D(xq, yq), Point2D(xb, yb)

    EKOX(C.rotate(angle, A))
    
    

    b1 = symbols("b1", real=True)
    
    #angle = -pi/6

    
    if False :
        A, C = sympy.Point2D(xa, ya), sympy.Point2D(xc, yc),
        Cp = C.rotate(angle, A)
        EKOX(Cp)


    
    Ao = A + Point2D(f, 0)
    EKOX(Ao)
    Cp = Ao.rotate(angle, A)

    d = Circle(Cp, o)

    c = Circle(B, b)

    F = intersection(d, c)
    
    ca = Circle(A, f)
    EKOX(ca)

    F0 = F[0]
    EKOX(F0)

    dfcp = distance(F0, Cp)
    EKOX(dfcp)
    EKOX(F0)
    EKOX(Cp)

    H = (F0 - Cp) / dfcp * b1 + Cp

    
    #pac = ca.intersection(cc)[0] # delivers the answer after 30 sec
    #EKOX(pac)
    #Xpaq = intersection(ca, cq)[0]
    #Xpac = intersection(ca, cc)[0]
    #EKOX(Xpac)
    #paq = ca.intersection(cq)[0]
    EKO()
    sss = [(angle, -pi/15),
           (xa, -16), (ya, 11.2),
           (xb, -15.9), (yb, 6.4),
           (o, 5.3), (b, 4.5), 
           (f, 6.9),
           (b1, 8.1),
           (xc, 7), (yc, 3), (h, 2) ]
    sss += [ (xq, 4), (yq, 3), (g, 4)]
    #cc.equation().subs(sss)
    #EKOX(pac.subs(sss))
    EKO()
    #P = cr.intersection(ca)
    EKO()
    #B = cr.intersection(cc)
    EKO()
    
    #EKOX(ca.subs(sss))
    #graphics(geometry(ca.subs(sss), fill=False, label='ca'))

    EKO()

    
    graphics(geometry(ca.subs(sss), fill=False, label='ca'),
             #geometry(cq.subs(sss), fill=False, label='cq'),
             geometry(A.subs(sss), label='A'),
             geometry(Cp.subs(sss), label='Cp'),
             geometry(Line(A, Cp).subs(sss), label=""),
             geometry(C.subs(sss), label='C'),
             geometry(B.subs(sss), label='B'),
             #geometry(P[1].subs(sss), label='P1'),
             #geometry(Q[0].subs(sss), label='Q0'),
             geometry(Line(Cp, F0).subs(sss), label=""),
             geometry(Line(B, F0).subs(sss), label=""),
             geometry(Ao.subs(sss), label='Ao'),

             geometry(H.subs(sss), label='H'),             
             

             #geometry(Xpac.subs(sss), label='Xpac')
             )
    EKO()


def fff() :
    eqq = True
    if eqq :
        EKOX(Pf("A B J C F G K N H O"))
        a, b, d, n, e, k, l, m, o, aa = symbols("a b d n e k l m o aa")    
        A, B, J, C, F, G, K, N, H, O = Pf("A B J C F G K N H O")

        C = Point2D(0, aa)

        N = J + (K - J) * e
        H = C + (F - C) * k
        O = O = Point2D(100, A.y)

        eq = [ Line(O, A).angle_between(Line(A, C)) - aa,           C.distance(A) - a]
        eq = [ C.distance(A) - a]
        sol = solve(eq, [ c for p in [C] for c in p.coordinates])    
        EKOX(sol)

        eq = [ Line(O, A).angle_between(Line(A, C)) - aa,
               C.distance(A) - a,
               F.distance(B) - b,
               F.distance(C) - o,
              ]

        eq = [ 
               C.distance(A) - a,
               F.distance(B) - b,
               F.distance(C) - o,
              ]


        sol = solve(eq, [ c for p in [C, F] for c in p.coordinates])    
        EKOX(sol)

        sol = solve(eq, [ c for p in [C, F] for c in p.coordinates])    
        EKOX(sol)

        eq = [ Line(O, A).angle_between(Line(A, C)) - aa,
               C.distance(A) - a,
               F.distance(B) - b,
               K.distance(C) - d,
               K.distance(J) - n,
               O.distance(N) - l,
               K.distance(J) - m]
        EKO()
        sol = solve(eq, [ c for p in (H, O, N, K, F, C) for c in p.coordinates])
        EKOX(sol)


    EKO()
    experiment = True

    if experiment :
        """
    a partir d'une equation, y = x*x 
        resoudre l'équation en fct de x : x = sqrt(y) 
        transformer la solution en code *torch* !
        permet de construire le graphe de calcul en torch pour faire la GD
        """
        x, y, z, u, v  = symbols("x y z u v")

        e1, e2 = y**2 + z, z - y
        eq = [ e1 - u, e2 - v]

        ss = [(y,2),(z,3)]
        EKOX((e1.subs(ss), e2.subs(ss)))


        s = solve(eq, [y, z])
        EKOX(s)
        f1 = lambdify([ u, v], s, "numpy")
        os.makedirs("gen", exist_ok = True)
        with open("gen/f.py", "w") as fd :
            ss = """
    from torch import *
    def f(u, v) :
        return %s """
            fd.write(ss % str(s))
            imp = __import__("f")
            EKOX(imp.f(torch.tensor(7.) ,torch.tensor(1.)))
            EKOX(f1(7., 1.))


        #sys.exit(0)





    A = Pf("A")
    B = Pf("B")
    xp = A.x + ra * cos(a)
    yp = A.y + ra * sin(a)
    P, Q = Point2D(xp, yp), Pf("Q")

    """
    étant donnés 2 points A et B
    P tourne autour de A ( rayon ra)
    l'angle Ax, AP = a
    Q tourne autour de B ( rayon rb)
    dist(P,Q) = r

    trouver les positions de P et Q
    """

    eq = [ P.distance(Q) - r, P.distance(A) - ra, Q.distance(B) - rb]
    EKO()
    sol = solve(eq, list(P.coordinates) + list(Q.coordinates))
    EKOX(sol)

    """
    listes :
    bras, fix point, mobile point, scalars (distances) 

    build system : calls to stratX()
    distances are params
    select 2 points mobiles, pm1, pm2
    objective = pm1 pm2 location at t=0 and t=end
    optimize distances wrt objectives

    """

    alpha = S("alpha")

    fixed = [ Pf("A"), Pf("B")]
    mobiles = []
    scalars = []
    bras = []
    eee = []

    def choose(l) :
        i = random.randint(0, len(l)-1)
        return l[i] 

    def brasf(P1, P2, d) :
        """
        rend l'eq correspondant a un bras de longueur d entre P1 et P2
        """
        eq = [ P1.distance(P2) - d]
        return { "eq" : eq, "p" : [P1, P2] }

    """
    mobiles : liste des points mobiles
    fixed : liste des points fixes
    scalars : liste des valeurs scalaires ( variables a optimiser )
    bras : liste des bras 
    """

    """
    les strats ajoutent des points fixes, des points mobiles, des bras 
    et des scalaires
    Xf un pt fixe
    Xm un pt mobiles
    B9 un bras


    depart : Af - un point fixe => a, d, Pm, bras(Af, Pm, d) / d(Af, Pm) == d, 
    initialisation : 
    mobiles = [Pm0], fixes = [ Af0 ], bras = [ bras(Af0, Pm0, d) ], scalars = [d]
    a = angle(Af0_x, Af0_Pm)


    Strat :
        init() :
           solve(eq, p)
           f = torch(eq)
        exec() => return f(angle)

    1/ => Af, nouveau point fixe

    2/ P1, P2 => d1, d2, Qm, bras(Qm, P1, d1), bras(Qm, P2, d2) 

    3/ bras(P, Q) => d, R, bras(P, R, d1), bras(P, Q), bras(Q, R) / R = P + v(P,Q) * u 


    construit strats
    optimizer = optim(scalars)

    construction du graphe:
    start = next_strat()

    Ta_0, Tb_0 : position initiale des extremités du segment
    Ta_90, Tb_90 : position finale des extremités du segment

    loop :
        for s in strats :
           l0.append(exec(s, angle = 0))
           l90.append(exec(s, angle = 90))

        p0_1, p0_2 = l0[-2:-1]
        p90_1, p90_2 = l90[-2:-1]
        loss = dist(p0_1, Ta_0) + dist(p0_2, Tb_0) 
        loss += dist(p90_1, Ta_90) + dist(p00_2, Tb_90) 

        loss.back()
        step()

    """

    def strat1() :
        """
        called once
        use alpha = angle of b1
        pick 2 fix points pf1, pf2
        define 2 mobile points pm1, pm2
        define 3 distances d1, d2, d3
        b1 = bras(pf1, pm1, d1)
        b2 = bras(pf2, pm2, d2)
        b3 = bras(pm1, pm2, d3)
        yield :
               pm1, pm2 : mobile
               d1, d2, d3 : scalaires
               b1, b2, b3 : bras
        """

        Pf1, Pf2 = choose(fixed), choose(fixed)
        Pm2 = Pf("p2")
        d1, d2, d3 = S("d1"), S("d2"), S("d3")


        xp = Pf1.x + d1 * cos(a)
        yp = Pf1.y + d1 * sin(a)
        Pm1 = Point2D(xp, yp)

        b1 = brasf(Pf1, Pm1, d1)
        b2 = brasf(Pf2, Pm2, d2)
        b3 = brasf(Pm1, Pm2, d3)

        return { "mobiles" : [ Pm1], "fixed" : [Pf1, Pf2], "scalars" : [ d1, d2, d3], "bras" : [ b1, b2, b3]}

    def strat4() :
        """
        pick fix point pf, mobile point pm1
        b1 = bras(pf, pm, d1)
        b2 = bras(pm1, pm, d2)
        yield :
             pm, d1, d2, b1, b2
        """
        Pf1 = choose(fixed)
        Pm1 = choose(mobiles)
        d1, d2 = S("d1"), S("d2")
        Pm = Pf("pm")
        b1 = brasf(Pf1, Pm, d1)
        b2 = brasf(Pm1, Pm, d2)

        return { "mobiles" : [ Pm ], "fixed" : [], "scalars" : [ d1, d2], "bras" : [ b1, b2]}


    def add(strt) :
        """
        ajoute des éléments décrits par str
        """
        global fixed, mobiles, scalars, bras
        d = strt()
        fixed += d["fixed"]
        mobiles += d["mobiles"]
        scalars += d["scalars"]
        bras += d["bras"]

        eqs = [ ee for e in d["bras"] for ee in e["eq"] ]
        EKOX(eqs)
        EKOX(d["mobiles"])
        s = solve(eqs, [ c for p in d["mobiles"] for c in p.coordinates] )
        EKOX(s)
        eee.append(s)

    add(strat1)
    add(strat4)

    EKOX(bras)
    EKOX(mobiles)
    eqs = [ ee for e in bras for ee in e["eq"] ]
    EKOX(eqs)
    EKOX(mobiles)
    #s = solve(eqs, [ c for p in mobiles for c in p.coordinates] )
    #EKOX(s)




    def strat2() :
        """
        pick 2 fixed points
        define 2 params
        pf = u pf1 + v pf2
        yield :
               pf1, pf2
        """

    def strat3() :
        """
        pick a bras b = bras(A, B, p)
        define 2 params
        yield : 
               pm = u A + v B
        """




if __name__ == "__main__" :
    xxx()
