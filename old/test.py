import numpy as np
from matplotlib import pyplot as plt
from matplotlib import animation

(MIN_LAT, MAX_LAT) = (20, 40)
(MIN_LONG, MAX_LONG) = (280, 320)

X, Y = np.meshgrid(
        np.linspace(MIN_LAT, MAX_LAT, 21),
        np.linspace(MIN_LONG, MAX_LONG, 21))
U = np.cos(X)
V = np.sin(Y)

fig, ax = plt.subplots(1,1)
Q = ax.quiver(X, Y, U, V, pivot='mid', color='r', units='inches')

def update_quiver(num, Q, X, Y):
    """updates the horizontal and vertical vector components by a
    fixed increment on each frame
    """

    U = np.cos(X + num*0.1)
    V = np.sin(Y + num*0.1)

    Q.set_UVC(U,V)

    return Q,

# you need to set blit=False, or the first set of arrows never gets
# cleared on subsequent frames
anim = animation.FuncAnimation(fig, update_quiver, fargs=(Q, X, Y),
                               interval=50, blit=False)
fig.tight_layout()
plt.show()
