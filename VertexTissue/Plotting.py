from matplotlib import rc
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

legend_size=14
tick_size=16
label_size=18
usetex=True
mpl.rcParams["text.latex.preamble"]=r'\usepackage{amsmath}'
mpl.rcParams["text.usetex"]=True
mpl.rcParams['mathtext.fontset'] = 'cm'

mpl.rcParams['xtick.labelsize'] = tick_size
mpl.rcParams['ytick.labelsize'] = tick_size
mpl.rcParams['axes.labelsize'] = label_size
mpl.rcParams['legend.fontsize'] = legend_size
rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
rc('text', usetex=True)

# tick_style={'labelsize':18}
# label_style={'usetex':True,'fontsize':20};
# title_style={'usetex':True,'fontsize':22};
legend_style={'fontsize':12, 'loc':'upper left'};


def pcolor(X, Y, C, shading='nearest', cmap=None, tight=True, **kw):
    plt.pcolormesh(X, Y, C, shading=shading, cmap=cmap, **kw)
    if tight:
        plt.xlim(X[0], X[-1])
        plt.ylim(Y[0], Y[-1])

def hatched_contour(X, Y, Z, hatches = ['\\\\\\'], color='#ff000080', upscale=1, levels = None, linewidth=0.5, alpha=0.0, hatch_alpha=0.05):
    # if upscale>1:
    XY, Z = upsample(X, Y, Z, fold=upscale, midpoint=True)

    color = mpl.colors.to_rgba(color)
    if  hatch_alpha is not None:
        color = (*color[:3], hatch_alpha)
    color = mpl.colors.to_hex(color, keep_alpha=True)
    plt.rcParams['hatch.color']=color
    plt.rcParams['hatch.linewidth']=linewidth


    plt.contourf(*XY, Z, levels=levels, hatches=hatches, alpha=alpha)
    plt.contour(*XY, Z, levels =levels, colors=mpl.colors.to_hex(color, keep_alpha=False), linewidths=linewidth)

def contour(X, Y, Z, color='r', upscale=1, levels = None, linewidth=1):
    # if upscale>1:
    XY, Z = upsample(X, Y, Z, fold=upscale, midpoint=True)

    plt.contour(*XY, Z, levels =levels, colors=color, linewidths=linewidth)

def upsample(*args, fold=10, midpoint=True):
    *coords, vals = args 
    ndim = len(vals.shape)

    for i, x in enumerate(coords):
        Nx=len(x)
        coords
        x0=np.array(x)
        if midpoint:
            x = np.array([x0[0], *((x0[1:]+x0[:-1])/2), x0[-1]])
            X = np.array([*np.linspace(0, 1, fold), *np.linspace(1,Nx-1,fold*(Nx-2)+2)[1:-1], *np.linspace(Nx-1,Nx,fold) ])
        else:
            X=np.linspace(0, Nx-1, fold*Nx)
        coords[i] = np.interp(X, range(len(x)), x)


    Z = vals
    
    for d in range(ndim):
        Z = np.repeat(Z, fold, axis=d)


    return coords, Z

def add_color_bar_to_right(ax, **kw):
    pos=ax.get_position().bounds
    fig=ax.get_figure()
    cb_ax=fig.add_axes([pos[0]+pos[2]+pos[3]/50, pos[1], pos[3]/50, pos[3]])
    plt.colorbar(cax=cb_ax,ax=ax, **kw)
    return cb_ax