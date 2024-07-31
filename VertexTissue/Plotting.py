from matplotlib import rc
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from scipy.interpolate import LinearNDInterpolator
from copy import deepcopy

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


def panelled_row( data, plot_func=plt.plot, 
                 panel_dims=(3,3.5), window_title=None, panel_titles=None, ylabel=None, xlabel=None,
                 colorbar=False, colorbar_side='right', colorbar_label=None, ax_callback=None, 
                 data_transform=None, colorbar_tick_transform=None):
    N=len(data)
    fig, axs = plt.subplots(1,N)
    fig.set_size_inches((panel_dims[0]+0.25)*N, panel_dims[1])
   
    axs=axs.ravel()
    plt.get_current_fig_manager().set_window_title(window_title)



    plt.sca(axs[0])
    plt.ylabel(ylabel)

    if panel_titles is None:
        panel_titles=[None,]*N

    if data_transform is None:
        data_transform = lambda d: d

    for d, ax, title in zip(data, axs, panel_titles):
        plt.sca(ax)
        plot_func(data_transform(d))
        ax.set_title(title, usetex=True, fontsize=16) 
        plt.xlabel(xlabel)

    plt.tight_layout()
    
    if ax_callback:
        assert callable(ax_callback)
        for d, ax in zip(data, axs):
            ax_callback(ax, d)
        
    if colorbar:
        cb_ax=add_colorbar_to_side( axs[-1], side=colorbar_side, label=colorbar_label)
        cb_ax.yaxis.label.set_size(tick_size)

        if colorbar_tick_transform:
            assert callable(colorbar_tick_transform)
            if colorbar_side=='right':
                ticks = cb_ax.get_yticks()
                lbls = [ plt.Text(1,t,f'$\\mathdefault{{{colorbar_tick_transform(t)}}}$') for t in ticks]
            else:
                ticks = cb_ax.get_yticks()
                lbls = [ plt.Text(t,0,f'$\\mathdefault{{{colorbar_tick_transform(t)}}}$') for t in ticks]
            
            cb_ax.set_yticks(ticks, labels=lbls)

        return cb_ax
        
            
    


def pcolor(X, Y, C, shading='nearest', cmap=None, tight=True, upscale=1, sampling='nearest', **kw):
    if upscale>1:
        XY, C = upsample(X, Y, C, fold=upscale, midpoint=True, sampling=sampling)
        X,Y = XY

    out = plt.pcolormesh(X, Y, C, shading=shading, cmap=cmap, **kw)

    if tight:
        plt.xlim(X[0], X[-1])
        plt.ylim(Y[0], Y[-1])

    return out

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

def contour(X, Y, Z, color='r', upscale=1, sampling='nearest', levels = None, ax=None, linewidth=1):
    # if upscale>1:
    XY, Z = upsample(X, Y, Z, fold=upscale, midpoint=True, sampling=sampling)

    if ax is None:
        ax=plt.gca()

    return ax.contour(*XY, Z, levels =levels, colors=color, linewidths=linewidth)

def upsample(*args, fold=10, midpoint=True, sampling='nearest'):
    *coords, vals = args 
    ndim = len(vals.shape)

    coords_orig=deepcopy(coords)
    for i, x in enumerate(coords):
        Nx=len(x)
        x0=np.array(x)
        if midpoint:
            x = np.array([x0[0], *((x0[1:]+x0[:-1])/2), x0[-1]])
            x_upsampled = np.array([*np.linspace(0, 1, fold), *np.linspace(1,Nx-1,fold*(Nx-2)+2)[1:-1], *np.linspace(Nx-1,Nx,fold) ])
        else:
            x_upsampled=np.linspace(0, Nx-1, fold*Nx)
            
        coords[i] = np.interp(x_upsampled, range(len(x)), x)


    
    match sampling:
        case 'nearest':
            Z = vals
            for d in range(ndim):
                Z = np.repeat(Z, fold, axis=d)
        case 'linear':
                interpolator = LinearNDInterpolator(np.array([x.flatten() for x in np.meshgrid(*coords_orig)]).T, np.array(vals).flatten())
                Z = interpolator(*np.meshgrid(*coords))

    return coords, Z

spectral = plt.cm.nipy_spectral
def spectral_rainbow(N):
    return spectral(np.linspace(0,.85,N))

def make_lines_rainbow(ax, reversed=False):
    """ Make all the lines in a plot follow a rainbow color scheme 
        using the `matplotlib.pyplot.cm.nipy_spectral` colormap """
    
    N=len(ax.lines)
    colors = spectral_rainbow(N)

    for i,l in enumerate(ax.lines):
            l.set_color(colors[i if not reversed else -(i+1)])


def add_colorbar_to_side(*args, ax=None, side='right', **kw):
    if not args:
        ax=plt.gca()
    else:
        match len(args):
            case 1:
                ax=args[0]
            case 2:
                img=args[0]
                ax=args[1]
        
    pos=ax.get_position().bounds
    fig=ax.get_figure()
    

    match side:
        case 'right':
            offset=pos[3]/50
            thickness=offset
            cb_ax=fig.add_axes([pos[0]+pos[2]+offset, pos[1], thickness, pos[3]])
        case 'bottom':
            offset=pos[3]/10
            thickness=offset/2
            pos_label = ax.figure.transFigure.inverted().transform_bbox(ax.get_tightbbox()).bounds
            cb_ax=fig.add_axes([pos[0], pos_label[1]-offset, pos[2], thickness])
            kw['orientation']='horizontal'


    match len(args):
        case 1:
            plt.colorbar(*args[1:], cax=cb_ax, ax=ax, **kw)
        case 2:
           print('using Figure.colorbar')
           fig.colorbar(img, cax=cb_ax, ax=ax, **kw)

    return cb_ax