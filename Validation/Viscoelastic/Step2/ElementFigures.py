import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from VertexTissue.visco_funcs import fluid_element
import VertexTissue.Plotting

lens = np.linspace(0,2,500)
L0 = np.zeros(lens.shape)+1.0

# plt.ion()

from matplotlib.transforms import ScaledTranslation
# fig, axs = plt.subplots(1,3)

# axs=[ax for ax in axs.values()]

# plt.get_current_fig_manager().canvas.set_window_title('Middle')
# axs=axs.ravel()


# mpl.rcParams["text.latex.preamble"]=r'\usepackage{amsmath}'
# mpl.rcParams["text.usetex"]=True
# mpl.rcParams['mathtext.fontset'] = 'cm'
# tick_style={'usetex':True,'fontsize':18}
# label_style={'usetex':True,'fontsize':20};
# # title_style={'usetex':True,'fontsize':16};
# legend_style={'fontsize':12, 'loc':'upper left'};

ec=0.4
phi0=0.65
crumpler = fluid_element(phi0=phi0, ec=ec)
extender = fluid_element(phi0=phi0, ec=ec, contract=False, extend=True)
sym = fluid_element(phi0=phi0, ec=ec, extend=True)


crumpler = lens-1
crumpler[lens-1>-ec]=0
extender = lens-1
extender[lens-1<ec]=0
sym = lens-1
sym[np.logical_and(lens-1>-ec,lens-1<ec)]=0

for dLdt, lbls, title in zip((crumpler, extender, sym),
                                [((-ec,r'$-\varepsilon_{c}$'),),
                                ((ec,r'$\varepsilon_{c}$'),),
                                ((-ec,r'$-\varepsilon_{c}$'),(ec,r'$\varepsilon_{c}$'), )],
                                ('asymm_contraction','asymm_extension', 'symmetric')):
        fig=plt.figure()
        ax=fig.gca()
        if dLdt is crumpler:
                ylabel_style={}
        else:
                ylabel_style={'alpha':0}
        ylabel_style={}

        ax.set_ylabel(r'$\tau \dfrac{\dot{L}}{L}$', rotation = 0, verticalalignment='center', labelpad=16, **ylabel_style)
        fig.set_size_inches(14.0*0.8/3.0, 3)
        # dLdt = lens-1
        # dLdt[lens-1>-ec]=0
        plt.axhline(y=1.0, color='k', linestyle='-', linewidth=0.5)
        plt.axvline(x=0, color='k', linestyle='-', linewidth=0.5)     

        plt.plot(lens-1, lens-1, color='k', linestyle='--', linewidth=1, label='SLS')
        plt.plot(lens-1, dLdt, linewidth=2, label='Modified SLS')

        plt.xlim((-1,1))
        plt.ylim((-1,1))

        plt.draw()
        xticks=[-1.0, 0,  1.0]
        xlabels = [str(x) for x in xticks]
        print(xlabels[0])
        for lbl in lbls:
                x=lbl[0]
                lbl=lbl[1]
                if x==-ec or x==ec:
                        plt.axvline(x=x, color='k', linestyle=':', linewidth=0.2)
                if not np.any(xticks==x):
                        xticks.append(x)
                        xlabels.append(lbl)
                else:
                        xlabels[np.argwhere(xticks==x)[0,0]]=lbl
        ax.set_xticks(xticks)
        ax.set_xticklabels(xlabels)
        ax.set_yticks((-1,0,1))
        ax.set_yticklabels(("-1","0","1"))
        ax.set_xlabel(r'$\varepsilon$')

        # ax.set_title(title, **title_style)
        fig.set_layout_engine('constrained')
        plt.legend()
        plt.savefig(f'SLS_elements_scheme_{title}.pdf')
        # plt.savefig(f'SLS_elements_scheme_{title}.png',dpi=200)
        # plt.show()


plt.show(block=True)