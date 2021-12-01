#@title ## Basic imports

import sys

print(sys.version)
import importlib  # for reloading imports to source fun
# ctions
#from IPython.display import HTML, Math
import IPython.display
import os
import time

import itertools
import inspect
import math

import numpy as np
import pandas as pd
import scipy as sp
import matplotlib.pyplot as plt
import matplotlib.ticker
import collections

import warnings
warnings.simplefilter("default")

#from inverting import euler_inversion
import inverting
import plotting
import viscoporoelastic_model
from viscoporoelastic_model import CohenModel

#####

importlib.reload(viscoporoelastic_model)
from viscoporoelastic_model import CohenModel, getCohenModelModified

VPEs = [
        (CohenModel(), fr"$Cohen$"),
        #(getCohenModelModified(E1=3.5), fr"$Cohen, E_1=3.5$")
        ]
do_plot = True  #@param {type: "boolean"}
s_real = 1 #@param {type:"slider", min:-10, max:10, step:0.1}
(s, ds)=np.linspace(start=s_real-50j, stop=s_real+50j, num=100000, endpoint=True, retstep=True)
#t1 = np.array([.01,.1, 1.5])  # nondimensional
t1 = np.array([.01,.1, 1.0])  # nondimensional
#t1 = np.concatenate( (np.arange(0.01,0.1,0.01), np.arange(.1,2,.1)) )  # nondimensional



t0_tg = VPEs[0][0].t0_tg
tg = VPEs[0][0].tg


func = [vpe.laplace_value for vpe, label in VPEs]
inv_funcs_anal = [vpe.inverted_value for vpe, label in VPEs]


if do_plot:
    fig, axs = plt.subplots(nrows=5, ncols=2, figsize=(10,20),
                            subplot_kw={
                                #"sharex":"all"
                                })

    for ax in axs.flat[0:-2]:  # Don't do the final two, which will be merged
        ax.grid(which="major")  # set major grid lines
        ax.grid(which="minor", alpha=0.75, linestyle=":")  # set minor grid lines, but make them less visible
        ax.xaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator())
        ax.yaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator())
        ##ax.set_xlabel("$Im[s]=(s-Re[s])/i$, " + f"Re[s]={s_real}")
        #ax.set_xlabel(r"$\mathrm{Im\left[s\right]}$ ($\mathrm{Re}\left[s\right]="+f"{s_real}$)")
        ax.set_xlabel(r"$\mathrm{Im\left[s\right]}$ (where $s="+f"{s_real}"+r"+\mathrm{Im}\left[s\right]j$)")

    #plt.suptitle(f"$s={s_real}+"+r"\mathrm{Im}\left[s\right]j$")
    #fig.supxlabel(f"Component of complex output")  # requires matplotlib 3.4



plot_s = np.imag(s[...,None])
for vpe, label in VPEs:
    # The [...,None] index allows t1 to be either a single value (returns single value) or a matrix (returns matrix)
    ys = [
          vpe.laplace_value(s, dimensional=True),
          np.exp(s[...,None]*t1)/(2j*np.pi)]
    ys.append( (ys[0].T*ys[1].T).T )
    ys.append( np.cumsum(ys[-1], axis=0)*ds )
    # No dollar signs ($) here - added later
    row_names = [r"F(s)",
                 r"\mathrm{e}^{s \cdot t_1}/(2 \pi j)"
                 #"\frac{exp(s \cdot t1)}{2j\cdot \pi}"
                 ]
    row_names.append( row_names[0]+row_names[1] )
    row_names.append( r"\int_{"+f"{np.min(s):.0f}" +r"}^{s}" + row_names[-1] + r" \mathrm{d}s")
    #row_names.append( r"\int_{-\infty}^{s}" + row_names[-1] + " ds")



    if do_plot:
        #axs.flat[0+ind*2]
        gs = axs.flat[-2].get_gridspec()
        # remove the underlying axes
        for ax in axs.flat[-2:0]:
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_axis_off( )
            ax.remove()
            #ax.remove
        axbig = fig.add_subplot(gs[-1, -2:])
        axbig.grid(which="major")
        #axbig.plot( np.array([1,2]), np.array([3,4]) )
        #axbig.annotate('Big Axes \nGridSpec[1:, -1]', (0.1, 0.5),
        #            xycoords='axes fraction', va='center')
        axbig.plot(t1, ys[3][-1,:],"o", label="est $f(t)$")
        t_plot = np.linspace(0,2,num=100,endpoint=True)
        axbig.plot(t_plot, vpe.inverted_value(t=t_plot*vpe.tg), label="Analytic $f(t)$")
        axbig.plot(t_plot, inverting.euler_inversion(vpe.laplace_value, t_plot), "--", label="Numerical f(t) - Euler")
        axbig.plot(t_plot, inverting.talbot_inversion(vpe.laplace_value, t_plot), "--", label="Numerical f(t) - Talbot")


        axbig.set_xlabel("$t/t_g$")
        axbig.set_xlim([0, np.max(t_plot)])
        axbig.set_ylim([0, None])
        axbig.legend()
        axbig.set_xticks(np.arange(0,2.5,0.1))
        #axbig.set_xticks([1,2])

        for ind,y in enumerate(ys):
            axs.flat[0+ind*2].plot(plot_s, np.real(y))
            axs.flat[1+ind*2].plot(plot_s, np.imag(y))
            axs.flat[0+ind*2].set_ylabel(r"$\mathrm{Re} \left[" +f"{row_names[ind]}"+r"\right]$")
            axs.flat[1+ind*2].set_ylabel(r"$\mathrm{Im} \left[" +f"{row_names[ind]}"+r"\right]$")
            if y.ndim>1:
                axs.flat[1+ind*2].legend([f"For $t_1={t1_val:.2f}$" for t1_val in t1 ])

        # put this OUTSIDE the for loop
        #Draw horizontal line where the analytic value should be
        analytic_val = vpe.inverted_value(t=t1*vpe.tg)
        axs.flat[0+ind*2].set_prop_cycle(None)  # Resets the color cycle so colors of lines match the colors of plot

        axs.flat[0+ind*2].plot(
            plot_s[ np.array([0,-1])],
            #np.repeat(plot_s[(-1)][...,None],2),
            np.repeat(analytic_val[...,None], 2, axis=1).T, ":"
            )
        """
        for ind, t1_val in enumerate(t1):
            axs.flat[0+ind*2].annotate(
                f"{analytic_val[ind]:0.2f}", (0, analytic_val[ind]),
                xycoords="data", va='center')
            """








    #axs.flat[0].set_ylabel("Re[Cohen]")
    #axs.flat[1].set_ylabel("Im[Cohen]")
    print(f"t1={t1}")
    print(np.sum(np.real(ds*ys[2]), axis=0))
    print(np.sum(np.imag(ds*ys[2]), axis=0), " j" )
    #print( *zip(t1, np.real_if_close(np.sum(ds*ys[2], axis=0) ) )  )
    print()


    integrand_vals = ds*ys[2]
    inv_laplace_integ = np.sum(ds*ys[2], axis=0)
    inv_laplace_integ_re = np.real_if_close(inv_laplace_integ)
    analytic_values = vpe.inverted_value(t=t1*vpe.tg)

    df_params = pd.DataFrame({
        "t/tg": t1,
        "f(t)": analytic_values,
        **{
        #f"L-1[F(s)]_using_sum{ind}": inv_laplace_integ_re,
        #f"RawError{ind}": inv_laplace_integ_re-analytic_values,
        f"PercentError{ind}": np.round((inv_laplace_integ_re-analytic_values)/analytic_values*100, 1)
        for ind, inv_laplace_integ_re in enumerate(
            [
             np.real_if_close(np.sum(ds*ys[2], axis=0) ),
             np.real_if_close(np.trapz(ys[2],dx=ds,axis=0) ),
            ]
            )
        },
        "LastPlotted": np.real_if_close(ys[3][-1,:]),
        #"Re":np.real(inv_laplace_integ),
        #"Im":np.imag(inv_laplace_integ),
        },
        index=[f"{vpe.get_model_name()} - t1={t1_val:.2f}" for ind, t1_val in enumerate(t1)]
        )
    display(df_params)


if do_plot:
    fig.tight_layout()

plt.show()