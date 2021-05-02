##%%
import numpy as np
import matplotlib.pyplot as plt
import time as timer

from src.euler_inversion import euler_inversion


def plot_laplace_analysis(func, func_name, x_names, s_vals, input_times, plot_times, input_times_anal=None, plot_times_anal=None, inv_func_anal=None ):
    t1=timer.time();
    inverted_vals=euler_inversion(func, input_times)
    t2=timer.time()-t1
    print("Time taken in sec:", t2)

    # Non-positive s values give an error "invalid value encountered in sqrt"
    laplace_vals = func(s_vals)
    #inverted_vals_analytical = None if inv_func_anal is None else inv_func_anal(input_times_anal)
    #percent_error = (inverted_vals-inverted_vals_analytical)/inverted_vals
    if inv_func_anal is not None:
        inverted_vals_analytical = inv_func_anal(input_times_anal)
        percent_error = (inverted_vals-inverted_vals_analytical)/inverted_vals * 100

    fig, axs = plt.subplots(2,2)
    fig.set_figwidth(9)
    fig.set_figheight(3*2+1)
    fig.set_dpi(150)

    axs[0,0].plot(s_vals, laplace_vals, ".-b")
    axs[0,0].set_xlabel(x_names["s"])
    # theoretically, there should be no limit on s, but non-positive values throw an error in the function
    axs[0,0].set_xlim([0, None])
    axs[0,0].set_ylabel(func_name["s"])
    axs[0,0].grid()
    axs[0,0].title.set_text("Laplace")

    axs[0,1].plot(plot_times, inverted_vals, ".-r")
    axs[0,1].set_xlabel(x_names["t"])
    axs[0,1].set_xlim([0, None])
    axs[0,1].set_ylabel(func_name["t"])
    axs[0,1].grid()
    axs[0,1].title.set_text("Numerical Inverse Laplace")

    if inverted_vals_analytical is not None:
      axs[1,1].plot(plot_times_anal, inverted_vals_analytical, ".-y")
      axs[1,1].set_xlabel(x_names["t"])
      axs[1,1].set_xlim([0, None])
      axs[1,1].set_ylabel(func_name["t"])
      axs[1,1].grid()
      axs[1,1].title.set_text("Analytical Inverse Laplace")


      axs[1,0].plot(plot_times, percent_error, ".-g")
      axs[1,0].set_xlabel(x_names["t"])
      axs[1,0].set_xlim([0, None])
      #axs[1,0].set_ylabel(func_name["t"])
      axs[1,0].set_ylabel("% error")
      if min(abs(percent_error)) < 0.1:
        axs[1,0].set_ylim([-100, 100])
      axs[1,0].grid()
      axs[1,0].title.set_text("Percent Error (Numerical to Analytical)")

    return fig