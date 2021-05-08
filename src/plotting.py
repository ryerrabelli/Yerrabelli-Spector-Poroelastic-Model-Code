##%%
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import time as timer

from src.euler_inversion import euler_inversion


def overlap(a, b, ):
    """
    https://www.followthesheep.com/?p=1366
    :param a:
    :type a:
    :param b:
    :type b:
    :return:
    :rtype:
    """
    # return the indices in a that overlap with b, also returns
    # the corresponding index in b only works if both a and b are unique!
    # This is not very efficient but it works
    bool_a = np.isin(a, b, assume_unique=True)
    ind_a = np.arange(len(a))
    ind_a = ind_a[bool_a]

    ind_b = np.array([np.argwhere(b == a[x]) for x in ind_a]).flatten()
    return ind_a,ind_b

def plot_laplace_analysis(func,
                          func_name,
                          x_names,
                          s_vals,
                          input_times,
                          plot_times,
                          tg,
                          input_times_anal=None,
                          plot_times_anal=None,
                          inv_func_anal=None,
                          Marg=None,
                          assume_times_unique=True,  # assume_times_unique may speed up the function for plotting the numerical to analytic solution error
                          ):
    t1=timer.time();
    inverted_vals_numerical = euler_inversion(func, input_times/tg, Marg=Marg)
    t2=timer.time()-t1
    print("Time taken in sec:", t2)

    # Non-positive s values give an error "invalid value encountered in sqrt"
    laplace_vals = func(s_vals)

    #inverted_vals_analytical = None if inv_func_anal is None else inv_func_anal(input_times_anal)
    #inversion_error = (inverted_vals_numerical-inverted_vals_analytical)/inverted_vals_numerical
    if inv_func_anal is not None:
        if plot_times_anal is None:
            plot_times_anal = plot_times
        if input_times_anal is None:
            input_times_anal = input_times
        inverted_vals_analytical = inv_func_anal(input_times_anal)
        # np.isin(.) return array of booleans unlike np.intersect1d(.)
        # np.is1d(.) is an old version of np.isin(.)
        # In either function, assume_unique can speed up the analysis.
        # np.isin(.) returns a bool at position n if the nth val in first array is also
        # present in the 2nd array for every value in the first array
        is_in_anal_times_too = np.isin(input_times, input_times_anal, assume_unique=assume_times_unique)
        if is_in_anal_times_too.any():  # If no elements in common, no point in getting the reverse indices
            is_in_num_times_too = np.isin(input_times, input_times_anal, assume_unique=assume_times_unique)
            #inversion_error = (inverted_vals_numerical-inverted_vals_analytical)/inverted_vals_analytical
            inversion_error = (inverted_vals_numerical[is_in_anal_times_too] - inverted_vals_analytical[is_in_num_times_too]) \
                             / inverted_vals_analytical[is_in_num_times_too]

    # Plotting
    fig, axs = plt.subplots(2,2)
    fig.tight_layout()
    fig.set_figwidth(9)
    fig.set_figheight(3*2+1)
    fig.set_dpi(150)

    ax00 = axs[0, 0]
    ax01 = axs[0, 1]
    ax10 = axs[1, 0]
    ax11 = axs[1, 1]

    ax00.plot(s_vals, laplace_vals, ".-b")
    #ax00.plot(s_vals, laplace_vals*s_vals, ".-b")
    ax00.set_xlabel(x_names["s"])
    # theoretically, there should be no limit on s, but non-positive values throw an error in the function
    ax00.set_xlim([0, None])
    ax00.set_ylabel(func_name["s"])
    ax00.grid()
    ax00.title.set_text("Laplace")

    ax01.plot(plot_times, inverted_vals_numerical, ".-r")
    ax01.set_xlabel(x_names["t"])
    ax01.set_xlim([0, None])
    ax01.set_ylabel(func_name["t"])
    ax01.grid()
    ax01.title.set_text("Numerical Inverse Laplace")

    if inverted_vals_analytical is not None:
        ax11.plot(plot_times_anal, inverted_vals_analytical, ".-y")
        ax11.set_xlabel(x_names.get("t_anal") or x_names["t"])
        ax11.set_xlim([0, None])
        ax11.set_ylabel(func_name.get("t_anal") or func_name["t"])
        ax11.grid()
        ax11.title.set_text("Analytical Inverse Laplace")

        if is_in_anal_times_too.any():
            ax10.plot(plot_times_anal[is_in_num_times_too], inversion_error*100.0, ".-g")
            if min(abs(inversion_error * 100.0)) < 0.1:
                ax10.set_ylim([-100, 100])
            elif max(ax10.get_ylim()) < 0:
                ax10.set_ylim(top=0)
            elif min(ax10.get_ylim()) > 0:
                ax10.set_ylim(bottom=0)
        else:
            ax10.set_ylim([-100, 100])
        ax10.set_xlabel(x_names["t"])
        ax10.set_xlim([0, None])
        ax10.set_ylabel(func_name["t"])
        ax10.set_ylabel("% error")
        # Note- if a y value is 0.5, then percentformatter makes it 0.5% not 50%.
        # Thus have to multiply by 100 before getting the y values.
        ax10.yaxis.set_major_formatter(mtick.PercentFormatter())

        ax10.grid()
        ax10.title.set_text("Percent Error (Numerical to Analytical)")

    return fig

