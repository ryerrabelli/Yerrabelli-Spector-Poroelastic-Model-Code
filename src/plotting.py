##%%
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker
import time as timer

#from src.euler_inversion import euler_inversion
from euler_inversion import euler_inversion


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
                          input_s,       # input_s should include all values of plot_s
                          input_times,   # input_times should include all values of plot_times
                          plot_s=None,
                          plot_times=None,
                          time_const=1,  # default is times area already nondimensional, otherwise, divide by this to become nondimensional
                          input_times_anal=None,
                          plot_times_anal=None,
                          inv_func_anal=None,
                          Marg=None,
                          assume_times_unique=True,  # assume_times_unique may speed up the function for plotting the numerical to analytic solution error
                          model_name=None,
                          ):

    if plot_times is None:
        plot_times = input_times
    if plot_s is None:
        plot_s = input_s

    plot_times_indices_in_input = np.arange(len(plot_times))
    plot_s_indices_in_input = np.arange(len(plot_s))

    # Non-positive s values give an error "invalid value encountered in sqrt"
    laplace_vals = func(input_s)

    t1=timer.time();
    inverted_vals_numerical = euler_inversion(func, input_times / time_const, Marg=Marg)
    t2=timer.time()-t1
    print(f"It took {t2:0.4f} sec to numerically invert laplace the func for {len(input_times)} input times.")

    #inverted_vals_analytical = None if inv_func_anal is None else inv_func_anal(input_times_anal)
    #inversion_error = (inverted_vals_numerical-inverted_vals_analytical)/inverted_vals_numerical
    if inv_func_anal is None:
        inverted_vals_analytical = None
    else:
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
    subplot_row_ct = 1 if inv_func_anal is None else 2
    #subplot_row_ct = 1
    subplot_col_ct = 2
    fig, axs = plt.subplots(subplot_row_ct, subplot_col_ct)

    if model_name is not None:
        plt.suptitle(model_name)
        fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.85])   # [left, bottom, right, top] in normalized (0, 1) figure coordinates
        fig.set_figheight( (4*subplot_row_ct-1)/0.85 )

    else:
        fig.tight_layout()  # defaults to using the entire figure
        fig.set_figheight(  4*subplot_row_ct-1 )
    fig.set_dpi(150)
    fig.set_figwidth(9)

    if subplot_row_ct == 1:
        ax00 = axs[0]
        ax01 = axs[1]
    else:
        ax00 = axs[0, 0]
        ax01 = axs[0, 1]
        ax10 = axs[1, 0]
        ax11 = axs[1, 1]

    ax00.plot(plot_s, laplace_vals[plot_s_indices_in_input], ".-b")
    #ax00.plot(input_s, laplace_vals*input_s, ".-b")
    ax00.set_xlabel(x_names["s"])
    # theoretically, there should be no lower limit on s, but non-positive values throw an error in the function
    #ax00.set_xlim([0, None])
    ax00.set_xlim([0, max(plot_s)])
    ax00.set_ylabel(func_name["s"])
    ax00.title.set_text("Laplace")
    ax00.grid(which="major")
    ax00.grid(which="minor", alpha=0.75, linestyle=":")
    ax00.xaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator())
    ax00.yaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator())

    #np.abs(input_times / time_const - plot_times / time_const).argmin()
    ax01.plot(plot_times/time_const, inverted_vals_numerical[plot_times_indices_in_input], ".-r")
    ax01.set_xlabel(x_names["t"])
    ax01.set_xlim([0, max(plot_times / time_const)])
    ax01.set_ylabel(func_name["t"])
    ax01.title.set_text("Numerical Inverse Laplace")
    ax01.grid(which="major")
    ax01.grid(which="minor", alpha=0.75, linestyle=":")
    ax01.xaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator())
    ax01.yaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator())

    if inverted_vals_analytical is not None:
        ax11.plot(plot_times_anal, inverted_vals_analytical, ".-y")
        ax11.set_xlabel(x_names.get("t_anal") or x_names["t"])
        ax11.set_xlim([0, max(plot_times_anal)])
        ax11.set_ylabel(func_name.get("t_anal") or func_name["t"])
        ax11.title.set_text("Analytical Inverse Laplace")
        ax11.grid(which="major")
        ax11.grid(which="minor", alpha=0.75, linestyle=":")
        ax11.xaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator())
        ax11.yaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator())

        if False and  is_in_anal_times_too.any():
            ax10.plot(plot_times_anal[is_in_num_times_too], inversion_error * 100.0, ".-g")
        if is_in_anal_times_too.any():
            ax10.plot(plot_times_anal[is_in_num_times_too], inversion_error, ".-g")
            if min(abs(inversion_error * 100.0)) < 0.1:
                ax10.set_ylim([-100, 100])
            else:
                # Center y axis around 0
                yabs_max = abs(max(ax10.get_ylim(), key=abs))*1.1
                ax10.set_ylim(bottom=-yabs_max, top=yabs_max)
                """ 
                # Force 0 to be in the y axis range
                if max(ax10.get_ylim()) < 0:
                    ax10.set_ylim(top=0)
                elif min(ax10.get_ylim()) > 0:
                    ax10.set_ylim(bottom=0)
                """
        else:
            ax10.set_ylim([-100, 100])
        ax10.set_xlabel(x_names["t"])
        ax10.set_xlim([0, None])
        ax10.set_ylabel(func_name["t"])
        #ax10.set_ylabel("% error: (Numer-Anal)/Anal")
        ax10.set_ylabel(r"% $error=\frac{f_{Numer}(t)-f_{Anal}(t)}{f_{Anal}(t)}$")
        # Note- if a y value is 0.5, then percentformatter makes it 0.5% not 50%.
        # Thus have to multiply by 100 before getting the y values.
        ax10.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter())

        ax10.title.set_text("Percent Error")
        ax10.grid(which="major")
        ax10.grid(which="minor", alpha=0.75, linestyle=":")
        ax10.xaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator())
        ax10.yaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator())

    return fig, laplace_vals, inverted_vals_numerical, inverted_vals_analytical

