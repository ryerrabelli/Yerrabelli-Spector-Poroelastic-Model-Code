##%%
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker
import time as timer
import collections

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
                          plot_props,
                          input_s,  # input_s should include all values of plot_s
                          input_times,  # input_times should include all values of plot_times
                          plot_s=None,
                          plot_s_s=None,
                          plot_times=None,
                          time_const=1,  # default is times area already nondimensional, otherwise, divide by this to become nondimensional
                          input_times_anal=None,
                          plot_times_anal=None,
                          inv_funcs_anal=None,  # Has to be None or a list of same length as func (if so, each element can still be None)
                          Marg=None,
                          assume_times_unique=True,  # assume_times_unique may speed up the function for plotting the numerical to analytic solution error
                          model_name=None,
                          func_labels=None,
                          legends_to_show_ct=1,
                          legend_fontsize="small",  # string values for fontsize are relative, while int are absolute in pts. String values: {'xx-small', 'x-small', 'small', 'medium', 'large', 'x-large', 'xx-large'}
                          do_plot_laplace_times_s=True,
                          ):

    if plot_times is None:
        plot_times = input_times
    if plot_s is None:
        plot_s = input_s

    funcs = func
    return_singles = False
    if not isinstance(funcs, collections.abc.Iterable):
        funcs = [funcs]
        return_singles = True
    if not isinstance(inv_funcs_anal, collections.abc.Iterable):
        inv_funcs_anal = [inv_funcs_anal] * len(funcs)
    if func_labels is None:
        func_labels = [None] * len(funcs)

    func_name = {key: val["y"]["name"] for key, val in plot_props.items()}
    x_names = {key: val["x"]["name"] for key, val in plot_props.items()}

    funcs_ct = len(funcs)
    legends_shown_ct = 0

    # Makes assumptions of the order of input_t and input_s having the plotting values at certain points
    plot_times_indices_in_input = np.arange(len(plot_times))
    plot_s_indices_in_input = np.arange(len(plot_s))
    if plot_s_s is None:
        plot_s_s = plot_s
        plot_s_s_indices_in_input = plot_s_indices_in_input
    else:
        plot_s_s_indices_in_input = len(plot_s) + np.arange(len(plot_s_s))

    # Non-positive s values give an error "invalid value encountered in sqrt"
    laplace_vals_all = [func(input_s) for func in funcs]

    t1=timer.time();
    inverted_vals_numerical_all = [euler_inversion(func, input_times / time_const, Marg=Marg) for func in funcs ]
    t2=timer.time()-t1
    print(f"It took {t2:0.4f} sec to numerically invert laplace the func for {len(input_times)} input times.")

    #inverted_vals_analytical = None if inv_func_anal is None else inv_func_anal(input_times_anal)
    #inversion_error = (inverted_vals_numerical-inverted_vals_analytical)/inverted_vals_numerical
    inverted_vals_analytical_all = [ (None if inv_func_anal is None else inv_func_anal(input_times_anal)) for inv_func_anal in inv_funcs_anal]
    if any(inverted_vals_analytical is not None for inverted_vals_analytical in inverted_vals_analytical_all):
        if plot_times_anal is None:
            plot_times_anal = plot_times
        if input_times_anal is None:
            input_times_anal = input_times
        # np.isin(.) return array of booleans unlike np.intersect1d(.)
        # np.is1d(.) is an old version of np.isin(.)
        # In either function, assume_unique can speed up the analysis.
        # np.isin(.) returns a bool at position n if the nth val in first array is also
        # present in the 2nd array for every value in the first array
        is_in_anal_times_too = np.isin(input_times, input_times_anal, assume_unique=assume_times_unique)
        if is_in_anal_times_too.any():  # If no elements in common, no point in getting the reverse indices
            is_in_num_times_too = np.isin(input_times, input_times_anal, assume_unique=assume_times_unique)
            #inversion_error = (inverted_vals_numerical-inverted_vals_analytical)/inverted_vals_analytical
            inversion_error_all = [(inverted_vals_numerical[is_in_anal_times_too] - inverted_vals_analytical[is_in_num_times_too]) \
                                   / inverted_vals_analytical[is_in_num_times_too] for inverted_vals_numerical,inverted_vals_analytical in zip(inverted_vals_numerical_all,inverted_vals_analytical_all)  ]

    # Plotting
    subplot_row_ct = 1 if all(inv_func_anal is None for inv_func_anal in inv_funcs_anal) and not do_plot_laplace_times_s else 2
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
    fig.set_figwidth(4.5 * subplot_col_ct)

    if subplot_row_ct == 1:
        ax00 = axs[0] if subplot_col_ct >= 1 else None
        ax01 = axs[1] if subplot_col_ct >= 2 else None
        axO2 = axs[2] if subplot_col_ct >= 3 else None
    else:
        ax00 = axs[0, 0]
        ax01 = axs[0, 1]
        ax10 = axs[1, 0]
        ax11 = axs[1, 1]

    ax_curr = ax00
    for laplace_vals, func_label in zip(laplace_vals_all, func_labels):
        ax_curr.plot(plot_s, laplace_vals[plot_s_indices_in_input], ".-b" if funcs_ct == 1 else ".-", label=func_label)
        #ax_curr.plot(input_s, laplace_vals*input_s, ".-b")
    if legends_shown_ct < legends_to_show_ct and any(func_label is not None for func_label in func_labels):
        ax_curr.legend(fontsize=legend_fontsize)   # fontsize argument only used if prop argument is not specified.
        legends_shown_ct += 1
    ax_curr.set_xlabel(plot_props.get("s").get("x").get("name"))  # plot_props["t"]["x"]["name"]
    ax_curr.set_ylabel(plot_props.get("s").get("y").get("name"))
    ax_curr.set_xscale(plot_props.get("s").get("x").get("scale") or "linear")
    ax_curr.set_yscale(plot_props.get("s").get("y").get("scale") or "linear")
    x_data = plot_s
    for each_axis, each_scale, set_lim, each_data in [
        (ax_curr.xaxis, ax_curr.get_xscale(), ax_curr.set_xlim, x_data),
        (ax_curr.yaxis, ax_curr.get_yscale(), ax_curr.set_ylim, None),
    ]:
        if each_scale == "linear":
            each_axis.set_minor_locator(
                matplotlib.ticker.AutoMinorLocator()
            )
            # theoretically, there should be no lower limit on s, but non-positive values throw an error in the function
            # ax_curr.set_xlim([0, None])
            ax_curr.set_xlim([min(0, *plot_s), max(plot_s)])
            #if each_data is not None:
            #    set_lim([min(0,*each_data), max(*each_data)])
        elif each_scale == "log":
            each_axis.set_minor_locator(
                matplotlib.ticker.LogLocator(base=10, subs=np.arange(2, 10), numticks=10 * each_axis.get_tick_space())
            )
    ax_curr.title.set_text("Laplace")
    ax_curr.grid(which="major")
    ax_curr.grid(which="minor", alpha=0.75, linestyle=":")

    ax_curr = ax01
    #np.abs(input_times / time_const - plot_times / time_const).argmin()
    for inverted_vals_numerical, func_label in zip(inverted_vals_numerical_all, func_labels):
        ax_curr.plot(plot_times/time_const, inverted_vals_numerical[plot_times_indices_in_input], ".-r" if funcs_ct == 1 else ".-", label=func_label)
    if legends_shown_ct < legends_to_show_ct and any(func_label is not None for func_label in func_labels):
        ax_curr.legend(fontsize="small")  # string values for fontsize are relative, while int are absolute
        legends_shown_ct += 1
    ax_curr.set_xlabel(plot_props.get("t").get("x").get("name"))  # plot_props["t"]["x"]["name"]
    ax_curr.set_ylabel(plot_props.get("t").get("y").get("name"))
    ax_curr.set_xscale(plot_props.get("t").get("x").get("scale") or "linear")
    ax_curr.set_yscale(plot_props.get("t").get("y").get("scale") or "linear")
    for each_axis, each_scale in [(ax_curr.xaxis, ax_curr.get_xscale()),
                                  (ax_curr.yaxis, ax_curr.get_yscale()), ]:
        if each_scale == "linear":
            each_axis.set_minor_locator(
                matplotlib.ticker.AutoMinorLocator()
            )
            ax_curr.set_xlim([0, None])
            #ax_curr.set_xlim([0, max(plot_times / time_const)])
        elif each_scale == "log":
            each_axis.set_minor_locator(
                matplotlib.ticker.LogLocator(base=10, subs=np.arange(2, 10), numticks=10 * each_axis.get_tick_space())
            )
    ax_curr.title.set_text("Numerical Inverse Laplace")
    ax_curr.grid(which="major")
    ax_curr.grid(which="minor", alpha=0.75, linestyle=":")

    if do_plot_laplace_times_s:
        ax_curr = ax10
        for laplace_vals, func_label in zip(laplace_vals_all, func_labels):
            ax_curr.plot(plot_s_s, plot_s_s*laplace_vals[plot_s_s_indices_in_input], ".-b" if funcs_ct == 1 else ".-", label=func_label)
        if legends_shown_ct < legends_to_show_ct and any(func_label is not None for func_label in func_labels):
            ax_curr.legend(fontsize=legend_fontsize)   # fontsize argument only used if prop argument is not specified.
            legends_shown_ct += 1
        ax_curr.set_xlabel(plot_props.get("s").get("x").get("name"))  # plot_props["t"]["x"]["name"]
        ax_curr.set_ylabel(r"$s\cdot$"+plot_props.get("s").get("y").get("name"))
        ax_curr.set_xscale(plot_props.get("s").get("x").get("scale") or "linear")
        ax_curr.set_yscale(plot_props.get("s").get("y").get("scale") or "linear")
        for each_axis, each_scale in [(ax_curr.xaxis, ax_curr.get_xscale()),
                                      (ax_curr.yaxis, ax_curr.get_yscale()), ]:
            if each_scale == "linear":
                each_axis.set_minor_locator(
                    matplotlib.ticker.AutoMinorLocator()
                )
                # theoretically, there should be no lower limit on s, but non-positive values throw an error in the function
                ax_curr.set_xlim([0, None])
                #ax_curr.set_xlim([0, max(plot_s_s)])
            elif each_scale == "log":
                each_axis.set_minor_locator(
                    matplotlib.ticker.LogLocator(base=10, subs=np.arange(2, 10), numticks=10 * each_axis.get_tick_space())
                )
        ax_curr.grid(which="major")
        ax_curr.grid(which="minor", alpha=0.75, linestyle=":")
        ax_curr.title.set_text(r"$s\times$Laplace")

    if any(inverted_vals_analytical is not None for inverted_vals_analytical in inverted_vals_analytical_all):
        ax_curr = ax11
        for inverted_vals_analytical in inverted_vals_analytical_all:
            ax_curr.plot(plot_times_anal, inverted_vals_analytical, ".-y" if funcs_ct == 1 else ".-")

        ax_curr.set_xlabel( (plot_props.get("t_anal") or plot_props.get("t")).get("x").get("name"))
        ax_curr.set_ylabel( (plot_props.get("t_anal") or plot_props.get("t")).get("y").get("name"))
        ax_curr.set_xscale( (plot_props.get("t_anal") or plot_props.get("t")).get("x").get("scale") or "linear")
        ax_curr.set_yscale( (plot_props.get("t_anal") or plot_props.get("t")).get("y").get("scale") or "linear")
        for each_axis, each_scale in [(ax_curr.xaxis, ax_curr.get_xscale()),
                                      (ax_curr.yaxis, ax_curr.get_yscale()), ]:
            if each_scale == "linear":
                each_axis.set_minor_locator(
                    matplotlib.ticker.AutoMinorLocator()
                )
                ax_curr.set_xlim([0, None])
                #ax_curr.set_xlim([0, max(plot_times_anal)])
            elif each_scale == "log":
                base = 10
                each_axis.set_minor_locator(
                    matplotlib.ticker.LogLocator(base=base, subs=np.arange(2, base), numticks=base*each_axis.get_tick_space())
                )
        ax_curr.grid(which="major")
        ax_curr.grid(which="minor", alpha=0.75, linestyle=":")
        ax_curr.title.set_text("Analytical Inverse Laplace")


        if not do_plot_laplace_times_s:
            ax_curr = ax10
            if is_in_anal_times_too.any():
                for inversion_error in inversion_error_all:
                    #ax_curr.plot(plot_times_anal[is_in_num_times_too], inversion_error * 100.0, ".-g")
                    ax_curr.plot(plot_times_anal[is_in_num_times_too], inversion_error, ".-g" if funcs_ct == 1 else ".-")
                if all(min(abs(inversion_error * 100.0)) < 0.1 for inversion_error in inversion_error_all):
                    ax_curr.set_ylim([-100, 100])
                else:
                    # Center y axis around 0
                    yabs_max = abs(max(ax10.get_ylim(), key=abs))*1.1
                    ax_curr.set_ylim(bottom=-yabs_max, top=yabs_max)
                    """ 
                    # Force 0 to be in the y axis range
                    if max(ax_curr.get_ylim()) < 0:
                        ax_curr.set_ylim(top=0)
                    elif min(ax_curr.get_ylim()) > 0:
                        ax_curr.set_ylim(bottom=0)
                    """
            else:
                ax_curr.set_ylim([-100, 100])

            ax_curr.set_xlabel(plot_props.get("t").get("x").get("name"))
            #ax_curr.set_ylabel(plot_props.get("t").get("y").get("name"))
            ax_curr.set_ylabel(r"% $error=\frac{f_{Numer}(t)-f_{Anal}(t)}{f_{Anal}(t)}$")
            ax_curr.set_xscale(plot_props.get("t").get("x").get("scale") or "linear")
            ax_curr.set_yscale(plot_props.get("t").get("y").get("scale") or "linear")
            for each_axis, each_scale in [(ax_curr.xaxis, ax_curr.get_xscale()),
                                          (ax_curr.yaxis, ax_curr.get_yscale()), ]:
                if each_scale == "linear":
                    each_axis.set_minor_locator(
                        matplotlib.ticker.AutoMinorLocator()
                    )
                    ax_curr.set_xlim([0, None])
                elif each_scale == "log":
                    each_axis.set_minor_locator(
                        matplotlib.ticker.LogLocator(base=10, subs=np.arange(2, 10), numticks=10 * each_axis.get_tick_space())
                    )
            ax_curr.grid(which="major")
            ax_curr.grid(which="minor", alpha=0.75, linestyle=":")
            ax_curr.title.set_text("Percent Error")

            # Note- if a y value is 0.5, then percentformatter makes it 0.5% not 50%.
            # Thus have to multiply by 100 before getting the y values.
            ax_curr.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter())

    elif subplot_row_ct >= 2:
        ax_curr = ax11
        ax_curr.axis("off")

    assert all(len(data) == funcs_ct for data in [laplace_vals_all,inverted_vals_numerical_all,inverted_vals_analytical_all])
    if return_singles:
        assert funcs_ct == 1
        # .pop is like accessing an element (like [0]) but works for non-indexable collections (like sets) too; Note-
        # pop removes an element from the collection while returning it
        laplace_vals_all = laplace_vals_all.pop()
        inverted_vals_numerical_all = inverted_vals_numerical_all.pop()
        inverted_vals_analytical_all = inverted_vals_analytical_all.pop()
    return fig, axs, laplace_vals_all, inverted_vals_numerical_all, inverted_vals_analytical_all

