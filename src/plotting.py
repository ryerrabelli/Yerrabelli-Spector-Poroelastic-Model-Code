##%%
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker
import time as timer
import collections
import copy
import itertools

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


def format_axes(ax_curr,
                plot_x_vals=None,
                plot_y_vals=None,   # Usually not entered to not set limits on y
                plot_props={},
                key=""
                ):
    log_base = 10
    ax_curr.set_xlabel(plot_props.get(key).get("x").get("name"))  # aka plot_props[key]["x"]["name"]
    ax_curr.set_ylabel(plot_props.get(key).get("y").get("name"))
    ax_curr.set_xscale(plot_props.get(key).get("x").get("scale") or "linear")
    ax_curr.set_yscale(plot_props.get(key).get("y").get("scale") or "linear")
    for each_axis, each_scale, each_set_lim, each_data in [
        (ax_curr.xaxis, ax_curr.get_xscale(), ax_curr.set_xlim, plot_x_vals),
        (ax_curr.yaxis, ax_curr.get_yscale(), ax_curr.set_ylim, plot_y_vals),
    ]:
        if each_scale == "linear":
            each_axis.set_minor_locator(
                matplotlib.ticker.AutoMinorLocator()
            )
            if each_data is not None:
                # t shouldn't be negative and while theoretically, there should be no lower limit on s, non-positive
                # s values throw an error in the function
                each_set_lim([min(0, *each_data), max(*each_data)])
        elif each_scale == "log":
            each_axis.set_minor_locator(
                matplotlib.ticker.LogLocator(base=log_base, subs=np.arange(2, log_base), numticks=log_base * each_axis.get_tick_space())
            )
    ax_curr.grid(which="major")
    ax_curr.grid(which="minor", alpha=0.75, linestyle=":")


def plot_laplace_analysis(func,  # func (funcs) can either be a function or an iterable of functions
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

    ####################################################
    # SET SOME DEFAULTS
    ####################################################
    if plot_times is None:
        plot_times = input_times
    if plot_s is None:
        plot_s = input_s

    if plot_props.get("t_anal") is None:
        plot_props["t_anal"] = copy.deepcopy(plot_props.get("t"))   # the dict constructor causes a shallow copy to be made
    if plot_props.get("t_error") is None:
        plot_props["t_error"] = copy.deepcopy(plot_props.get("t"))
        plot_props.get("t_error").get("y")["name"] = r"% $error=\frac{f_{Numer}(t)-f_{Anal}(t)}{f_{Anal}(t)}$"
    if plot_props.get("sx") is None:
        plot_props["sx"] = copy.deepcopy(plot_props.get("s"))
        plot_props.get("sx").get("y")["name"] = r"$s\cdot$" + plot_props.get("s").get("y").get("name")
    funcs = func
    return_singles = False
    if not isinstance(funcs, collections.abc.Iterable):
        funcs = [funcs]
        return_singles = True
    if not isinstance(inv_funcs_anal, collections.abc.Iterable):
        inv_funcs_anal = [inv_funcs_anal] * len(funcs)
    if func_labels is None:
        func_labels = [None] * len(funcs)

    ####################################################
    # SETUP
    ####################################################
    funcs_ct = len(funcs)
    legends_shown_ct = 0

    # Makes assumptions of the order of input_t and input_s having the plotting values at certain points
    # Assumes sequential order of input_s having plot_s then plot_s_s
    plot_times_indices_in_input = np.arange(len(plot_times))
    plot_s_indices_in_input = np.arange(len(plot_s))
    if plot_s_s is None:
        plot_s_s = plot_s
        plot_s_s_indices_in_input = plot_s_indices_in_input
    else:
        plot_s_s_indices_in_input = len(plot_s) + np.arange(len(plot_s_s))

    # func_name = {key: val["y"]["name"] for key, val in plot_props.items()}
    # x_names = {key: val["x"]["name"] for key, val in plot_props.items()}
    # Default values
    # funcs = itertools.repeat(None)
    # laplace_vals_all = itertools.repeat(None)
    # inverted_vals_numerical_all = itertools.repeat(None)
    # inverted_vals_analytical_all = itertools.repeat(None)
    # inversion_error_all = itertools.repeat(None)

    ####################################################
    # GET CALCULATED VALUES TO PLOT
    ####################################################
    t0 = timer.time()
    # Non-positive s values give an error "invalid value encountered in sqrt"
    laplace_vals_all = [func(input_s) for func in funcs]
    t1 = timer.time();
    print(f"It took {t1-t0:0.4f} sec to evaluate the Laplace space func for {len(input_s)} input s vals.")
    inverted_vals_numerical_all = np.array([euler_inversion(func, input_times / time_const, Marg=Marg) for func in funcs ])
    t2 = timer.time()
    print(f"It took {t2-t1:0.4f} sec to numerically invert Laplace the func for {len(input_times)} input times.")

    inverted_vals_analytical_all = np.array([ (None if inv_func_anal is None else inv_func_anal(input_times_anal)) for inv_func_anal in inv_funcs_anal])
    # Default value, can be overwritten later
    inversion_error_all = np.array([None]*funcs_ct)
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
            inversion_error_all = np.array([(inverted_vals_numerical[is_in_anal_times_too] - inverted_vals_analytical[is_in_num_times_too]) \
                                   / inverted_vals_analytical[is_in_num_times_too] for inverted_vals_numerical,inverted_vals_analytical in zip(inverted_vals_numerical_all,inverted_vals_analytical_all)  ])

    ####################################################
    # SET UP PLOT STRUCTURE
    ####################################################
    subplot_row_ct = 1 if all(inv_func_anal is None for inv_func_anal in inv_funcs_anal) and not do_plot_laplace_times_s else 2
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

    ####################################################
    # CREATE PLOTS
    ####################################################

    #################################################
    ax_curr = ax00
    plot_x_vals = plot_s
    for laplace_vals, func_label in zip(laplace_vals_all, func_labels):
        ax_curr.plot(plot_x_vals, laplace_vals[plot_s_indices_in_input],
                     ".-b" if funcs_ct == 1 else ".-", label=func_label)
    if legends_shown_ct < legends_to_show_ct and any(func_label is not None for func_label in func_labels):
        ax_curr.legend(fontsize=legend_fontsize)   # fontsize argument only used if prop argument is not specified.
        legends_shown_ct += 1
    format_axes(ax_curr, plot_x_vals, None, plot_props=plot_props, key="s")
    ax_curr.title.set_text("Laplace")

    #################################################
    ax_curr = ax01
    plot_x_vals = plot_times/time_const
    for inverted_vals_numerical, func_label in zip(inverted_vals_numerical_all, func_labels):
        ax_curr.plot(plot_x_vals, inverted_vals_numerical[plot_times_indices_in_input],
                     ".-r" if funcs_ct == 1 else ".-", label=func_label)
    format_axes(ax_curr, plot_x_vals, None, plot_props=plot_props, key="t")
    ax_curr.title.set_text("Numerical Inverse Laplace")
    if legends_shown_ct < legends_to_show_ct and any(func_label is not None for func_label in func_labels):
        ax_curr.legend(fontsize=legend_fontsize)  # fontsize arg only used if prop arg is not specified
        legends_shown_ct += 1

    if do_plot_laplace_times_s:
        #################################################
        ax_curr = ax10
        plot_x_vals = plot_s_s
        for laplace_vals, func_label in zip(laplace_vals_all, func_labels):
            ax_curr.plot(plot_x_vals, plot_s_s*laplace_vals[plot_s_s_indices_in_input],
                         ".-b" if funcs_ct == 1 else ".-", label=func_label)
        format_axes(ax_curr, plot_x_vals, None, plot_props=plot_props, key="sx")
        ax_curr.title.set_text(r"$s\times$Laplace")
        if legends_shown_ct < legends_to_show_ct and any(func_label is not None for func_label in func_labels):
            ax_curr.legend(fontsize=legend_fontsize)  # fontsize arg only used if prop arg is not specified
            legends_shown_ct += 1
        #ax_curr.set_ylabel(r"$s\cdot$"+plot_props.get("s").get("y").get("name"))

    if any(inverted_vals_analytical is not None for inverted_vals_analytical in inverted_vals_analytical_all):
        #################################################
        ax_curr = ax11
        plot_x_vals = plot_times_anal
        for inverted_vals_analytical, in zip(inverted_vals_analytical_all):
            ax_curr.plot(plot_x_vals, inverted_vals_analytical, ".-y" if funcs_ct == 1 else ".-")
        format_axes(ax_curr, plot_x_vals, None, plot_props=plot_props, key="t_anal")
        ax_curr.title.set_text("Analytical Inverse Laplace")
        if legends_shown_ct < legends_to_show_ct and any(func_label is not None for func_label in func_labels):
            ax_curr.legend(fontsize=legend_fontsize)  # fontsize arg only used if prop arg is not specified
            legends_shown_ct += 1

        if not do_plot_laplace_times_s:
            #################################################
            ax_curr = ax10
            plot_x_vals = plot_times_anal[is_in_num_times_too]
            if is_in_anal_times_too.any():
                for inversion_error, in zip(inversion_error_all):
                    ax_curr.plot(plot_x_vals, inversion_error,
                                 ".-g" if funcs_ct == 1 else ".-")
                if all(min(abs(inversion_error * 100.0)) < 0.1 for inversion_error, in zip(inversion_error_all)):
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

            format_axes(ax_curr, plot_x_vals, None, plot_props=plot_props, key="t_error")
            ax_curr.title.set_text("Percent Error")
            if legends_shown_ct < legends_to_show_ct and any(func_label is not None for func_label in func_labels):
                ax_curr.legend(fontsize=legend_fontsize)  # fontsize arg only used if prop arg is not specified
                legends_shown_ct += 1

            # Note- if a y value is 0.5, then percentformatter makes it 0.5% not 50%.
            # Thus have to multiply by 100 before getting the y values.
            ax_curr.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter())

    elif subplot_row_ct >= 2:
        ax_curr = ax11
        ax_curr.axis("off")

    assert all(len(data) == funcs_ct for data in [
        laplace_vals_all, inverted_vals_numerical_all, inverted_vals_analytical_all
    ])
    if return_singles:
        assert funcs_ct == 1
        # .pop is like accessing an element (like [0]) but works for non-indexable collections (like sets) too; Note-
        # pop removes an element from the collection while returning it
        laplace_vals_all = laplace_vals_all.pop()
        inverted_vals_numerical_all = inverted_vals_numerical_all.pop()
        inverted_vals_analytical_all = inverted_vals_analytical_all.pop()

    return fig, axs, laplace_vals_all, inverted_vals_numerical_all, inverted_vals_analytical_all

