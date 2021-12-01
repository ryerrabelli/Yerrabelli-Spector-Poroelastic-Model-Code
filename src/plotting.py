##%%
import mpmath
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker
import time as timer
import collections
import copy
import warnings
import itertools

#from src.inverting import inverting.euler_inversion
#from inverting import euler_inversion
import inverting
import utils


def reload_imports():
    import importlib
    importlib.reload(inverting)
    importlib.reload(utils)


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
    return ind_a, ind_b


def format_axes(ax_curr,
                plot_x_vals=None,
                plot_y_vals=None,   # Usually not entered to not set limits on y
                # default value is {} when accessed via plot_props["missing_key"]. Does not change output access via .get(.)
                plot_props=collections.defaultdict(dict),
                key="",  # common values include "t", "s", "t_anal", ec
                default_scale="linear",  # i.e. {"linear","log"}
                default_log_base=10,
                ):
    log_base = default_log_base
    # .get(.) takes key and default value as inputs
    ax_curr.set_xlabel(plot_props.get(key, {}).get("x", {}).get(
        "name", key))  # default value of key for x axis
    ax_curr.set_ylabel(plot_props.get(key, {}).get("y", {}).get("name"))
    ax_curr.set_xscale(plot_props.get(key, {}).get(
        "x", {}).get("scale", default_scale))
    ax_curr.set_yscale(plot_props.get(key, {}).get(
        "y", {}).get("scale", default_scale))
    for axis_type, each_axis, each_scale, each_set_lim, each_data in [
        ("x", ax_curr.xaxis, ax_curr.get_xscale(), ax_curr.set_xlim, plot_x_vals),
        ("y", ax_curr.yaxis, ax_curr.get_yscale(), ax_curr.set_ylim, plot_y_vals),
    ]:
        if each_scale == "linear":
            each_axis.set_minor_locator(
                matplotlib.ticker.AutoMinorLocator()
            )
            if each_data is not None:
                # t shouldn't be negative and while theoretically, there should be no lower limit on s, non-positive
                # s values throw an error in the function
                # *{"a":1,"b":2} converts dict to separate arguments i.e. a=1, b=2. This is necessary for finding the min of either 0 or each_data
                each_set_lim([min(0, *each_data), max(*each_data)])
        elif each_scale == "log":
            each_axis.set_minor_locator(
                matplotlib.ticker.LogLocator(base=log_base, subs=np.arange(2, log_base),
                                             numticks=log_base * each_axis.get_tick_space())
            )
    ax_curr.grid(which="major")  # set major grid lines
    # set minor grid lines, but make them less visible
    ax_curr.grid(which="minor", alpha=0.75, linestyle=":")


def get_linenumber():
    """
    https://stackoverflow.com/questions/3056048/filename-and-line-number-of-python-script
    """
    import inspect
    currentframe = inspect.currentframe()
    # f_back to exit this helper function and to get the name of the calling code
    # If not inside a helper function, simply do either:
    # A) inspect.currentframe().f_lineno
    # B) inspect.getframeinfo(inspect.currentframe).lineno
    calling_frame = currentframe.f_back
    # frame info is named tuple: FrameInfo(frame, filename, lineno, function, code_context, index)
    frameinfo = inspect.getframeinfo(calling_frame)
    return frameinfo.lineno
    #return inspect.currentframe().f_back.f_lineno


def get_nested_defaultdict(from_dict: dict = None):
    """
    https://stackoverflow.com/questions/5369723/multi-level-defaultdict-with-variable-depth
    https://stackoverflow.com/questions/50013768/how-can-i-convert-nested-dictionary-to-defaultdict
    """
    if from_dict is None:
        # Return empty defaultdict
        # This is used for all the recursive calls (all the calls except the
        # first call) as well as the first call if a starting dict is not included
        return collections.defaultdict(get_nested_defaultdict)
    else:
        def defaultify(d):
            if isinstance(d, dict):
                #return defaultdict(lambda: None, {k: defaultify(v) for k, v in d.items()})
                return get_nested_defaultdict({k: defaultify(v) for k, v in d.items()})
            elif isinstance(d, list):  # (set, list, dict, tuple)
                return [defaultify(e) for e in d]
            elif isinstance(d, set):
                return {defaultify(e) for e in d}
            elif isinstance(d, tuple):
                return tuple(defaultify(e) for e in d)
            elif isinstance(d, collections.abc.Collection):
                return [defaultify(e) for e in d]
            else:
                return d

        return collections.defaultdict(get_nested_defaultdict, from_dict)


def plot_laplace_analysis(funcs,  # func (funcs) can either be a function or an iterable of functions
                          input_s,  # input_s should include all values of plot_s
                          input_times=None,  # input_times should include all values of plot_times
                          plot_props=None,  # default will be defaultdict instance
                          # deprecated as plot_props is preferred (also replaced prior func_name argument)
                          y_names={},
                          x_names={},  # deprecated as plot_props is preferred
                          func_name={},  # deprecated, replaced by y_names, which was then deprecated for plot_props
                          plot_s=None,
                          plot_s_s=None,
                          plot_times=None,
                          # default is times area already nondimensional (aka time_const=1), otherwise, divide by this to become nondimensional
                          time_const=None,
                          tg=None,   # deprecated, replaced by time_const
                          input_times_anal=None,
                          plot_times_anal=None,
                          # inv_funcs_anal has to be None or a list of same length as func (if so, each element can still be None)
                          inv_funcs_anal=None,
                          Marg=None,
                          # assume_times_unique may speed up the function for plotting the numerical to analytic solution error
                          assume_times_unique=True,
                          model_name=None,  # used as general figure title if not None
                          func_labels=None,  # if not None or single value, then array length should match funcs
                          legends_to_show_ct=1,
                          # string values for fontsize are relative, while int are absolute in pts. String values: {'xx-small', 'x-small', 'small', 'medium', 'large', 'x-large', 'xx-large'}
                          legend_fontsize="small",
                          do_plot_laplace_times_s=None,  # boolean with None as default
                          func=None,  # deprecated, replaced by funcs
                          plot_mode="standard",  # standard vs simple
                          ):

    plot_mode_options = ["standard", "simple"]
    if plot_mode is None:
        plot_mode = plot_mode_options[0]
    if plot_mode.lower() not in plot_mode_options:
        raise ValueError(
            "Invalid plot_mode type. Expected one of: %s" % plot_mode_options)
    if plot_mode.lower() in ["simple"] and do_plot_laplace_times_s is True:
        raise ValueError(
            "Plot_mode value of '%s' is not compatible with do_plot_laplace_times_s=True." % plot_mode)

    ####################################################
    # CREATE COMPATIBILITY WITH DEPRECATED ARGUMENTS
    ####################################################
    if tg:
        warnings.warn("Deprecated variable(s) sent. Noted at line #"
                      + str(get_linenumber()), DeprecationWarning)
        if not time_const:
            time_const = tg
    if func:
        warnings.warn("Deprecated variable(s) sent. Noted at line #"
                      + str(get_linenumber()), DeprecationWarning)
        if not funcs:
            funcs = func
    if func_name:
        warnings.warn("Deprecated variable(s) sent. Noted at line #"
                      + str(get_linenumber()), DeprecationWarning)
        if not y_names:
            # func_name was deprecated for y_names, which was then deprecated in favor of plot_props
            y_names = func_name
    if x_names or y_names:
        warnings.warn("Deprecated variable(s) sent. Noted at line #"
                      + str(get_linenumber()), DeprecationWarning)
        if not plot_props:
            """plot_props = {
                "t":{"y":{"name":r"$\sigma(\frac{t}{t_g})$"}, "x":{"name":r"$\overline{\sigma}(s)$", "scale":"linear"}},
                "s":{"y":{"name":r"$t/t_g$, unitless"},  "x":{"name":r"$s$, unitless"}},
            }"""
            """
            plot_props = {
                key1: {"y":{"name":y_name}, "x":{"name":x_name}} for ((key1, y_name), (key2, x_name)) in zip(y_names.items(), x_names.items()) if key1==key2
            }"""
            # Get the union of the keys of each set. They really should be the same, but this is just better logic
            # all_plot_type_keys is often a set like {"t", "s", "t_anal"}
            all_plot_type_keys = set(x_names) | set(y_names)
            plot_props = collections.defaultdict(dict, {
                key: {"y": {"name": y_names.get(key)}, "x": {"name": x_names.get(key)}} for key in all_plot_type_keys
            })
        else:
            # Can define the other values based off of these, but don't need to as they have been phased out
            # y_names = {key: val["y"]["name"] for key, val in plot_props.items()}
            # x_names = {key: val["x"]["name"] for key, val in plot_props.items()}
            pass

    ####################################################
    # SET SOME DEFAULTS
    ####################################################
    if time_const is None:  # aka tg
        time_const = 1
    if plot_s is None:
        plot_s = input_s
    if plot_times is None and input_times is None:
        raise ValueError(
            "At least one of plot_times or input_times must not be none.")
    elif plot_times is None:   # can't just do "if not plot_times" since truth value of an array is ambiguous
        plot_times = input_times
    elif input_times is None:
        input_times = plot_times

    if input_times_anal is None:
        input_times_anal = input_times
    if plot_times_anal is None:
        plot_times_anal = plot_times

    if plot_props is None:
        # defaultdict is a type that will by default return a dict
        plot_props = collections.defaultdict(dict)
    else:
        plot_props = collections.defaultdict(dict, plot_props)

    if not plot_props["t"]:
        plot_props["t"] = {"y": {}, "x": {}}
    if not plot_props["s"]:
        plot_props["s"] = {"y": {}, "x": {}}
    if not plot_props.get("t_anal"):
        # Create a deep copy of the dict and all subdicts so that changes to the new one are not made to the old one too
        plot_props["t_anal"] = copy.deepcopy(plot_props.get("t"))
    if not plot_props.get("t_error"):
        plot_props["t_error"] = copy.deepcopy(plot_props.get("t"))
        plot_props.get("t_error").get(
            "y")["name"] = r"% $error=\frac{f_{Numer}(t)-f_{Anal}(t)}{f_{Anal}(t)}$"
    if not plot_props.get("sx"):  # stands for "s x Laplace"
        plot_props["sx"] = copy.deepcopy(plot_props.get("s"))
        if plot_props.get("sx", {}).get("y", {}).get("name", None) is not None:
            #plot_props.get("s").get("y").get("name")
            plot_props["sx"]["y"]["name"] = r"$s\cdot$" + \
                plot_props["s"]["y"]["name"]
    if do_plot_laplace_times_s is None:
        do_plot_laplace_times_s = (plot_mode.lower() not in ["simple"])
        #do_plot_laplace_times_s = False if plot_mode.lower() in ["simple"] else True

    # return_singles argument is used to keep track of whether funcs was passed in as a function or as a list of functions
    # The output would respectively be a value or list of values
    return_singles = False  # default value
    # Collection guarantees only __contains__, __iter__, __len__ methods
    # Need __len__ for assess len and confirming the lens match up between different collections
    # __iter__ is needed to iterate
    # https://docs.python.org/3/library/collections.abc.html
    # isinstance returns true if class has all the required methods, but wasn't
    # actually defined explicitly as a subclass of the collection type
    if not isinstance(funcs, collections.abc.Collection):
        funcs = [funcs]
        return_singles = True
    funcs_ct = len(funcs)
    if not isinstance(inv_funcs_anal, collections.abc.Collection):
        inv_funcs_anal = [inv_funcs_anal] * funcs_ct
    if func_labels is None:
        func_labels = [None] * len(funcs)

    # Default values
    # funcs = itertools.repeat(None)
    # laplace_vals_all = itertools.repeat(None)
    # inverted_vals_numerical_all = itertools.repeat(None)
    # inverted_vals_analytical_all = itertools.repeat(None)
    # inversion_error_all = itertools.repeat(None)

    ####################################################
    # SETUP
    ####################################################
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

    ####################################################
    # GET CALCULATED VALUES TO PLOT
    ####################################################
    t0 = timer.time()
    # Non-positive s values give an error "invalid value encountered in sqrt"
    laplace_vals_all = [func(input_s) for func in funcs]
    t1 = timer.time()
    print(
        f"It took {t1-t0:0.4f} sec to evaluate the Laplace space func for {len(input_s)} input s vals.")
    inverted_vals_numerical_all = np.array([
        inverting.euler_inversion(func, input_times / time_const, Marg=Marg) for func in funcs
        #mpmath.invertlaplace(func, input_times / time_const, method='talbot') for func in funcs
    ])
    t2 = timer.time()
    print(
        f"It took {t2-t1:0.4f} sec to numerically invert Laplace the func for {len(input_times)} input times.")

    inverted_vals_analytical_all = np.array([(None if inv_func_anal is None else inv_func_anal(
        input_times_anal)) for inv_func_anal in inv_funcs_anal])
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
        is_in_anal_times_too = np.isin(
            input_times, input_times_anal, assume_unique=assume_times_unique)
        if is_in_anal_times_too.any():  # If no elements in common, no point in getting the reverse indices
            is_in_num_times_too = np.isin(
                input_times, input_times_anal, assume_unique=assume_times_unique)
            #inversion_error = (inverted_vals_numerical-inverted_vals_analytical)/inverted_vals_analytical
            inversion_error_all = np.array([
                (inverted_vals_numerical[is_in_anal_times_too]
                 - inverted_vals_analytical[is_in_num_times_too])
                / inverted_vals_analytical[is_in_num_times_too]
                for inverted_vals_numerical, inverted_vals_analytical
                in zip(inverted_vals_numerical_all, inverted_vals_analytical_all)
            ])

    ####################################################
    # SET UP PLOT STRUCTURE
    ####################################################
    if (not do_plot_laplace_times_s
            and (all(inv_func_anal is None for inv_func_anal in inv_funcs_anal) or
                 plot_mode is None or plot_mode.lower() in ["simple"])):
        subplot_row_ct = 1
    else:
        subplot_row_ct = 2
    subplot_col_ct = 2
    fig, axs = plt.subplots(subplot_row_ct, subplot_col_ct)

    if model_name is not None:
        plt.suptitle(model_name)
        # [left, bottom, right, top] in normalized (0, 1) figure coordinates
        fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.85])
        fig.set_figheight((4*subplot_row_ct-1)/0.85)

    else:
        fig.tight_layout()  # defaults to using the entire figure
        fig.set_figheight(4*subplot_row_ct-1)
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
        # fontsize argument only used if prop argument is not specified.
        ax_curr.legend(fontsize=legend_fontsize)
        legends_shown_ct += 1
    format_axes(ax_curr, plot_x_vals, None, plot_props=plot_props, key="s")

    #ax_curr.text(3, 8, 'boxed italics text in data coords', style='italic', bbox={'facecolor': 'red', 'alpha': 0.5, 'pad': 10})
    s_eqn = plot_props.get("s", {}).get("eqn")
    if s_eqn:
        if s_eqn.get("annotate"):
            #ax_curr.annotate(s_eqn.get("text"), xy=(0, 1),  xycoords="axes fraction", ha="left", va="top")
            #ax_curr.annotate(s_eqn.get("text"), xy=(1, 1), xycoords="axes fraction", ha="right", va="top", fontweight='bold',
            #    bbox=dict(boxstyle="square", fc="w", alpha=0.2, ec="k", lw=2))
            import matplotlib.offsetbox
            at = matplotlib.offsetbox.AnchoredText(s_eqn.get("text"),
                                                   frameon=True, loc='upper right')
            #at.patch.set_boxstyle("square,pad=0.,alpha=0.5")
            at.patch.set_boxstyle("square,pad=0.")
            ax_curr.add_artist(at)
            ax_curr.title.set_text("Laplace")
        else:
            ax_curr.title.set_text("Laplace: " + s_eqn.get("text"))
    else:
        ax_curr.title.set_text("Laplace")

    #################################################
    ax_curr = ax01
    plot_x_vals = plot_times/time_const
    for inverted_vals_numerical, func_label in zip(inverted_vals_numerical_all, func_labels):
        ax_curr.plot(plot_x_vals, inverted_vals_numerical[plot_times_indices_in_input],
                     ".-r" if funcs_ct == 1 else ".-", label=func_label)

    format_axes(ax_curr, plot_x_vals, None, plot_props=plot_props, key="t")
    # Will be overriden if analytic solution plotted on same graph
    ax_curr.title.set_text("Numerical Inverse Laplace")

    if any(inverted_vals_analytical is not None for inverted_vals_analytical in inverted_vals_analytical_all):
        # Simple mode plots the analytic and numerical solutions in the same plot and tries to ensure only one subplot row is used
        if plot_mode is None or plot_mode.lower() in ["simple"]:
            # ax_curr still is ax01 here
            # Resets the color cycle so colors of lines match the colors of plot
            ax_curr.set_prop_cycle(None)
            # Plot analytic solution on the same graph as numerical
            plot_x_vals = plot_times_anal if plot_times_anal is not None else plot_x_vals
            for inverted_vals_analytical, in zip(inverted_vals_analytical_all):
                ax_curr.plot(plot_x_vals, inverted_vals_analytical,
                             "--k" if funcs_ct == 1 else "--")
            #format_axes(ax_curr, plot_x_vals, None, plot_props=plot_props, key="t")   # repeated call
            ax_curr.title.set_text("Inverse Laplace")  # repeated call
            if funcs_ct == 1 and func_labels == [None]:
                ax_curr.legend([
                    "Numerical "
                    + plot_props.get("t", {}).get("eqn", {}).get("text", ""),
                    "Analytic "
                    + plot_props.get("t_anal", {}).get("eqn",
                                                       {}).get("text", ""),
                    ])
        else:
            # ax_curr still is ax01 here
            if legends_shown_ct < legends_to_show_ct and any(func_label is not None for func_label in func_labels):
                # fontsize arg only used if prop arg is not specified
                ax_curr.legend(fontsize=legend_fontsize)
                legends_shown_ct += 1

            #################################################
            ax_curr = ax11
            plot_x_vals = plot_times_anal
            for inverted_vals_analytical, in zip(inverted_vals_analytical_all):
                ax_curr.plot(plot_x_vals, inverted_vals_analytical,
                             ".-y" if funcs_ct == 1 else ".-")
            format_axes(ax_curr, plot_x_vals, None,
                        plot_props=plot_props, key="t_anal")
            ax_curr.title.set_text("Analytical Inverse Laplace")
            if legends_shown_ct < legends_to_show_ct and any(func_label is not None for func_label in func_labels):
                # fontsize arg only used if prop arg is not specified
                ax_curr.legend(fontsize=legend_fontsize)
                legends_shown_ct += 1

            if not do_plot_laplace_times_s:
                # Since we have an extra plot box, let's plot the error
                #################################################
                ax_curr = ax10
                plot_x_vals = plot_times_anal[is_in_num_times_too]
                if is_in_anal_times_too.any():
                    for inversion_error, in zip(inversion_error_all):
                        ax_curr.plot(plot_x_vals, inversion_error * 100.0,
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

                format_axes(ax_curr, plot_x_vals, None,
                            plot_props=plot_props, key="t_error")
                ax_curr.title.set_text("Percent Error")
                if legends_shown_ct < legends_to_show_ct and any(func_label is not None for func_label in func_labels):
                    # fontsize arg only used if prop arg is not specified
                    ax_curr.legend(fontsize=legend_fontsize)
                    legends_shown_ct += 1

                # Note- if a y value is 0.5, then percentformatter makes it 0.5% not 50%.
                # Thus have to multiply by 100 before getting the y values.
                ax_curr.yaxis.set_major_formatter(
                    matplotlib.ticker.PercentFormatter())

    elif subplot_row_ct >= 2:
        ax_curr = ax11
        ax_curr.axis("off")

    if do_plot_laplace_times_s:
        #################################################
        ax_curr = ax10
        plot_x_vals = plot_s_s
        for laplace_vals, func_label in zip(laplace_vals_all, func_labels):
            ax_curr.plot(plot_x_vals, plot_s_s*laplace_vals[plot_s_s_indices_in_input],
                         ".-b" if funcs_ct == 1 else ".-", label=func_label)
        format_axes(ax_curr, plot_x_vals, None,
                    plot_props=plot_props, key="sx")
        ax_curr.title.set_text(r"$s\times$Laplace")
        if legends_shown_ct < legends_to_show_ct and any(func_label is not None for func_label in func_labels):
            # fontsize arg only used if prop arg is not specified
            ax_curr.legend(fontsize=legend_fontsize)
            legends_shown_ct += 1
        #ax_curr.set_ylabel(r"$s\cdot$"+plot_props.get("s").get("y").get("name"))

    assert all(len(outputted_results) == funcs_ct for outputted_results in [
        laplace_vals_all, inverted_vals_numerical_all, inverted_vals_analytical_all
    ])
    if return_singles:
        assert funcs_ct == 1
        # next(iter(var)) instead of just var[0] because var doesn't have to be a list, just any collection (thus, only guarantees having __contains__, __iter__, __len__ methods)
        # This way, you can still access the first (and only) element, which should be a numpy array
        laplace_vals_all = next(iter(laplace_vals_all))
        inverted_vals_numerical_all = next(iter(inverted_vals_numerical_all))
        inverted_vals_analytical_all = next(iter(inverted_vals_analytical_all))

    return fig, axs, laplace_vals_all, inverted_vals_numerical_all, inverted_vals_analytical_all
