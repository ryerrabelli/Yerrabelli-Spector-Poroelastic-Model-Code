def abbreviate(num_array: list, sep=",") -> str:
    #diffs = num_array[1:] - num_array[:-1]
    diffs = [b-a for a,b in zip(num_array[:-1], num_array[1:])]
    abbr = []
    is_series = False
    
    for ind, elem in enumerate(num_array):
        if ind>0 and diffs[ind-1]==1:
            is_series = True
        else:
            if is_series:
                is_series = False
                abbr[-1] = abbr[-1] + "-" + str(num_array[ind-1])
            #abbr += [str(elem)]
            abbr.append(str(elem))

    if is_series:
        is_series = False
        abbr[-1] = abbr[-1] + "-" + str(elem)
    
    return sep.join(abbr)


if __name__ == '__main__':
    print(abbreviate([1,2,3,7]))
