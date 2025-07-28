def series_to_supervised(data, n_in=time_steps, n_out=output_length, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = DataFrame(data)
    cols, names = list(), list()
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
    for i in range(0, n_out):
        cols.append(df.iloc[:, 0].shift(-i))  # Only Active_Power
        if i == 0:
            names += ['Active_Power(t)']
        else:
            names += ['Active_Power(t+%d)' % i]
    agg = concat(cols, axis=1)
    agg.columns = names
    if dropnan:
        agg.dropna(inplace=True)
    return agg
