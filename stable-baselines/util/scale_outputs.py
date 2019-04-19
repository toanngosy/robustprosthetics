
def scale_range(x, x_min, x_max, y_min, y_max):
    """ Scales the entries in x which have a range between x_min and x_max
    to the range defined between y_min and y_max. """
    # y = a*x + b
    # a = deltaY/deltaX
    # b = y_min - a*x_min (or b = y_max - a*x_max)
    y = (y_max - y_min) / (x_max - x_min) * x + (y_min*x_max - y_max*x_min) / (x_max - x_min)
    return y



def scale_discrete(x):
    """ Discretize the entries in x """
    y = []
    for i in range(len(x)):
        if x[i]<0:
            y.append(0)
        else:
            y.append(1)
    return y

