import numpy as np
def columnsize(N, volumetric_flow):
    # size in feet unless stated otherwise
    # heuristics taken from Ludwig Rules of Thumb

    # NB UNIT CONVERSIONS

    # height
    space = (20 / 12 + 2) / 2  # tray spacing
    trayheight = (N - 1) * space  # height due to tray spacing
    vapallowance = 4  # ft
    liqallowance = 6  # ft
    L = trayheight + vapallowance + liqallowance

    # diameter
    # classed according to heuristic ranges in order to maintain satisfactory diameters
    n_intervals = 10
    interval = np.linspace(27, 170, n_intervals-1)
    interval = list(interval)

    velocity = 0.6  # m/s using Ludwig

    Area = volumetric_flow/velocity
    Di = np.sqrt(Area*4/np.pi)
    Di *= 3.28


    """
    LDratio = [1 / 9, 1 / 10, 1 / 11, 1 / 12, 1 / 13, 1 / 14, 1 / 15, 1 / 16, 1 / 17]
    for i in range(n_intervals-2):
        maximum = interval[i + 1]
        minimum = interval[i]
        if L > minimum and L < maximum:
            Di = LDratio[i] * L
            break
    if L < interval[0]:
        Di = LDratio[0] * L

    if L > interval[-1]:
        Di = LDratio[-1] * L

    if N < 11:
        print("Too few stages- resulting column height too low")
    """

    return L, Di


