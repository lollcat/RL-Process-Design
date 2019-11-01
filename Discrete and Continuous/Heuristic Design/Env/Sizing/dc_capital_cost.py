import math
import numpy as np


def expensiveDC(Di, L, Po, To, N):

    # VESSEL COSTING
    # Di: feet
    # L: feet
    # Po: kpa
    # To: deg c

    To = (To * 9 / 5) + 32  # convert to Celcius to farenheit
    Po = (Po - 101.325) * 0.145038  # kPa to Psig
    # calculating operational pressure
    if Po < 10:  # all pressures in psig
        Pd = 10
    elif Po < 1000:
        Pd = math.exp(0.60608 + 0.91615 * (math.log(Po)) + 0.0015655 * (math.log(Po)) ** 2)
    else:
        Pd = 1.1 * Po
    #print(Pd)

    # calculating S
    Td = To + 50  # degrees F
    interval = np.linspace(650, 900, 6)
    interval = list(interval)
    interval.insert(0, -20)
    Sreturn = [15000, 15000, 15000, 14750, 14200, 13100]
    for i in range(6):
        maximum = interval[i + 1]
        minimum = interval[i]

        if Td > minimum and Td < maximum:
            S = Sreturn[i]
            break

    if Td > interval[-1]:
        S = Sreturn[-1]
        print("Smax")

    if Td < interval[0]:
        S = Sreturn[0]
        print("Smin")



    # calculating E
    E = 0.85  # NB can change dependent on tp

    # calculating vessel wall thickness (tp)
    tp = (Pd * Di * 12) / (2 * S * E - 1.2 * Pd)  # converts Di to inches
    #print(tp)

    # weight of vessel and heads
    W = math.pi * (Di * 12 + tp) * (
                L + 0.8 * Di) * 12 * 0.284  # assumes no wind, earthquakes or corrosion, else tp changes
    #print(W)  # pounds

    # calculating Cv, Cpl and Cp
    # assumes carbon steel
    CEPCI2003 = 394
    CEPCI2019 = 613.16
    Fm = 1  # carbon steel, can change
    ZAR = 15.23  # rands per dollar for 01/10/2019

    Cv2003 = math.exp(7.0374 + 0.18255 * (math.log(W)) + 0.02297 * (math.log(W)) ** 2)
    Cv = Cv2003 * CEPCI2019 / CEPCI2003

    Cpl2003 = 237.1 * Di ** 0.63316 * L ** 0.80161
    Cpl = Cpl2003 * CEPCI2003 / CEPCI2019

    Cp = Fm * Cv + Cpl  # USD
    Cp = Cp * ZAR
    #print(Cp)

    # calculating tray cost
    # assumes carbon steel sieve trays - else see Seader pg. 532
    Fmt = 1  # material factor for carbon steel
    Ftt = 1  # tray factor for sieve trays
    if N >= 20:
        Fnt = 1
    else:
        Fnt = 2.25 / (1.0414 ** N)

    CBT = 369 * math.exp(0.1739 * Di)
    Ctray2003 = N * Fmt * Ftt * Fnt * CBT
    Ctray = Ctray2003 * CEPCI2019 / CEPCI2003  # USD
    Ctray = Ctray * ZAR

    # calculating total capital cost
    Ctot = Cp + Ctray

    """
    # throwing error if operating outside of cost correlation limits
    if W < 4200:
        raise ValueError("Vessel weight is too low")
    elif W > 1000000:
        raise ValueError("Vessel weight is too high")
    elif Di < 3:
        raise ValueError("Vessel internal diameter is too low")
    elif Di > 17:
        raise ValueError("Vessel internal diameter is too high")
    elif L < 27:
        raise ValueError("Vessel height is too low")
    elif L > 170:
        raise ValueError("Vessel height is too large")
    elif Td < -20:
        raise ValueError("Vessel design temp is too low")
    elif Td > 900:
        raise ValueError("Vessel design temp is too high")
    else:
        return Ctot
    """

    # throwing error if operating outside of cost correlation limits
    """
    if W < 4200:
        print("Vessel weight is too low")
    elif W > 1000000:
        print("Vessel weight is too high")
    elif Di < 3:
        print("Vessel internal diameter is too low")
    elif Di > 17:
        print("Vessel internal diameter is too high")
    elif L < 27:
        print("Vessel height is too low")
    elif L > 170:
        print("Vessel height is too large")
    elif Td < -20:
        print("Vessel design temp is too low")
    elif Td > 900:
        print("Vessel design temp is too high")
    """
    return Ctot
