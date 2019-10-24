def DCoperationcost(N, R, totflow):
    # N = number of stages
    # R = reflux ratio
    # totflow is total flowrate into DC column in kmol/h
    # see operational cost paper (Massimiliano Erricoa) for breakdown of

    C = 296.1  # $/m2yr
    Cdash = 17.76  # $/m2yr
    Cddash = 20.61 * 10 ** -3  # $/kmol
    Y = 8000  # h/yr
    E = 90  # % (efficiency assumed)
    G = 219.71  # kmol/hm2
    Gdash = 0.49  # kmol/hm2
    CEPCI2009 = 521.9
    CEPCI2019 = 613.16
    ZAR = 15.23  # rands per dollar for 01/10/2019

    TAC = (C / (E * Y * G) * N * (1 + R) + Cdash / (Y * Gdash) * (1 + R) + Cddash * (1 + R)) * CEPCI2019 / CEPCI2009
    TAC = TAC * ZAR  # TAC in rands
    TAC = TAC * totflow
    return TAC*0
