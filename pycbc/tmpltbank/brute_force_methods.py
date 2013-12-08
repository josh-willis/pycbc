from __future__ import division
import copy
import numpy
from pycbc.tmpltbank.coord_utils import get_cov_params
from pycbc.tmpltbank.lambda_mapping import get_beta_sigma_from_aligned_spins

def get_physical_covaried_masses(xis, bestMasses, bestXis, req_match, \
                                 massRangeParams, metricParams, fUpper, \
                                 giveUpThresh = 5000):
    """
    This function takes the position of a point in the xi parameter space and
    iteratively finds a close point in the physical coordinate space (masses
    and spins).
 
    Parameters
    -----------
    xis : list or array
        Desired position of the point in the xi space. If only N values are
        provided and the xi space's dimension is larger then it is assumed that
        *any* value in the remaining xi coordinates is acceptable.
    bestMasses : list
        Contains [totalMass, eta, spin1z, spin2z]. Is a physical position
        mapped to xi coordinates in bestXis that is close to the desired point.
        This is aimed to give the code a starting point.
    bestXis : list
        Contains the position of bestMasses in the xi coordinate system.
    req_match : float
        Desired maximum mismatch between xis and the obtained point. If a point
        is found with mismatch < req_match immediately stop and return that
        point. A point with this mismatch will not always be found.
    massRangeParams : massRangeParameters instance
        Instance holding all the details of mass ranges and spin ranges.
    metricParams : metricParameters instance
        Structure holding all the options for construction of the metric
        and the eigenvalues, eigenvectors and covariance matrix
        needed to manipulate the space.
    fUpper : float
        The value of fUpper that was used when obtaining the xi_i
        coordinates. This lets us know how to rotate potential physical points
        into the correct xi_i space. This must be a key in metricParams.evals,
        metricParams.evecs and metricParams.evecsCV
        (ie. we must know how to do the transformation for
        the given value of fUpper)
    giveUpThresh : int, optional (default = 5000)
        The program will try this many iterations. If no close matching point
        has been found after this it will give up.

    Returns
    --------
    mass1 : float
        The heavier mass of the obtained point.
    mass2 : float
        The smaller mass of the obtained point
    spin1z : float
        The heavier bodies spin of the obtained point.
    spin2z : float
        The smaller bodies spin of the obtained point.
    count : int
        How many iterations it took to find the point. For debugging.
    mismatch : float
        The mismatch between the obtained point and the input xis.
    new_xis : list
        The position of the point in the xi space
    """
    # TUNABLE PARAMETERS GO HERE!
    # This states how far apart to scatter test points in the first proposal
    origScaleFactor = 1

    # Set up
    xi_size = len(xis)
    scaleFactor = origScaleFactor
    bestChirpmass = bestMasses[0] * (bestMasses[1])**(3./5.)
    count = 0
    unFixedCount = 0
    currDist = 100000000000000000
    while(1):
        # If we are a long way away we use larger jumps
        if count:
            if currDist > 1 and scaleFactor == origScaleFactor:
                scaleFactor = origScaleFactor*10
        # Get a set of test points with mass -> xi mappings
        chirpmass, totmass, eta, spin1z, spin2z, diff, mass1, mass2, beta, \
              sigma, gamma, chis, new_xis = get_mass_distribution(\
                    [bestChirpmass,bestMasses[1],bestMasses[2],bestMasses[3]],\
                    scaleFactor, massRangeParams, metricParams, fUpper)
        cDist = (new_xis[0] - xis[0])**2
        for j in range(1,xi_size):
            cDist += (new_xis[j] - xis[j])**2
        if (cDist.min() < req_match):
            idx = cDist.argmin()
            scaleFactor = origScaleFactor
            new_xis_list = [new_xis[ldx][idx] for ldx in range(len(new_xis))]
            return mass1[idx], mass2[idx], spin1z[idx], spin2z[idx], count, \
                   cDist.min(), new_xis_list
        if (cDist.min() < currDist):
            idx = cDist.argmin()
            bestMasses[0] = totmass[idx]
            bestMasses[1] = eta[idx]
            bestMasses[2] = spin1z[idx]
            bestMasses[3] = spin2z[idx]
            bestChirpmass = bestMasses[0] * (bestMasses[1])**(3./5.)
            currDist = cDist.min()
            unFixedCount = 0
            scaleFactor = origScaleFactor
        count += 1
        unFixedCount += 1
        if unFixedCount > giveUpThresh:
            # Stop at this point
            diff = (bestMasses[0]*bestMasses[0] * (1-4*bestMasses[1]))**0.5
            mass1 = (bestMasses[0] + diff)/2.
            mass2 = (bestMasses[0] - diff)/2.
            new_xis_list = [new_xis[ldx][0] for ldx in range(len(new_xis))]
            return mass1, mass2, bestMasses[2], bestMasses[3], count, \
                   currDist, new_xis_list
        if not unFixedCount % 100:
            scaleFactor *= 2
        if scaleFactor > 64:
            scaleFactor = 1
    # Shouldn't be here!
    raise BrokenError

def get_mass_distribution(bestMasses, scaleFactor, massRangeParams, \
                          metricParams, fUpper, \
                          numJumpPoints=100, chirpMassJumpFac=0.0001, \
                          etaJumpFac=0.01, spin1zJumpFac=0.01, \
                          spin2zJumpFac=0.01):
    """
    Given a set of masses, this function will create a set of points nearby
    in the mass space and map these to the xi space.

    Parameters
    -----------
    bestMasses : list
        Contains [ChirpMass, eta, spin1z, spin2z]. Points will be placed around
        tjos
    scaleFactor : float
        This parameter describes the radius away from bestMasses that points
        will be placed in.
    massRangeParams : massRangeParameters instance
        Instance holding all the details of mass ranges and spin ranges.
    metricParams : metricParameters instance
        Structure holding all the options for construction of the metric
        and the eigenvalues, eigenvectors and covariance matrix
        needed to manipulate the space.
    fUpper : float
        The value of fUpper that was used when obtaining the xi_i
        coordinates. This lets us know how to rotate potential physical points
        into the correct xi_i space. This must be a key in metricParams.evals,
        metricParams.evecs and metricParams.evecsCV
        (ie. we must know how to do the transformation for
        the given value of fUpper)
    numJumpPoints : int, optional (default = 100)
        The number of points that will be generated every iteration
    chirpMassJumpFac : float, optional (default=0.0001)
        The jump points will be chosen with fractional variation in chirpMass
        up to this multiplied by scaleFactor.
    etaJumpFac : float, optional (default=0.01)
        The jump points will be chosen with fractional variation in eta
        up to this multiplied by scaleFactor.
    spin1zJumpFac : float, optional (default=0.01)
        The jump points will be chosen with absolute variation in spin1z up to
        this multiplied by scaleFactor.
    spin2zJumpFac : float, optional (default=0.01)
        The jump points will be chosen with absolute variation in spin2z up to
        this multiplied by scaleFactor.

    Returns 
    --------
    Chirpmass : numpy.array
        chirp mass of the resulting points
    Totmass : numpy.array
        Total mass of the resulting points
    Eta : numpy.array
        Symmetric mass ratio of the resulting points
    Spin1z : numpy.array
        Spin of the heavier body of the resulting points
    Spin2z : numpy.array
        Spin of the smaller body of the resulting points
    Diff : numpy.array
        Mass1 - Mass2 of the resulting points
    Mass1 : numpy.array
        Mass1 (mass of heavier body) of the resulting points
    Mass2 : numpy.array
        Mass2 (mass of smaller body) of the resulting points
    Beta : numpy.array
        1.5PN spin phasing coefficient of the resulting points
    Sigma : numpy.array
        2PN spin phasing coefficient of the resulting points
    Gamma : numpy.array
        2.5PN spin phasing coefficient of the resulting points
    Chis : numpy.array
        0.5 * (spin1z + spin2z) for the resulting points
    new_xis : list of numpy.array
        Position of points in the xi coordinates
    """
    bestChirpmass = bestMasses[0]
    bestEta = bestMasses[1]
    bestSpin1z = bestMasses[2]
    bestSpin2z = bestMasses[3]

    # Firstly choose a set of values for masses and spins
    chirpmass = bestChirpmass * (1 - (numpy.random.random(numJumpPoints)-0.5) \
                                       * chirpMassJumpFac * scaleFactor )
    eta = bestEta * ( 1 - (numpy.random.random(numJumpPoints) - 0.5) \
                           * etaJumpFac * scaleFactor)
    # Note that these two are changed by spinxzFac, *not* spinxzFac/spinxz
    spin1z = bestSpin1z + ( (numpy.random.random(numJumpPoints) - 0.5) \
                            * spin1zJumpFac * scaleFactor)
    spin2z = bestSpin2z + ( (numpy.random.random(numJumpPoints) - 0.5) \
                            * spin2zJumpFac * scaleFactor)

    # Point[0] is always set to the original point
    chirpmass[0] = bestChirpmass
    eta[0] = bestEta
    spin1z[0] = bestSpin1z
    spin2z[0] = bestSpin2z

    # Remove points where eta becomes unphysical
    eta[eta > massRangeParams.maxEta] = massRangeParams.maxEta
    if massRangeParams.minEta:
        eta[eta > massRangeParams.minEta] = massRangeParams.minEta
    else:
        eta[eta < 0.0001] = 0.0001

    # Total mass, masses and mass diff
    totmass = chirpmass / (eta**(3./5.))
    diff = (totmass*totmass * (1-4*eta))**0.5
    mass1 = (totmass + diff)/2.
    mass2 = (totmass - diff)/2.

    # Get the various spin-derived quantities
    beta, sigma, gamma, chis = get_beta_sigma_from_aligned_spins(\
          totmass, eta, spin1z, spin2z)

    # Remove points where spins are outside of physical range
    numploga1 = numpy.logical_and(mass1 > 2,99,\
                                  abs(spin1z) <= massRangeParams.maxBHSpinMag)
    if massRangeParams.nsbhFlag:
        numploga = numploga1
    else:
        numploga2 = numpy.logical_and(mass1 < 3.01, \
                                   abs(spin1z) <= massRangeParams.maxNSSpinMag)
        numploga = numpy.logical_or(numploga1,numploga2)
    numplogb2 = numpy.logical_and(abs(spin2z) <= massRangeParams.maxNSSpinMag,\
                                  mass2 < 3.01)
    if massRangeParams.nsbhFlag:
        numplogb = numplogb2
    else:
        numplogb1 = numpy.logical_and(mass2 > 2.99, \
                                   abs(spin2z) <= massRangeParams.maxBHSpinMag)
        numplogb = numpy.logical_or(numplogb1,numplogb2)
    numplog1 = numpy.logical_and(numploga,numplogb)
    numplog = numpy.logical_not(numplog1)
    if numplog[0] == True:
        raise ValueError("Cannot remove the guide point!")
    beta[numplog] = 0
    sigma[numplog] = 0
    gamma[numplog] = 0
    chis[numplog] = 0
    spin1z[numplog] = 0
    spin2z[numplog] = 0

    # And remove points where the individual masses are outside of the physical
    # range. Or the total masses are.
    totmass[mass1 < massRangeParams.minMass1] = 0.0001
    totmass[mass1 > massRangeParams.maxMass1] = 0.0001
    totmass[mass2 < massRangeParams.minMass2] = 0.0001
    totmass[mass2 > massRangeParams.maxMass2] = 0.0001
    # There is some numerical error which can push this a bit higher. We do
    # *not* want to reject the initial guide point. This error comes from
    # Masses -> totmass, eta -> masses conversion, we will have points pushing
    # onto the boudaries of the space.
    totmass[totmass > massRangeParams.maxTotMass*1.0001] = 0.0001
    totmass[totmass < massRangeParams.minTotMass*0.9999] = 0.0001

    if totmass[0] < 0.00011:
        raise ValueError("Cannot remove the guide point!")

    # Then map to xis
    new_xis = get_cov_params(totmass, eta, beta, sigma, gamma, chis, \
                             metricParams, fUpper)
    return chirpmass, totmass, eta, spin1z, spin2z, diff, mass1, mass2, beta, \
           sigma, gamma, chis, new_xis

def stack_xi_direction_brute(xis, bestMasses, bestXis, direction_num,
                             req_match, massRangeParams, metricParams, fUpper,\
                             scaleFactor=0.8, numIterations=3000):
    """
    This function is used to assess the depth of the xi_space in a specified
    dimension at a specified point in the higher dimensions. It does this by
    iteratively throwing points at the space to find maxima and minima.

    Parameters
    -----------

    xis : list or array
        Position in the xi space at which to assess the depth. This can be only
        a subset of the higher dimensions than that being sampled.
    bestMasses : list
        Contains [totalMass, eta, spin1z, spin2z]. Is a physical position
        mapped to xi coordinates in bestXis that is close to the xis point.
        This is aimed to give the code a starting point.
    bestXis : list
        Contains the position of bestMasses in the xi coordinate system.
    direction_num : int
        The dimension that you want to assess the depth of (0 = 1, 1 = 2 ...)
    req_match : float
        When considering points to assess the depth with, only consider points
        with a mismatch that is smaller than this with xis.
    massRangeParams : massRangeParameters instance
        Instance holding all the details of mass ranges and spin ranges.
    metricParams : metricParameters instance
        Structure holding all the options for construction of the metric
        and the eigenvalues, eigenvectors and covariance matrix
        needed to manipulate the space.
    fUpper : float
        The value of fUpper that was used when obtaining the xi_i
        coordinates. This lets us know how to rotate potential physical points
        into the correct xi_i space. This must be a key in metricParams.evals,
        metricParams.evecs and metricParams.evecsCV
        (ie. we must know how to do the transformation for
        the given value of fUpper)
    scaleFactor : float, optional (default = 0.8)
        The value of the scale factor that is used when calling
        pycbc.tmpltbank.get_mass_distribution.
    numIterations : int, optional (default = 3000)
        The number of times to make calls to get_mass_distribution when
        assessing the maximum/minimum of this parameter space. Making this
        smaller makes the code faster, but at the cost of accuracy.   
 
    Returns
    --------
    xi_min : float
        The minimal value of the specified dimension at the specified point in
        parameter space.
    xi_max : float
       The maximal value of the specified dimension at the specified point in
        parameter space.
    """

    # Find minimum
    ximin = find_xi_extrema_brute(xis, bestMasses, bestXis, direction_num, \
                                  req_match, massRangeParams, metricParams, \
                                  fUpper, find_minimum=True, \
                                  scaleFactor=scaleFactor, \
                                  numIterations=numIterations)
 
    # Find maximum
    ximax = find_xi_extrema_brute(xis, bestMasses, bestXis, direction_num, \
                                  req_match, massRangeParams, metricParams, \
                                  fUpper, find_minimum=False, \
                                  scaleFactor=scaleFactor, \
                                  numIterations=numIterations)

    return ximin, ximax

def find_xi_extrema_brute(xis, bestMasses, bestXis, direction_num, req_match, \
                          massRangeParams, metricParams, fUpper, \
                          find_minimum=False, scaleFactor=0.8, \
                          numIterations=3000):   
    """
    This function is used to find the largest or smallest value of the xi
    space in a specified
    dimension at a specified point in the higher dimensions. It does this by
    iteratively throwing points at the space to find extrema.

    Parameters
    -----------

    xis : list or array
        Position in the xi space at which to assess the depth. This can be only
        a subset of the higher dimensions than that being sampled.
    bestMasses : list
        Contains [totalMass, eta, spin1z, spin2z]. Is a physical position
        mapped to xi coordinates in bestXis that is close to the xis point.
        This is aimed to give the code a starting point.
    bestXis : list
        Contains the position of bestMasses in the xi coordinate system.
    direction_num : int
        The dimension that you want to assess the depth of (0 = 1, 1 = 2 ...)
    req_match : float
        When considering points to assess the depth with, only consider points
        with a mismatch that is smaller than this with xis.
    massRangeParams : massRangeParameters instance
        Instance holding all the details of mass ranges and spin ranges.
    metricParams : metricParameters instance
        Structure holding all the options for construction of the metric
        and the eigenvalues, eigenvectors and covariance matrix
        needed to manipulate the space.
    fUpper : float
        The value of fUpper that was used when obtaining the xi_i
        coordinates. This lets us know how to rotate potential physical points
        into the correct xi_i space. This must be a key in metricParams.evals,
        metricParams.evecs and metricParams.evecsCV
        (ie. we must know how to do the transformation for
        the given value of fUpper)
    find_minimum : boolean, optional (default = False)
        If True, find the minimum value of the xi direction. If False find the
        maximum value.
    scaleFactor : float, optional (default = 0.8)
        The value of the scale factor that is used when calling
        pycbc.tmpltbank.get_mass_distribution.
    numIterations : int, optional (default = 3000)
        The number of times to make calls to get_mass_distribution when
        assessing the maximum/minimum of this parameter space. Making this
        smaller makes the code faster, but at the cost of accuracy.   

    Returns
    --------
    xi_extent : float
        The extremal value of the specified dimension at the specified point in
        parameter space.
    """

    # Setup
    xi_size = len(xis)
    origMasses = copy.deepcopy(bestMasses)
    bestChirpmass = bestMasses[0] * (bestMasses[1])**(3./5.)
    if find_minimum:
        xiextrema = 10000000000
    else:
        xiextrema = -100000000000

    for i in range(numIterations):
        # Evaluate extrema of the xi direction specified
        chirpmass, totmass, eta, spin1z, spin2z, diff, mass1, mass2, beta, \
          sigma, gamma, chis, new_xis = get_mass_distribution(\
               [bestChirpmass,bestMasses[1],bestMasses[2],bestMasses[3]], \
               scaleFactor, massRangeParams, metricParams, fUpper)
        cDist = (new_xis[0] - xis[0])**2
        for j in range(1, xi_size):
            cDist += (new_xis[j] - xis[j])**2
        redCDist = cDist[cDist < req_match]
        redXis = (new_xis[direction_num])[cDist < req_match]
        if len(redCDist):
            if not find_minimum:
                new_xis[direction_num][cDist > req_match] = -10000000
                currXiExtrema = (new_xis[direction_num]).max()
                idx = (new_xis[direction_num]).argmax()
            else:
                new_xis[direction_num][cDist > req_match] = 10000000
                currXiExtrema = (new_xis[direction_num]).min()
                idx = (new_xis[direction_num]).argmin()
            if ( ((not find_minimum) and (currXiExtrema > xiextrema)) or \
                         (find_minimum and (currXiExtrema < xiextrema)) ):
                xiextrema = currXiExtrema
                bestMasses[0] = totmass[idx]
                bestMasses[1] = eta[idx]
                bestMasses[2] = spin1z[idx]
                bestMasses[3] = spin2z[idx]
                m1 = mass1[idx]
                m2 = mass2[idx]
                bestChirpmass = bestMasses[0] * (bestMasses[1])**(3./5.)
    return xiextrema

