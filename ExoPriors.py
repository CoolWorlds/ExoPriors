from __future__ import division
import numpy as np

def LL_penalty_geometric_only(per, rhostar, omega, e):
    """
    One of four possible functions called by transit_LL_penalty. Calculates the log-likelihood penalty for a transit event due to geometric bias only (i.e. does not include
    SNR detection bias).
    
    Inputs:
    per = orbital period
    rhostar = density of host star
    omega = argument of periapsis
    e = eccentricity
    
    Returns:
    Log likelihood penalty, to be added to the log likelihood.
    """
    
    LLP = -(2./3.)*np.log(per) - (1./3.)*np.log(rhostar) + np.log(1. + e*np.sin(omega)) - np.log(1. - e**2)
    
    return LLP
    
def LL_penalty_nongrazing_only(per, rhostar, omega, e, ror, b):
    """
    One of four possible functions called by transit_LL_penalty. Calculates the log-likelihood penalty for a transit event due to geometric and SNR detection bias in the case
    where grazing events are not considered detections.
    
    Inputs:
    per = orbital period
    rhostar = density of host star
    omega = argument of periapsis
    e = eccentricity
    ror = ratio of planet radius to stellar radius
    b = impact parameter
    
    
    Returns:
    Log likelihood penalty, to be added to the log likelihood.
    """

    log_kappa = -np.log(per) - 0.5*np.log(rhostar) - 0.75*np.log(1. - e**2) + 0.5*np.log(1. + e*np.sin(omega))    
    log_eta = np.log(1. + ror) + 2.*np.log(ror) - 0.25*np.log((1. + ror)**2 - b**2) + np.log(np.sqrt((1. + ror)**2 - b**2) + np.sqrt((1. - ror)**2 - b**2))
    
    LLP = log_kappa + log_eta
        
    return LLP
    
def LL_penalty_general(per, rhostar, omega, e, ror, b):
    """
    One of four possible functions called by transit_LL_penalty. Calculates the log-likelihood penalty for a transit event due to geometric and SNR detection bias in the case
    where both non-grazing and grazing events are considered detections.
    
    Inputs:
    per = orbital period
    rhostar = density of host star
    omega = argument of periapsis
    e = eccentricity
    ror = ratio of planet radius to stellar radius
    b = impact parameter
    
    Returns:
    Log likelihood penalty, to be added to the log likelihood.
    """
    
    if b < 0. or (1. + ror) <= b:
        LLP = -np.inf
    
    log_kappa = -np.log(per) - 0.5*np.log(rhostar) - 0.75*np.log(1. - e**2) + 0.5*np.log(1. + e*np.sin(omega))
    
    if (0. <= b) and (b < (1. - ror)):
        log_eta = np.log(1. + ror) + 2.*np.log(ror) - 0.25*np.log((1. + ror)**2 - b**2) + np.log(np.sqrt((1. + ror)**2 - b**2) + np.sqrt((1. - ror)**2 - b**2))
        LLP = log_kappa + log_eta
    
    if ((1. - ror) <= b) and (b < (1. + ror)):
        log_zeta = np.log(1. + ror) + 2.*np.log(1. + ror - b) + 0.25*np.log((1. + ror)**2 - b**2) + np.log(np.log(4.) - np.log(1. + ror - b) + np.log(ror)) - np.log(np.log(16.))
        LLP = log_kappa + log_zeta
    
    return LLP
    
def LL_penalty_occultation(per, rhostar, omega, e, ror, b, docc):
    """
    One of four possible functions called by transit_LL_penalty. Calculates the log-likelihood penalty for a transit event due to geometric and SNR detection bias in the case
    where both non-grazing and grazing events are considered detections.
    
    Inputs:
    per = orbital period
    rhostar = density of host star
    omega = argument of periapsis
    e = eccentricity
    ror = ratio of planet radius to stellar radius
    b = impact parameter
    docc = occultation depth
    
    
    Returns:
    Log likelihood penalty, to be added to the log likelihood.
    """
    
    if b < 0. or (1. + ror) <= b:
        LLP = -np.inf
    
    log_kappa_occ = -np.log(per) - 0.5*np.log(rhostar) - 0.75*np.log(1. - e**2) + 0.5*np.log(1. - e*np.sin(omega))
    
    if (0. <= b) and (b < (1. - ror)):
        log_eta_occ = np.log(1. + ror) + np.log(docc) - 0.25*np.log((1. + ror)**2 - b**2) + np.log(np.sqrt((1. + ror)**2 - b**2) + np.sqrt((1. - ror)**2 - b**2))
        LLP = log_kappa_occ + log_eta_occ
    
    if ((1. - ror) <= b) and (b < (1. + ror)):
        log_zeta_occ = np.log(1. + ror) + 2.*np.log(1. + ror - b) + 0.25*np.log((1. + ror)**2 - b**2) + np.log(np.log(4.) - np.log(1. + ror - b) + np.log(ror)) - np.log(np.log(16.)) + np.log(docc) - 2.*np.log(ror)
        LLP = log_kappa_occ + log_zeta_occ
    
    return LLP

def transit_LL_penalty(case, **kwargs):
    """
    Calculates the log-likelihood penalty for a transit event due to observational bias (geometric and/or SNR detection bias) by calling
    one of the four log-likelihood penalty functions defined above.
    
    There are four possible types of penalty:
    1. case='geometric_only': Only geometric bias is included. (This is equivalent to assuming infinite SNR.)
    2. case='nongrazing_only': Both geometric and SNR detection bias are included. Grazing events are NOT considered detections and are not accounted for.
    3. case='general': Both geometric and SNR detection bias are included. Detected transits may be grazing or non-grazing.
    4. case='occultation': Both geometric and SNR detection bias are included. This is an occultation event instead of a transit.
    
    Inputs:
    case = 'geometric_only', 'nongrazing_only', 'general', or 'occultation'. Specifies one of the four cases outlined above.
    kwargs = {
            per = orbital period of transiting planet
            rhostar = density of host star
            omega = argument of periapsis
            e = eccentricity
            ror (not necessary if case='geometric_only') = ratio of planet radius to stellar radius
            b (not necessary if case='geometric_only') = impact parameter
            occdepth (only necessary if case='occultation') = occultation depth 
            blend (optional) = blend factor = (target_star_flux + blended_source_flux)/target_star_flux 
            }
    """
    #ensure that all necessary variables are defined. If not, return None.
    try:
        per = kwargs.pop('per')
    except KeyError:
        print "Define 'per' !"
        return None
        
    try:
        rhostar = kwargs.pop('rhostar')
    except KeyError:
        print "Define 'rhostar' !"
        return None
    
    try:
        omega = kwargs.pop('omega')
    except KeyError:
        print "Define 'omega' !"
        return None
    
    try:
        e = kwargs.pop('e')
    except KeyError:
        print "Define 'e' !"
        return None
    
    #check to see if a blend factor is defined; if not, set blend factor equal to 1.0 (in which case it will not affect the log likelihood)
    try:
        blend = kwargs.pop('blend')
    except KeyError:
        blend = 1.0
    
    #calculate the appropriate log likelihood penalty for the chosen case
    if case=='geometric_only':
        LL_penalty = LL_penalty_geometric_only(per, rhostar, omega, e)
        
    elif case=='nongrazing_only':
        #ensure that all necessary variables are defined. If not, return None.
        try:
            ror = kwargs.pop('ror')
        except KeyError:
            print "Define 'ror' !"
            return None
        
        try:
            b = kwargs.pop('b')
        except KeyError:
            print "Define 'b' !"
            return None
            
        LL_penalty = LL_penalty_nongrazing_only(per, rhostar, omega, e, ror, b)
        
    elif case=='general':
        #ensure that all necessary variables are defined. If not, return None.
        try:
            ror = kwargs.pop('ror')
        except KeyError:
            print "Define 'ror' !"
            return None
        
        try:
            b = kwargs.pop('b')
        except KeyError:
            print "Define 'b' !"
            return None
            
        LL_penalty = LL_penalty_general(per, rhostar, omega, e, ror, b)
        
    elif case=='occultation':
        #ensure that all necessary variables are defined. If not, return None.
        try:
            ror = kwargs.pop('ror')
        except KeyError:
            print "Define 'ror' !"
            return None
        
        try:
            b = kwargs.pop('b')
        except KeyError:
            print "Define 'b' !"
            return None

        try:
            docc = kwargs.pop('docc')
        except KeyError:
            print "Define 'docc' !"
            return None
        
        LL_penalty = LL_penalty_occultation(per, rhostar, omega, e, ror, b, docc)
        
    else:
        print "Choose one of the four defined cases: 'geometric_only', 'nongrazing_only', 'general', or 'occultation'."
        return
    
    #apply blend factor. If blend factor is not defined by the user, it is set to 1.0, so this line has no effect.
    LL_penalty = LL_penalty - np.log(blend)
    
    return LL_penalty