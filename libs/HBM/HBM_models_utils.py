# -*- coding: utf-8 -*-

import pymc3 as pm
from collections import OrderedDict


def create_poly_modelspec(k=1):
    ''' 
    Convenience function:
    Create a polynomial modelspec string for patsy
    '''
    return ('y ~ 1 + x ' + ' '.join(['+ np.power(x,{})'.format(j) 
                                     for j in range(2, k+1)])).strip()


def run_models(df, upper_order=5):
    ''' 
    Convenience function:
    Fit a range of pymc3 models of increasing polynomial complexity. 
    Suggest limit to max order 5 since calculation time is exponential.
    '''
    
    models, traces = OrderedDict(), OrderedDict()

    for k in range(1, upper_order+1):

        nm = 'k{}'.format(k)
        fml = create_poly_modelspec(k)

        with pm.Model() as models[nm]:

            print('\nRunning: {}'.format(nm))
            pm.glm.GLM.from_formula(fml, df,
                                    priors={'Intercept':pm.Normal.dist(mu=0, sigma=100)},
                                    family=pm.glm.families.Normal())

            traces[nm] = pm.sample(2000)
            
    return models, traces
