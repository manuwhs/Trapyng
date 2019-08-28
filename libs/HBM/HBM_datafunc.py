
import numpy as np
import pandas as pd

rndst = np.random.RandomState(0)

def generate_data(n=20, p=0, a=1, b=1, c=0, latent_sigma_y=20):
    ''' 
    Create a toy dataset based on a very simple model that we might
    imagine is a noisy physical process:
        1. random x values within a range
        2. latent error aka inherent noise in y
        3. optionally create labelled outliers with larger noise

    Model form: y ~ a + bx + cx^2 + e
    
    NOTE: latent_sigma_y is used to create a normally distributed,
    'latent error' aka 'inherent noise' in the 'physical' generating
    process, rather than experimental measurement error. 
    Please don't use the returned `latent_error` values in inferential 
    models, it's returned in the dataframe for interest only.
    '''
    
    df = pd.DataFrame({'x':rndst.choice(np.arange(100), n, replace=False)})
                
    ## create linear or quadratic model
    df['y'] = a + b*(df['x']) + c*(df['x'])**2 

    ## create latent noise and marked outliers
    df['latent_error'] = rndst.normal(0, latent_sigma_y, n)
    df['outlier_error'] = rndst.normal(0, latent_sigma_y*10, n)
    df['outlier'] = rndst.binomial(1, p, n)
    
    ## add noise, with extreme noise for marked outliers
    df['y'] += ((1-df['outlier']) * df['latent_error'])
    df['y'] += (df['outlier'] * df['outlier_error'])
   
    ## round
    for col in ['y','latent_error','outlier_error','x']:
        df[col] = np.round(df[col],3)
       
    ## add label
    df['source'] = 'linear' if c == 0 else 'quadratic'
    
    ## create simple linspace for plotting true model
    plotx = np.linspace(df['x'].min() - np.ptp(df['x'])*.1, 
                        df['x'].max() + np.ptp(df['x'])*.1, 100)

    ploty = a + b * plotx + c * plotx ** 2
    dfp = pd.DataFrame({'x':plotx, 'y':ploty})
        
    return df, dfp