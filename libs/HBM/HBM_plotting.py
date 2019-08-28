import pymc3 as pm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from HBM_datafunc import generate_data

def plot_datasets(df_lin, df_quad, dfp_lin, dfp_quad):
    '''
    Convenience function:
    Plot the two generated datasets in facets with generative model
    '''
    
    df = pd.concat((df_lin, df_quad), axis=0)
   
    g = sns.FacetGrid(col='source', hue='source', data=df, size=6,
                      sharey=False, legend_out=False)

    g.map(plt.scatter, 'x', 'y', alpha=0.7, s=100, lw=2, edgecolor='w')

    g.axes[0][0].plot(dfp_lin['x'], dfp_lin['y'], '--', alpha=0.6)
    g.axes[0][1].plot(dfp_quad['x'], dfp_quad['y'], '--', alpha=0.6)
                
        
def plot_traces(traces, retain=1000):
    ''' 
    Convenience function:
    Plot traces with overlaid means and values
    '''
    
    ax = pm.traceplot(traces[-retain:], figsize=(12,len(traces.varnames)*1.5),
        lines={k: v['mean'] for k, v in pm.summary(traces[-retain:]).iterrows()})

    for i, mn in enumerate(pm.summary(traces[-retain:])['mean']):
        ax[i,0].annotate('{:.2f}'.format(mn), xy=(mn,0), xycoords='data',
                         xytext=(5,10), textcoords='offset points', rotation=90,
                         va='bottom', fontsize='large', color='#AA0022')


def interact_dataset(n=20, p=0, a=-30, b=5, c=0, latent_sigma_y=20):
    ''' 
    Convenience function:
    Interactively generate dataset and plot
    '''
    
    df, dfp = generate_data(n, p, a, b, c, latent_sigma_y)

    g = sns.FacetGrid(df, size=8, hue='outlier', hue_order=[True,False]
                    ,palette=sns.color_palette('Set1'), legend_out=False)

    g.map(plt.errorbar, 'x', 'y', 'latent_error', marker="o",
          ms=10, mec='w', mew=2, ls='', elinewidth=0.7).add_legend()

    plt.plot(dfp['x'], dfp['y'], '--', alpha=0.8)

    plt.subplots_adjust(top=0.92)
    g.fig.suptitle('Sketch of Data Generation ({})'.format(df['source'][0]), fontsize=16)


def plot_posterior_cr(models, traces, rawdata, xlims,
                      datamodelnm='linear', modelnm='k1'):
    '''
    Convenience function:
    Plot posterior predictions with credible regions shown as filled areas.
    '''
    
    ## Get traces and calc posterior prediction for npoints in x
    npoints = 100
    mdl = models[modelnm]
    trc = pm.trace_to_dataframe(traces[modelnm][-1000:])
    trc = trc[[str(v) for v in mdl.cont_vars[:-1]]]

    ordr = int(modelnm[-1:])
    x = np.linspace(xlims[0], xlims[1], npoints).reshape((npoints,1))
    pwrs = np.ones((npoints,ordr+1)) * np.arange(ordr+1)
    X = x ** pwrs
    cr = np.dot(X, trc.T)

    ## Calculate credible regions and plot over the datapoints
    dfp = pd.DataFrame(np.percentile(cr,[2.5, 25, 50, 75, 97.5], axis=1).T,
                       columns=['025','250','500','750','975'])
    dfp['x'] = x

    pal = sns.color_palette('Greens')
    f, ax1d = plt.subplots(1,1, figsize=(7,7))
    f.suptitle('Posterior Predictive Fit -- Data: {} -- Model: {}'.format(datamodelnm,
                                                                          modelnm), fontsize=16)
    plt.subplots_adjust(top=0.95)

    ax1d.fill_between(dfp['x'], dfp['025'], dfp['975'], alpha=0.5,
                      color=pal[1], label='CR 95%')
    ax1d.fill_between(dfp['x'], dfp['250'], dfp['750'], alpha=0.5,
                      color=pal[4], label='CR 50%')
    ax1d.plot(dfp['x'], dfp['500'], alpha=0.6, color=pal[5], label='Median')
    
    plt.legend()
    ax1d.set_xlim(xlims)
    sns.regplot(x='x', y='y', data=rawdata, fit_reg=False,
                scatter_kws={'alpha':0.7,'s':100, 'lw':2,'edgecolor':'w'}, ax=ax1d)
    
### Exponential Time Series ###

def plot_lambda_func(trace,count_data ):
    n_count_data = len(count_data)
    lambda_1_samples = trace['lambda_1']
    lambda_2_samples = trace['lambda_2']
    tau_samples = trace['tau']
    
    fig = plt.figure(figsize=(20, 3))
    
    # tau_samples, lambda_1_samples, lambda_2_samples contain
    # N samples from the corresponding posterior distribution
    N = tau_samples.shape[0]
    expected_texts_per_day = np.zeros(n_count_data)
    
    for day in range(0, n_count_data):
        # ix is a bool index of all tau samples corresponding to
        # the switchpoint occurring prior to value of 'day'
        ix = day < tau_samples
        
        # Each posterior sample corresponds to a value for tau.
        # for each day, that value of tau indicates whether we're "before"
        # (in the lambda1 "regime") or
        #  "after" (in the lambda2 "regime") the switchpoint.
        
        # by taking the posterior sample of lambda1/2 accordingly, we can average
        # over all samples to get an expected value for lambda on that day.
        
        expected_texts_per_day[day] = (lambda_1_samples[ix].sum()
                                       + lambda_2_samples[~ix].sum()) / N
    
    for i in range(100):
        tau_i = tau_samples[i]; lambda1_i = lambda_1_samples[i];  lambda2_i = lambda_2_samples[i]; 
        # ix is a bool index of all tau samples corresponding to
        # the switchpoint occurring prior to value of 'day'
        ix = tau_i > np.arange(n_count_data)
        posterior_lambda_sample = np.zeros(n_count_data)
        posterior_lambda_sample[ix] = lambda1_i # 
        posterior_lambda_sample[~ix] = lambda2_i # 
        plt.plot(range(n_count_data), posterior_lambda_sample, lw=1, color="#E24A33",alpha = 0.1)
    
    plt.plot(range(n_count_data), expected_texts_per_day, lw=2, color="b",
             label="expected number of text-messages received")
    
    
    plt.xlim(0, n_count_data)
    plt.xlabel("Day")
    plt.ylabel("Expected # text-messages")
    plt.title("Expected number of text-messages received")
    plt.ylim(0, 60)
    plt.bar(np.arange(len(count_data)), count_data, color="#348ABD", alpha=0.65,
            label="observed texts per day")
    
    plt.legend(loc="upper left");

