import plotnine as p9
import numpy as np

def plot_chain_sk(location,size,i,id):
    ite = 1
    la_array = np.loadtxt(location+str(id)+'_bach'+str(ite)+'.txt', delimiter=',')
    la_array= pd.DataFrame(la_array)
    for ite in np.arange(1,size):
        la_array = pd.concat([la_array,
                           pd.DataFrame(np.loadtxt(location+str(id)+'_bach'+str(ite)+'.txt', delimiter=','))], axis = 1)
    la_array = la_array.iloc[[i,30+i]]
    la_array = la_array.transpose().reset_index(drop=True)
    la_array = la_array.unstack().reset_index()
    la_array.columns = ['parameter','sim','value']
    la_array['parameter'] = la_array['parameter'].astype(str)
    lim = [la_array['value'].min()*0.995, la_array['value'].max()*1.005]
    fig = p9.ggplot(la_array, p9.aes(x='sim',y='value' , color = 'parameter'))
    fig = fig + p9.geom_line()+p9.scale_y_continuous(limits = (lim[0],lim[1]))
    return fig



def plot_chain_cj(location,size,i):
    ite = 1
    la_array = np.loadtxt(location+str(id)+'_bach'+str(ite)+'.txt', delimiter=',')
    la_array= pd.DataFrame(la_array)
    for ite in np.arange(1,size):
        la_array = pd.concat([la_array,
                           pd.DataFrame(np.loadtxt(location+str(id)+'_bach'+str(ite)+'.txt', delimiter=','))], axis = 0)
    la_array = la_array.iloc[:,i].reset_index(drop=True)
    la_array = la_array.reset_index(drop=False)
    la_array = la_array.reset_index(drop=True)
    la_array.columns = ['sim','value']
    #la_array['parameter'] = la_array['parameter'].astype(str)
    lim = [la_array['value'].min()*0.995, la_array['value'].max()*1.005]
    fig = (
           p9.ggplot(la_array,p9.aes(x='sim',y='value'))+
           p9.geom_line()+p9.scale_y_continuous(limits = (lim[0],lim[1]))
    )
    return fig


def plot_chain_tht(location,size,i):
    ite = 1
    la_array = np.loadtxt(location+str(id)+'_bach'+str(ite)+'.txt', delimiter=',')
    la_array= pd.DataFrame(la_array)
    for ite in np.arange(1,size):
        la_array = pd.concat([la_array,
                           pd.DataFrame(np.loadtxt(location+str(id)+'_bach'+str(ite)+'.txt', delimiter=','))], axis = 1)
    la_array = la_array.iloc[i]
    la_array = la_array.transpose().reset_index(drop=True)
    la_array = la_array.reset_index(drop=False)
    la_array.columns = ['sim','value']
    #la_array['parameter'] = la_array['parameter'].astype(str)
    lim = [la_array['value'].min()*0.995, la_array['value'].max()*1.005]
    fig = (
           p9.ggplot(la_array,p9.aes(x='sim',y='value'))+
           p9.geom_line()+p9.scale_y_continuous(limits = (lim[0],lim[1]))
    )
    return fig


def plot_chain_phi(location,size,i):
    ite = 1
    la_array = np.loadtxt(location+str(id)+'_bach'+str(ite)+'.txt', delimiter=',')
    la_array= pd.DataFrame(la_array)
    for ite in np.arange(1,size):
        la_array = pd.concat([la_array,
                           pd.DataFrame(np.loadtxt(location+str(id)+'_bach'+str(ite)+'.txt', delimiter=','))], axis = 1)
    la_array = la_array.iloc[i]
    la_array = la_array.transpose().reset_index(drop=True)
    la_array = la_array.reset_index(drop=False)
    la_array.columns = ['sim','value']
    #la_array['parameter'] = la_array['parameter'].astype(str)
    lim = [la_array['value'].min()*0.995, la_array['value'].max()*1.005]
    fig = (
           ggplot(la_array,aes(x='sim',y='value'))+
           geom_line()+scale_y_continuous(limits = (lim[0],lim[1]))
    )
    return fig
