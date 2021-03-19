import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import cross_val_score
from sklearn import tree
from sklearn import metrics
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import networkx as nx
from sklearn.model_selection import KFold
from sklearn import tree
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import cross_validate
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.ensemble import ExtraTreesRegressor
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
import matplotlib.pyplot as plt
import scipy.stats  as stats




def linear_regression(X, y):
    
    #X=pd.DataFrame(X)  
    #scale=preprocessing.MinMaxScaler()
    #scale.fit(X)
    #X=scale.transform(X)
    #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.30, random_state=1)
    
    lm1 = LinearRegression()
    lm1.fit(X, y)
    y_pred1 = lm1.predict(X)
    score1=metrics.r2_score(y, y_pred1)

    lm2 = LinearRegression()
    #lm2.fit(X_train, y_train)
   # y_pred1a = lm2.predict(X_train)
   # y_pred2 = lm2.predict(X_test)
   # score2a=metrics.r2_score(y_train, y_pred1a)
   # score2b=metrics.r2_score(y_test, y_pred2)
   # score2={'train': round(score2a, 2), 'test': round(score2b,2)}
   
    y_pred = cross_val_predict(lm2, X, y, cv=5)
    score2=metrics.r2_score(y, y_pred)

    #scores = cross_val_score(lm2, X, y, cv=5)
    #CV_scores={'score': round(scores.mean(), 2), 'score_sd': round(scores.std() * 2,2)}
    

    return round(score1, 2), round(score2, 2)     #CV_scores['score'],CV_scores['score_sd']
    


def lasso_regression(X, y):
    
    #X=pd.DataFrame(X)   
    #scale=preprocessing.MinMaxScaler()
    #scale.fit(X)
    #X=scale.transform(X)
   # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.30, random_state=1)
   

    lm1 = Lasso()
    lm1.fit(X, y)
    y_pred1 = lm1.predict(X)
    score1=metrics.r2_score(y, y_pred1)

    lm2 = Lasso()
    #lm2.fit(X_train, y_train)
    #y_pred1a = lm2.predict(X_train)
    #y_pred2 = lm2.predict(X_test)
   # score2a=metrics.r2_score(y_train, y_pred1a)
   # score2b=metrics.r2_score(y_test, y_pred2)
   # score2={'train': round(score2a, 2), 'test': round(score2b,2)}
    y_pred = cross_val_predict(lm2, X, y, cv=5)
    score2=metrics.r2_score(y, y_pred)    
    
    #scores = cross_val_score(lm2, X, y, cv=5)
    #CV_scores={'score': round(scores.mean(), 2), 'score_sd': round(scores.std() * 2,2)}
    

    return round(score1, 2),round(score2, 2)     #CV_scores['score'],CV_scores['score_sd']


def random_forest(X, y):
    
    #X=pd.DataFrame(X)   
    #scale=preprocessing.MinMaxScaler()
    #scale.fit(X)
    #X=scale.transform(X)
   # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.30, random_state=1)

    lm1 = RandomForestRegressor()
    lm1.fit(X, y.values.ravel())
    y_pred1 = lm1.predict(X)
    score1=metrics.r2_score(y, y_pred1)

    lm2 = RandomForestRegressor()
    #lm2.fit(X_train, y_train.values.ravel())
    #y_pred1a = lm2.predict(X_train)
    #y_pred2 = lm2.predict(X_test)
   # score2a=metrics.r2_score(y_train, y_pred1a)
    #score2b=metrics.r2_score(y_test, y_pred2)
    #score2={'train': round(score2a, 2), 'test': round(score2b,2)}
    y_pred = cross_val_predict(lm2, X, y.values.ravel(), cv=5)
    score2=metrics.r2_score(y, y_pred)    
    
    #scores = cross_val_score(lm2, X, y, cv=5)
    #CV_scores={'score': round(scores.mean(), 2), 'score_sd': round(scores.std() * 2,2)}
 
    return round(score1, 2), round(score2, 2)     #CV_scores['score'],CV_scores['score_sd']
 



def reading(age,group, sign, test, criterion, n_groups, file):
    s = open('res/subgraph_under_{}/{}/{}/{}/criterion_{}/n_groups_{}/{}'.format(age, group, sign, test, criterion, n_groups, file), 'r').read()
    sel=eval(s)
    return sel



def open_file(file, limit, binary, absolute):
    
    mat = np.array(pd.read_csv(file, header = None, sep = ","))
    #mat=np.array(mat)
        
    if limit:
        #threshold = np.percentile(mat[np.tril_indices_from(abs(mat), k=-1)], limit)
        threshold = np.percentile(abs(mat), limit)
        mat[abs(mat)<threshold] = 0
 
    if binary:
        mat[abs(mat)>=threshold] = 1

    return mat



def whole_graph_features(subj_id, test, age,group, sign, nodes=False, binary=True, absolute=None):
    
   subj = open_file('datasets/under_{}/{}/{}/{}/main/{}.csv'.format(age, group, sign, test, subj_id), 80,binary, absolute)
   
   df_dict={}

   if nodes==False:
       g=nx.from_numpy_matrix(subj)
       df_dict['edges']=np.sum(np.tril(np.array(subj), k = -1))
       df_dict['avg_clus']=nx.average_clustering(g)
       try:               
           df_dict['diameter']=nx.algorithms.distance_measures.diameter(g)
       except:
           df_dict['diameter']=1000
   else:
       g=nx.from_numpy_matrix(subj)
       df_dict.update(dict(zip(['degree_'+str(n) for n in sorted(g.nodes())],
                                dict(nx.degree(g)).values())))
       df_dict.update(dict(zip(['clus_'+str(n) for n in sorted(g.nodes())], 
                                nx.clustering(g).values())))
       df_dict.update(zip(['bet_'+'_'+str(n) for n in sorted(g.nodes())], 
              nx.betweenness_centrality(g, normalized=True).values()))
       df_dict.update(dict(zip(['close_'+'_'+str(n) for n in sorted(g.nodes())], 
                                nx.closeness_centrality(g).values())))
              
   return df_dict


def edges(subj_id, test, age, group, sign, criterion, n_groups, contrasts, alpha, binary=True, absolute=None):

    subj = open_file('datasets/under_{}/{}/{}/{}/main/{}.csv'.format(age, group, sign, test, subj_id), 80,binary, absolute)
    
    df_dict = {}
    df_dict['edges_'+i[:-4]]=np.sum(np.tril(np.array(subj)[np.ix_(nodes, nodes)], k = -1))
    
    return df_dict

def csubgraph_features(subj_id, test, age, group, sign, criterion, n_groups, contrasts, alpha, 
                   nodes=False, binary=True, absolute=None):

    subj = open_file('datasets/under_{}/{}/{}/{}/main/{}.csv'.format(age, group, sign, test, subj_id), 80,binary, absolute)
   
    df_dict={}

    if nodes==False:
       for i in contrasts:
           nodes=reading(age, group, sign, test,criterion, n_groups, i)[alpha]
           g=nx.from_numpy_matrix(np.tril(np.array(subj)[np.ix_(nodes, nodes)], k = -1))
           df_dict['edges_'+i[:-4]]=np.sum(np.tril(np.array(subj)[np.ix_(nodes, nodes)], k = -1))
           df_dict['avg_clus_'+i[:-4]]=nx.average_clustering(g)
           try:               
               df_dict['diameter_'+i[:-4]]=nx.algorithms.distance_measures.diameter(g)
           except:
               df_dict['diameter_'+i[:-4]]=1000           

    else:
       for i in contrasts:
           nodes=reading(age, test,criterion, n_groups, i)[alpha]
           g=nx.from_numpy_matrix(np.tril(np.array(subj)[np.ix_(nodes, nodes)], k = -1))
           df_dict.update(dict(zip(['degree_'+i[:-4]+'_'+str(n) for n in sorted(nodes)],dict(nx.degree(g)).values())))
           df_dict.update(dict(zip(['clus_'+i[:-4]+'_'+str(n) for n in sorted(nodes)], 
                                    nx.clustering(g).values())))         
           df_dict.update(zip(['bet_'+i[:-4]+'_'+str(n) for n in sorted(nodes)], 
                  nx.betweenness_centrality(g, normalized=True).values()))
           df_dict.update(dict(zip(['close_'+i[:-4]+'_'+str(n) for n in sorted(nodes)], 
                                    nx.closeness_centrality(g).values())))
       
    return df_dict





def transform_data(df, scale=False, normalize=False, min_max=False):
    
    if scale:
        X=preprocessing.scale(df)
    
    #elif normalize:
     #   if len(df.shape)==1:
     #       X=np.asarray(preprocessing.normalize((np.asarray(df).reshape(1,-1)))).reshape(-1,1)
      #  else:    
       #     X=preprocessing.normalize(df)
    elif min_max:
        X=preprocessing.minmax_scale(df)
    else:
        X=df

    return pd.DataFrame(X, index=df.index, columns=df.columns)


def linear_reg_CV5_scatter(X,y,test, corr=False,feature_imp=False, save_plot=False, alpha=None, group=None, sign=None):
    i=0
    test_scores=[]
    train_scores=[]
    kf=KFold(n_splits=5, random_state=3, shuffle=True)
    for k,j in kf.split(X):
        i+=1
        if len(X.shape)==1:
            
            X_train, X_test = X[k], X[j]
            y_train, y_test = y[k], y[j]
            lm = LinearRegression()
            lm.fit(pd.DataFrame(X_train), y_train.ravel())
            
            train_pred=lm.predict(pd.DataFrame(X_train))
            test_pred=lm.predict(pd.DataFrame(X_test))            
        else:
            X_train, X_test = X.iloc[k, :], X.iloc[j,:]
            y_train, y_test = y.iloc[k], y.iloc[j]
            lm = LinearRegression()
            lm.fit(X_train, y_train.values.ravel())

            train_pred=lm.predict(X_train)
            test_pred=lm.predict(X_test)
            
        r_train, p_train=stats.pearsonr(y_train, train_pred)
        r_test, p_test=stats.pearsonr(y_test, test_pred)

        train_r2=round(metrics.r2_score(y_train, train_pred),2)
        test_r2=round(metrics.r2_score(y_test, test_pred),2)
        train_scores.append(train_r2)
        test_scores.append(test_r2)
        fig, (ax1, ax2) = plt.subplots(1,2)
        fig.suptitle('Predicted vs True values in cross validation - CV fold {}'.format(i))
        ax1.set_title("Train set")
        ax1.set_xlabel("{}".format(test))
        ax1.set_ylabel("Predicted")
        ax1.scatter(y_train, train_pred,s=7)
        ax1.plot(y_train, y_train, 'g')
        ax1.text(y_train.min(), train_pred.max(), r'$R^2=${}'.format(train_r2), fontsize=9)
        if corr==True:
            ax1.text(y_train.min(), train_pred.max()-2, r'r={}, p={}'.format(round(r_train,2), round(p_train,3)), fontsize=9)

        ax2.set_title("Test set")
        ax2.set_xlabel("{}".format(test))
        ax2.scatter(y_test, test_pred, s=7)
        ax2.plot(y_test, y_test, 'c')
        ax2.text(y_test.min(), test_pred.max()-1, r'$R^2=${}'.format(test_r2), fontsize=9)
        if corr==True:
            ax2.text(y_test.min(), test_pred.max()-2, r'r={}, p={}'.format(round(r_test,2), round(p_test,3)), fontsize=9) 
        if save_plot:
            plt.savefig("images/lm_{}_{}_{}_{}_train_test_CV{}.png".format(alpha,test,group,sign, i))
        plt.show()
        if feature_imp==True:
            importance = lm.coef_
            fig, ax = plt.subplots()
            # get importance
            # plot feature importance
            ax.bar([x for x in range(len(importance))], importance)
            ax.set_ylabel('Scores')
            ax.set_title('Features importance scores - Cv fold {}'.format(i))
            ax.set_xticks(np.arange(0,len(X.columns),1))
            ax.set_xticklabels([p[6:] for p in X.columns], rotation=90)
            plt.show()

    print(r'average train-cv R2:', round(np.mean(train_scores),2),r'average test-cv R2:', round(np.mean(test_scores),2))

def random_forest_CV5_scatter(X,y,test, tree_plot=False, save_tree_plot=False, feature_imp=False, save_plot=False, alpha=None, group=None, sign=None):
    i=0
    kf=KFold(n_splits=5, random_state=3, shuffle=True)
    train_scores=[]
    test_scores=[]
    for k,j in kf.split(X):
        i+=1
        X_train, X_test = X.iloc[k, :], X.iloc[j,:]
        y_train, y_test = y.iloc[k], y.iloc[j]
        rf = RandomForestRegressor(max_depth=5, min_samples_leaf=5)
        rf.fit(X_train, y_train.values.ravel())
        train_pred=rf.predict(X_train)
        test_pred=rf.predict(X_test)

        train_r2=round(metrics.r2_score(y_train, train_pred),2)
        test_r2=round(metrics.r2_score(y_test, test_pred),2)
        train_scores.append(train_r2)
        test_scores.append(test_r2)   
        fig, (ax1, ax2) = plt.subplots(1,2)
        fig.suptitle('Predicted vs True values in cross validation - CV fold {}'.format(i))
        ax1.set_title("Train set")
        ax1.set_xlabel("{}".format(test))
        ax1.set_ylabel("Predicted")
        ax1.scatter(y_train, train_pred,s=7)
        ax1.plot(y_train, y_train, 'g')
        ax1.text(y_train.min(), train_pred.max(), r'$R^2=${}'.format(train_r2), fontsize=9)

        ax2.set_title("Test set")
        ax2.set_xlabel("{}".format(test))
        ax2.scatter(y_test, test_pred, s=7)
        ax2.plot(y_test, y_test, 'c')
        ax2.text(y_test.min(), test_pred.max(), r'$R^2=${}'.format(test_r2), fontsize=9)
        if save_plot:
            plt.savefig("images/rf_{}_{}_{}_{}_train_test_CV{}.png".format(alpha,test,group,sign, i))
        plt.show()
        
        # plot feature importance
        if feature_imp:
            importance = rf.feature_importances_
            fig, ax = plt.subplots()
            ax.bar([x for x in range(len(importance))], importance)
            ax.set_ylabel('Scores')
            ax.set_title('Features importance scores - Cv fold {}'.format(i))
            ax.set_xticks(np.arange(0,20,1))
            ax.set_xticklabels([p[6:] for p in X_train.columns], rotation=90)
            #plt.savefig("images/importance_features_CV{}.png".format(i))
            plt.show()
        if tree_plot==True:
            fig, axes = plt.subplots(nrows = 1,ncols = 1,figsize = (8,8), dpi=256)
            tree.plot_tree(rf.estimators_[0],
                       feature_names = X_train.columns, 
                       filled = True)
            if save_plot:
                plt.savefig("images/rf_tree_{}_{}_{}_{}_train_test_CV{}.png".format(alpha,test,group,sign, i))

            plt.show()
    print(r'average train-cv R2:', round(np.mean(train_scores),2),r'average test-cv R2:', round(np.mean(test_scores),2))
        
def lm_train_test(X,y,test):
    # split in train and test and fit the linear model
    X_=sm.add_constant(X)
    X_train, X_test, y_train, y_test = train_test_split(X_, y, 
                                                        test_size = 0.30, random_state=1) 

    results = sm.OLS(y_train, X_train).fit()
    vif = pd.DataFrame()
    vif["features"] = X.columns
    vif["VIF Factor"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]


    print(vif.round(1))
    
    # predict over both train and test data and compute test residuals
    X_test=sm.add_constant(X_test)
    prediction = results.predict(X_test)
    x_pred=results.predict(X_train)
    residual = y_test - prediction

    print(results.summary())
    
    train_r2=round(metrics.r2_score(y_train, x_pred),2)
    test_r2=round(metrics.r2_score(y_test, prediction),2)

    fig, (ax1, ax2) = plt.subplots(1,2)
    fig.suptitle('Predicted vs True values')
    ax1.set_title("Train set")
    ax1.set_xlabel("{}".format(test))
    ax1.set_ylabel("Predicted")
    ax1.scatter(y_train, x_pred, s=7)
    ax1.plot(y_train, y_train, 'g')
    ax1.text(y_train.min(), x_pred.max(), r'$R^2=${}'.format(train_r2), fontsize=9)

    ax2.set_title("Test set")
    ax2.set_xlabel("{}".format(test))
    ax2.scatter(y_test, prediction, s=7)
    ax2.plot(y_test, y_test, 'c')
    ax2.text(y_test.min(), prediction.max(), r'$R^2=${}'.format(test_r2), fontsize=9)

    #plt.savefig("data_cs_images/model1_train_test.png")
    plt.show()

def get_covariates(df):
    X_pos=df['pos']
    X_neg=df['neg']
    X=df.drop(['y'], axis=1)
    y=df['y']
    
    return X_pos, X_neg, X, y


