from .model import Model

import pandas as pd
import numpy as np

from sklearn.cluster import KMeans
from yellowbrick.cluster.elbow import kelbow_visualizer

#from sklearn.dummy import DummyClassifier
#from sklearn.linear_model import LinearRegression, LogisticRegression
#from lightgbm import LGBMClassifier
#from sklearn.neighbors import KNeighborsClassifier
#from sklearn.tree import DecisionTreeClassifier
#from sklearn.tree import DecisionTreeClassifier
#from sklearn.ensemble import RandomForestClassifier
#from sklearn.ensemble import AdaBoostClassifier
#from sklearn.ensemble import GradientBoostingClassifier
#from sklearn.naive_bayes import GaussianNB, MultinomialNB
#from sklearn.neural_network import MLPClassifier
#import xgboost as xgb

import lazypredict
from lazypredict.Supervised import Classification, LazyClassifier, LazyRegressor, Regression

from sklearn.model_selection import KFold, cross_val_score, train_test_split, GridSearchCV, cross_validate
from sklearn.metrics import classification_report, accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
from ...multiscorer_master.multiscorer import MultiScorer

from sklearn.decomposition import PCA

from pandas.core.base import PandasObject

from imblearn.under_sampling import ClusterCentroids, OneSidedSelection
from imblearn.over_sampling import SMOTE, BorderlineSMOTE, ADASYN, RandomOverSampler
from imblearn.combine import SMOTEENN, SMOTETomek

from collections import Counter

models = [RandomOverSampler, BorderlineSMOTE, ClusterCentroids, OneSidedSelection, ADASYN, SMOTE, SMOTEENN, SMOTETomek]

class Analytics(Model):

    #def __init__(self):
    #    pass

    def __print_elbow(self, k, visualizer=None):
        self.k = k

        self.visualizer = kelbow_visualizer(KMeans(), self, self.k)
        self.visualizer.show()

    PandasObject.print_elbow = __print_elbow

#----------------------------------------------------------

    def __order_cluster(self, target_field_name, cluster_field_name, ascending):    
        # Add the string "new_" to cluster_field_name
        new_cluster_field_name = "new_" + cluster_field_name
    
        # Create a new dataframe by grouping the input dataframe by cluster_field_name and extract target_field_name 
        # and find the mean
        df_new = self.groupby(cluster_field_name)[target_field_name].mean().reset_index()
    
        # Sort the new dataframe df_new, by target_field_name in descending order
        df_new = df_new.sort_values(by=target_field_name, ascending=ascending).reset_index(drop=True)
    
        # Create a new column in df_new with column name index and assign it values to df_new.index
        df_new["index"] = df_new.index
    
        # Create a new dataframe by merging input dataframe df and part of the columns of df_new based on 
        # cluster_field_name
        df_final = pd.merge(self, df_new[[cluster_field_name, "index"]], on=cluster_field_name)
    
        # Update the dataframe df_final by deleting the column cluster_field_name
        df_final = df_final.drop([cluster_field_name], axis=1)
    
        # Rename the column index to cluster_field_name
        df_final = df_final.rename(columns={"index": cluster_field_name})
    
        return df_final

    PandasObject.order_cluster = __order_cluster

#----------------------------------------------------------    

    def get_models_scores(self, X_train, X_test, y_train, y_test, type=0):
      
        """
            type 0 = Classification
            type 1 = Regression

        """

        if (type == 1):
            reg = LazyRegressor(verbose=0, ignore_warnings=True, custom_metric=None)
            models, predictions = reg.fit(X_train, X_test, y_train, y_test)
        else:
            clf = LazyClassifier(verbose=0, ignore_warnings=True, custom_metric=None)
            models, predictions = clf.fit(X_train, X_test, y_train, y_test)

        return models

#----------------------------------------------------------

    def print_models_scores(self, X_train, X_test, y_train, y_test, type=0):
        print(self.get_models_scores(self, X_train, X_test, y_train, y_test, type))

#----------------------------------------------------------

    def __get_balanded_scores(self, X, y):
        models = [RandomOverSampler, BorderlineSMOTE, ClusterCentroids, OneSidedSelection, ADASYN, SMOTE, SMOTEENN, SMOTETomek]

        df_results = pd.DataFrame()

        for model in models:

            x_res, y_res = model().fit_resample(X, y)

            models_scores = self.__get_models_scores(self, X, y)

            best_model = models_scores.head(1)

            model_scores_dict = {
                                'Balancing_Model_name' : []
                              , 'Shape' : []
                                }

            model_scores_dict['Balancing_Model_name'].append(model.__name__)

            model_scores_dict['Shape'].append('{}'.format(Counter(y_res)))

            df_model_score = pd.DataFrame(model_scores_dict)

            df_model_score = pd.concat([df_model_score, best_model], axis=1)

            df_results = pd.concat([df_results, df_model_score], axis=0)

        return df_results.sort_values(by=["Accuracy", "F1_score", "Recall", "Precision"], ignore_index=True, ascending=False)

#----------------------------------------------------------

    def print_balanded_models_scores(self, X, y):
        print(self.__get_balanded_scores(self, X, y), end='')

#----------------------------------------------------------
"""
    def __get_models():        
        models = []
        models.append(("DummyClassifier_most_frequent", DummyClassifier(strategy='most_frequent', random_state=0)))
        models.append(("LinearRegression", LinearRegression()))
        models.append(("LogisticRegression", LogisticRegression()))
        models.append(("LGBMClassifier", LGBMClassifier()))       
        models.append(("KNeighborsClassifier", KNeighborsClassifier(3)))
        models.append(("DecisionTreeClassifier", DecisionTreeClassifier()))
        models.append(("RandomForestClassifier", RandomForestClassifier()))
        models.append(("AdaBoostClassifier", AdaBoostClassifier()))
        models.append(("GradientBoostingClassifier", GradientBoostingClassifier()))
        models.append(("NaiveBayesGaussian", GaussianNB()))
        models.append(("NaiveBayesMultinomialNB", MultinomialNB()))
        models.append(("MultiLayerPerceptronClassifier", MLPClassifier()))
        models.append(("XGBClassifier", xgb.XGBClassifier(eval_metric='mlogloss')))
        return models

#----------------------------------------------------------

    def __get_metrics():
        # Measuring the metrics of the different models
        scorer = MultiScorer({'Accuracy'  : (accuracy_score , {})
                            , 'F1_score'  : (f1_score       , {'pos_label': 3, 'average':'macro'})
                            , 'Recall'    : (recall_score   , {'pos_label': 3, 'average':'macro'})
                            , 'Precision' : (precision_score, {'pos_label': 3, 'average':'macro'})
                            })
        return scorer

#----------------------------------------------------------
    def __get_models_scores():
        model_scores_dict = {'Model_name' : []
                           , 'Accuracy'   : []
                           , 'F1_score'   : []
                           , 'Recall'     : []
                           , 'Precision'  : []
                            }
        return model_scores_dict
"""        