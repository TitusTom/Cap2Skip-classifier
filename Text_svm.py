#Imports

import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, accuracy_score, mean_squared_error, confusion_matrix

#constants/globals
input_filename   = "input_file.csv"
vector_path      = "./textVectors/"
train_size       = 7000
test_count_start = 7000
dataset_size     = 10000
vector_length    = 4800
vector_number    = 1


#Load Ground truth file
def gtFileLoad(GT_filename):
    
    ground_truth_file = pd.read_csv(GT_filename, names \
    =['name', 'captions','ground_truth' ,'arousal'], skiprows=1)
    name_modifier_val = lambda x: vector_path+x+'.npy'
    ground_truth_names_val  = ground_truth_file['name'].apply(name_modifier_val)
    print 'GT file  loaded'
    return ground_truth_names_val, ground_truth_file['ground_truth']


def classifySVM(ground_truth_names, ground_truth_values):
     
     print "Creating Training and Validation splits"
     X_train = ground_truth_names[0:train_size]
     X_test  = ground_truth_names[test_count_start:dataset_size]
     y_train = ground_truth_values[0:train_size]
     y_test  = ground_truth_values[test_count_start:dataset_size]

     #initializing text vector sizes.
     x_train_values = np.zeros([vector_number, vector_length])
     X_test_values = np.zeros([vector_number, vector_length])
     vid_ctr=0

     print "Loading Training variables"
     for filename in X_train:
         temp_file = np.load(filename)
         x_train_values = np.vstack([x_train_values,temp_file])

         #skip dummy size vector after loading first vector.
         if vid_ctr==0:
            x_train_values = x_train_values[1,:]
         vid_ctr+=1

     print "Training vector and ground truth shapes are:"
     print x_train_values.shape, y_train.shape   
     print "Loading Testing variables"
     vid_ctr=0
     for filename in X_test:
         temp_file = np.load(filename)
         X_test_values = np.vstack([X_test_values,temp_file])

         #skip dummy size vector after loading first vector.
         if vid_ctr==0:
             X_test_values = X_test_values[1,:]
         vid_ctr+=1

     print "Validation vector and ground truth shapes are:"
     print X_test_values.shape, y_test.shape

     print "Text Vector Loading Complete"

    #Grid search for best SVM hyperparameters
     print "Grid search for best SVM hyperparameters"
     scores = ['precision', 'recall']
     candidate_parameters = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4], 'C': [1, 10, 100, 1000]},{'kernel': ['linear'], 'C': [1, 10, 100, 1000]}]

         
     for score in scores:
        print "# Fine-tuning hyper-parameters - %s" % score 
        clf = GridSearchCV(SVC(C=1), candidate_parameters, cv=5, scoring='%s_macro' % score)
        clf.fit(x_train_values, y_train)

        print "Best parameters:"
        print clf.best_params_ 

        print "Grid search scores on the test set:" 
        means = clf.cv_results_['mean_test_score']
        stds = clf.cv_results_['std_test_score']
        for mean, std, params in zip(means, stds, clf.cv_results_['params']):
            print("%0.3f (+/-%0.03f) for %r"
                % (mean, std * 2, params))

        print "Detailed classification report:" 
        print "The scores are computed on the full validation set."
        y_true, y_pred = y_test, clf.predict(X_test_values)
        print classification_report(y_true, y_pred) 
        print "Accuracy is:"
        print accuracy_score(y_true, y_pred) 
        print "MSE is:"
        print mean_squared_error(y_true, y_pred)
        print "Confusion matrix is:"
        print confusion_matrix(y_true, y_pred)

print "Hello I am a small script to train an SVM classifier using Skip-thought vectors. "
ground_truth_names, ground_truth = gtFileLoad(input_filename)
classifySVM(ground_truth_names, ground_truth)    

