import pickle
import pandas as pd 
from sklearn.feature_extraction.text import TfidfVectorizer

folder_path = 'D:/PYTHON/data/X_data.pkl'
X_data = pickle.load(open(folder_path, 'rb'))
folder_path = 'D:/PYTHON/data/y_data.pkl'
y_data = pickle.load(open(folder_path, 'rb'))
folder_path = 'D:/PYTHON/data/X_test.pkl'
X_test = pickle.load(open(folder_path, 'rb'))
folder_path = 'D:/PYTHON/data/y_test.pkl'
y_test = pickle.load(open(folder_path, 'rb'))

#biến đổi nhãn về dạng số

from sklearn import preprocessing

encoder = preprocessing.LabelEncoder()
y_data_n = encoder.fit_transform(y_data)
y_test_n = encoder.fit_transform(y_test)

#Biến đỗi các doc về dạng if-idf

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer         
tfidf_vect = TfidfVectorizer(analyzer='word', max_features=51289)
print('Số lượng từ điển : ',25000)
tfidf_vect.fit(X_data) 
X_data_tfidf =  tfidf_vect.transform(X_data)         
X_test_tfidf =  tfidf_vect.transform(X_test)
print(len(tfidf_vect.vocabulary_))

#Train-model

from sklearn.model_selection import train_test_split
from sklearn import metrics


def train_model(classifier, X_data, y_data, X_test, y_test):       
    X_train, X_val, y_train, y_val = train_test_split(X_data, y_data, test_size=0.1, random_state=42)        
    classifier.fit(X_train, y_train)            
    train_predictions = classifier.predict(X_train)
    val_predictions = classifier.predict(X_val)
    test_predictions = classifier.predict(X_test)
      
    print('Train accuracy: ', metrics.accuracy_score(train_predictions, y_train))
    print("Validation accuracy: ", metrics.accuracy_score(val_predictions, y_val))
    print("Test accuracy: ", metrics.accuracy_score(test_predictions, y_test))

    return classifier.predict(X_data)


def f1(label, train_predict, y_train):
    tp = [0] * len(label)
    fp = [0] * len(label)
    fn = [0] * len(label)
    soluong = [0] * len(label)

    p = [0] * len(label)
    r = [0] * len(label)
    f1_  = [0] * len(label)

    for i in range(len(train_predict)):

        soluong[y_train[i]] += 1
        if y_train[i] == train_predict[i]:
            tp[y_train[i]] += 1

        else:
            fn[y_train[i]] += 1
            fp[train_predict[i]] += 1

    for i in range(len(label)):

        p[i] = tp[i]/(tp[i] + fp[i])
        r[i] = tp[i]/(tp[i] + fn[i])
        f1_[i] = 2*p[i]*r[i]/(p[i]+r[i])
        
    final_result = {}
    final_result['label'] = label
    final_result['precision'] = p
    final_result['recall'] = r
    final_result['f1']  = f1_ 
    final_result['so luong'] = soluong

    df = pd.DataFrame(final_result)
    df.to_csv("a.csv", index=False)


#Mô hình Naive-Bayes 
from sklearn.naive_bayes import MultinomialNB

train_predict = train_model(MultinomialNB(), X_data_tfidf, y_data_n, X_test_tfidf, y_test_n)
#f1(encoder.classes_, train_predict, y_data_n )

from sklearn.metrics import classification_report
print(classification_report(y_data_n, train_predict, target_names=encoder.classes_,digits=6))