# text-classification-using-Naive-Bayes
Phân loại văn bản Tiếng Việt sử dụng thư viện sklearn
Dữ liệu tham khảo từ : https://github.com/duyvuleo/VNTC/tree/master/Data
Bộ dữ liệu gồm 3 thể loại :
 
Data training:
Chinh tri Xa hoi  có :  1000 văn bản.
The Thao  có :  1000 văn bản.
Van hoa  có :  1000 văn bản.
=> Tổng số  3000  văn bản
=> Độ dài trung bình 1 văn bản là:  1447 từ.

Data testing:
Chinh tri Xa hoi  có :  1000 văn bản.
The Thao  có :  1000 văn bản.
Van hoa  có :  1000 văn bản.
=> Tổng số  3000  văn bản
=> Độ dài trung bình 1 văn bản là:  1447 từ.

Chuẩn bị dữ liệu:

Trước hết, ta loại bỏ những ký tự đặc biệt trong văn bản ban đầu như dấu chấm, dấu phẩy, dấu mở đóng ngoặc,... bằng cách sử dụng thư viện gensim. 
Sau đó chúng ta sẽ sử dụng thư viện PyVi để tách từ tiếng Việt, bao gồm cả từ ghép, từ láy.

Các thư viện cần thiết :
from pyvi import ViTokenizer
import gensim 
import os
import pickle

-	Thư viện gen sim dùng để xử lý xoá các ký tự đặc biệt.
-	Thư viện hỗ trợ ViTokenizer dùng để tách từ trong tiếng việt.
Cài đặt thư viện pyvi, tại comand line : $ pip install pyvi
-	Thư viện os để xử lý file trong python.
-	Thư viện pickle để dump dữ liệu về dạng .pkl để tiện cho việc sử dụng lại dữ liệu đã xử lý.

Tiếp theo ta đưa mỗi bài báo về dạng X, y trong đó: X là nội dung bài báo đã được loại bỏ kí tự đặc biệt và tách từ, y là thể loại của văn bản ấy. Ta có văn bản X[i] thuộc thể loại y[i].
def getData(folder_path):
    X = []
    y = []
    # trong bộ dữ liệu thì tên thư mục là tên loại văn bản 
    categories = os.listdir(folder_path)  #danh sach ten thu muc
    for category in categories:
        cate_path = os.path.join(folder_path,category) 
        documents = os.listdir(cate_path) #danh sach ten van ban 
        for document in documents:
            doc_path = os.path.join(cate_path,document)
            document = open(doc_path,'r', encoding="utf-16")
            contentDoc = document.read()
            contentDoc = ViTokenizer.tokenize(contentDoc)           #tach tu 
            contentDoc = gensim.utils.simple_preprocess(contentDoc) # xoa cac ki tu dac biet 
            X.append(contentDoc)
            y.append(category)
    return X,y

Xuất dữ liệu về dạng .pkl:
# File Train ---------------------------------------
train_path = 'D:/PYTHON/train/'
X_data, y_data = getData(train_path)
X_data = list_to_doc(X_data)
pickle.dump(X_data, open('data/X_data.pkl', 'wb'))
pickle.dump(y_data, open('data/y_data.pkl', 'wb'))
        
# File Test ----------------------------------------
test_path = 'D:/PYTHON/test/'        
X_test, y_test = getData(test_path)
X_test = list_to_doc(X_test)
pickle.dump(X_test, open('data/X_test.pkl', 'wb'))
pickle.dump(y_test, open('data/y_test.pkl', 'wb'))

Như vậy ta đã thực hiện xong bước chuẩn bị dữ liệu. 
Ta bắt đầu xử lý dữ liệu :
Word Level TF-IDF : Thực hiện tính toán dựa trên mỗi thành phần là một từ riêng lẻ.
#Biến đỗi các doc về dạng if-idf

from sklearn.feature_extraction.text import TfidfVectorizer 
tfidf_vect = TfidfVectorizer(analyzer='word', max_features=30000)
tfidf_vect.fit(X_data) 
X_data_tfidf =  tfidf_vect.transform(X_data)         
X_test_tfidf =  tfidf_vect.transform(X_test)

Python hỗ trợ thư viện TfidfVectorizer để biến đổi dữ liệu về dạng: 
(d = stt văn bản, w = stt từ trong từ điển) = tfidf(w,d)

Câu lệnh:  tfidf_vect = TfidfVectorizer(analyzer='word', max_features=30000)
Dùng để khởi tạo một vector với mỗi thành phần là một từ riêng lẻ, xây dựng một từ điển chỉ xét max_features = 30000 từ đầu tiên được sắp xếp theo tần số xuất hiện.
Câu lệnh: tfidf_vect.fit(X_data) # học từ và idf tập train 


Xây dựng hàm huấn luyện cho mô hình.
from sklearn.model_selection import train_test_split
from sklearn import metrics


def train_model(classifier, X_data, y_data, X_test, y_test):       
    X_train, X_val, y_train, y_val = train_test_split(X_data, y_data, test_size=0.1, random_state=None)        
    classifier.fit(X_train, y_train)            
    train_predictions = classifier.predict(X_train)
    val_predictions = classifier.predict(X_val)
    test_predictions = classifier.predict(X_test)
      
    print('Train accuracy: ', metrics.accuracy_score(train_predictions, y_train))
    print("Validation accuracy: ", metrics.accuracy_score(val_predictions, y_val))
    print("Test accuracy: ", metrics.accuracy_score(test_predictions, y_test))


Sử dụng thư viện train_test_split trong câu lệnh 
X_train, X_val, y_train, y_val = train_test_split(X_data, y_data, test_size=0.1, random_state=None)        
	Tạo ra tập train và validation dựa trên tập X_data đầu vào với kích thước tập validation là 10% .        
classifier.fit(X_train, y_train): học từ tập train.
classifier.predict : đưa ra kết quả với tập đầu vào (văn bản thuộc thể loại nào).
metrics.accuracy_score   : Tính % dự đoán độ chính xác trên các tập dữ liệu. 

#Mô hình Naive-Bayes 
from sklearn.naive_bayes import MultinomialNB
train_model(MultinomialNB(), X_data_tfidf, y_data_n, X_test_tfidf, y_test_n)




