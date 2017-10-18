def read_columns(config,parameter):
    '''Read and parse column names from config'''
    return [ eval(config.get(s,parameter)) for s in config.sections() ]


def read_columns_name(config):
    name_columns = read_columns(config,'columns_name')
    out = []
    for columns in name_columns:
        for i,column in enumerate(columns):
            clean_column = re.sub('_',' ',column).strip()
            columns[i] = clean_column
    return name_columns
    
def read_columns_ucd(config):
    ucd_columns = read_columns(config,'columns_ucd')
    for columns in ucd_columns:
        for i,column in enumerate(columns):
            primary_ucd = column.split(';')[0]
            columns[i] = primary_ucd
    return ucd_columns
    

import configparser
config = configparser.ConfigParser()
_= config.read('CATALOGS.ini')

name_columns = read_columns_name(config)
ucd_columns = read_columns_ucd(config)

data = [ n for names in name_columns for n in names ]
target_label = [ u for ucds in ucd_columns for u in ucds ]

d = { u:i for i,u in enumerate(set(target_label)) }
d_ = { d[u]:u for u in d }

target = map(lambda u:d[u], target_label)
from numpy import array
target = array(list(target))


#from sklearn.feature_extraction.text import CountVectorizer
#count_vect = CountVectorizer()
#X_train_counts = count_vect.fit_transform(data)
#
#from sklearn.feature_extraction.text import TfidfTransformer
#tfidf_transformer = TfidfTransformer()
#X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
#
#from sklearn.naive_bayes import MultinomialNB
#clf = MultinomialNB().fit(X_train_tfidf,target)


## Pipeline

# Naive Bayes
from sklearn.pipeline import Pipeline
text_clf = Pipeline([('vect', CountVectorizer()),
                    ('tfidf', TfidfTransformer()),
                    ('clf', MultinomialNB()),
])
text_clf = text_clf.fit(data,target)

predict = lambda w:d_.get(text_clf.predict([w])[0])
#>>> predict('mag')

# SVM
from sklearn.linear_model import SGDClassifier
text_clf_svm = Pipeline([('vect', CountVectorizer()),
                        ('tfidf', TfidfTransformer()),
                        ('clf-svm', SGDClassifier(loss='hinge',
                        penalty='l2',
                        alpha=1e-3, n_iter=5, random_state=42)),
])
_= text_clf_svm.fit(data,target)

predict_svm = lambda w:d_.get(text_clf_svm.predict([w])[0])
#>>> predict_svm('mag')

