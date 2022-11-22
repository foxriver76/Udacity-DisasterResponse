import nltk
nltk.download(['punkt', 'wordnet', 'omw-1.4'])

from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, recall_score, f1_score
from sklearn.model_selection import GridSearchCV

import pickle
import pandas as pd
from sqlalchemy import create_engine
import numpy as np
import sys


def load_data(database_filepath: str) -> [pd.DataFrame, pd.DataFrame]:
    """
    Loads data from the database and return train and test data
    
    :param database_filepath: path to the sqlite database
    """
    
    engine = create_engine(f'sqlite:///{database_filepath}')
    df = pd.read_sql_table(
        'disaster_response',
        con=engine
    )

    X = df.message 
    Y = df.iloc[:,4:]
    
    return X, Y


def tokenize(text: str) -> np.array:
    """
    This method performs tokenization
    
    :param text: text to process
    :return: array of tokens
    """
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


def build_model() -> GridSearchCV:
    """
    This method builds the model pipeline
    
    :return: GridSearchCV
    """
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])
    
    # just some params, else it gets too slow for a showcase
    parameters = {
        #'vect__ngram_range': ((1, 1), (1, 2)),
        #'clf__estimator__n_estimators': [50, 100, 200],
        #'clf__estimator__min_samples_split': [2, 3, 4]
        'clf__estimator__n_estimators': [50, 100]
    }

    cv = GridSearchCV(pipeline, param_grid=parameters, cv=5, n_jobs=-1, error_score='raise')
    
    return cv

def evaluate_model(model: Pipeline, X_test: pd.DataFrame, Y_test: pd.DataFrame) -> None:
    """
    Evaluates the model by printing scores
    
    :param model: The model or pipeline
    :param X_test: Test data
    :param Y_test: Test labels
    :return: None
    """
    Y_pred = model.predict(X_test)
    
    for i in range(Y_test.shape[1]):
        print(f'Metrics for {Y_test.iloc[:, i].name}:')
        print(classification_report(Y_pred[:, i], Y_test.values[:, i]))
        print('\n')


def save_model(model, model_filepath) -> None:
    """
    Saves the model as pickle dump
    
    :param model: The model to save
    :param model_filepath: Path to save the model too
    :return: None
    """
    pickle.dump(model, open(model_filepath, 'wb'))

def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y = load_data(database_filepath)

        # For showcase may switch to a higher test_size to further speed up the training process
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        cv = build_model()
        
        print('Training model...')
        cv.fit(X_train, Y_train)
        
        # only save the best estimator found via gridsearch
        print(f'Best parameters: {cv.best_params_} with score {cv.best_score_}')
        model = cv.best_estimator_
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()