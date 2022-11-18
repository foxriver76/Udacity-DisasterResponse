import pandas as pd
from sqlalchemy import create_engine
import sys


def load_data(messages_filepath: str, categories_filepath: str) -> pd.DataFrame:
    """
    Loads the data and merges messages and categories
    
    :param messages_filepath: Path to the messages csv file
    :param categories_filepath: Path to the categories csv file
    :return: DataFrame containing messages and categories merged
    """
    
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    
    # merge datasets
    df = pd.merge(messages, categories, on='id')
    
    return df
   
def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans the given DataFrame and returns cleaned version
    This splits the categories in separate columns and removes duplicates
    
    :param df: DataFrame which will be cleaned
    :return: Cleaned DataFrame
    """
    
    # create a dataframe of the 36 individual category columns
    categories = df.categories.str.split(';', expand=True)
    
    # select the first row of the categories dataframe
    row = categories.iloc[0]

    # use this row to extract a list of new column names for categories.
    # one way is to apply a lambda function that takes everything 
    # up to the second to last character of each string with slicing
    category_colnames = [r.split('-')[0] for r in row]
    
    # rename the columns of `categories`
    categories.columns = category_colnames

    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].str.split('-').str[1]

        # convert column from string to numeric
        categories[column] = pd.to_numeric(categories[column])
        
    # drop the original categories column from `df`
    df.drop(columns=['categories'], inplace=True)
    
    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df, categories], axis=1)
    
    # drop duplicates
    df.drop_duplicates(inplace=True)
    return df


def save_data(df: pd.DataFrame, database_filename: str) -> None:
    """
    This method takes a DataFrame and saves it to a SQLite database
    :param df: DataFrame to save to DB
    :param database_filename: Path to database file
    :return: None
    """
    
    engine = create_engine(f'sqlite:///{database_filename}')
    df.to_sql('disaster_response', engine, index=False)

def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()