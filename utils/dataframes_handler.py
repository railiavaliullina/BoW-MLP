import pandas as pd


def read_file(path):
    """
    Reads data from pickle/csv files.
    :param path: path where file is located
    :return: dataframe with data
    """
    ext = path.split('.')[-1]
    if ext == 'csv':
        df = pd.read_csv(path)
    elif ext == 'pickle':
        df = pd.read_pickle(path)
    else:
        raise Exception
    print(f'Loaded dataframe from {path}')
    return df


def save_file(path, columns_names, columns_content):
    """
    Saves dataframe as pickle/csv file.
    :param path: path to save dataframe to
    :param columns_names: names of columns to create in dataframe
    :param columns_content: content of columns to write by corresponding name
    """
    ext = path.split('.')[-1]
    df = pd.DataFrame()
    for col_name, col_content in zip(columns_names, columns_content):
        df[col_name] = col_content
    if ext == 'csv':
        df.to_csv(path)
    elif ext == 'pickle':
        df.to_pickle(path)
    else:
        raise Exception
    print(f'Dataframe saved to {path}.')
