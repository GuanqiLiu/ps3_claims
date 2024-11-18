import hashlib
import pandas as pd

def create_sample_split(df, id_column, training_frac=0.8):
    """Create sample split with exact training and testing proportions.

    Parameters
    ----------
    df : pd.DataFrame
        Input data.
    id_column : str
        Name of the ID column.
    training_frac : float, optional
        Fraction of data to assign to the training set (default is 0.8).

    Returns
    -------
    pd.DataFrame
        DataFrame with a new 'sample' column containing 'train' or 'test'.
    """
    # Helper function to generate a stable hash value
    def id_to_hash(value):
        hashed_value = int(hashlib.md5(str(value).encode('utf-8')).hexdigest(), 16)
        return hashed_value

    # Generate a stable hash value for each ID
    df['hash'] = df[id_column].apply(id_to_hash)

    # Sort the data by the hash value
    df = df.sort_values(by='hash').reset_index(drop=True)

    # Calculate the split index based on training_frac
    split_index = int(len(df) * training_frac)

    # Assign 'train' or 'test' based on the index
    df['sample'] = ['train' if i < split_index else 'test' for i in range(len(df))]

    # Drop the intermediate hash column
    #df.drop(columns=['hash'], inplace=True)

    return df

# Example usage
data = {
    'id': ['A', 'B', 'C', 'D', 'E'],
    'feature': [1, 2, 3, 4, 5]
}
df = pd.DataFrame(data)
result = create_sample_split(df, id_column='id', training_frac=0.8)
print(result)
