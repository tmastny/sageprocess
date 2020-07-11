
import pandas as pd
from os.path import join

from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelBinarizer, KBinsDiscretizer
from sklearn.compose import make_column_transformer
from sklearn.model_selection import train_test_split

if __name__ == "__main__":
    input_path = join("/opt/ml/processing/input/census-income.csv")
    
    df = pd.read_csv(input_path)
    
    X_train, X_test, y_train, y_test = train_test_split(
        df.drop('income', axis=1), 
        df['income'], 
        test_size=0.3, 
        random_state=0
    )
    
    kbin = ['age', 'num persons worked for employer']
    ss = ['capital gains', 'capital losses', 'dividends from stocks']
    ohe = ['education', 'major industry code', 'class of worker']
    
    
    preprocess = make_column_transformer(
        (KBinsDiscretizer(encode='onehot-dense', n_bins=10), kbin),
        (StandardScaler(), ss),
        (OneHotEncoder(sparse=False, handle_unknown='ignore'), ohe)
    )
    
    X_train_transformed = preprocess.fit_transform(X_train)
    
    
    output_path = join("/opt/ml/processing/train", "census-train-transform.csv")
    pd.DataFrame(X_train_transformed).to_csv(output_path, header=False, index=False)
    
    
