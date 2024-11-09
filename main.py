from src.walk_and_encode_examples import WalkAndEncode
from src.code_gen import CodeGen
import os
import pandas as pd

if __name__=='__main__':
    if not os.path.exists("example_mapper_rag_assist.csv"):
        wae = WalkAndEncode(api_key="sk-c3ddQQtQ1_M41pX0s9CY3A")
        encodings_df = wae.walk("./data/")
        encodings_df.to_csv("example_mapper_rag_assist.csv", index=False)
    else:
        encodings_df = pd.read_csv("example_mapper_rag_assist.csv")
    paths = [i.replace('data/', '') for i in encodings_df['file_path'].tolist()]
    cg = CodeGen(api_key="sk-c3ddQQtQ1_M41pX0s9CY3A", df=encodings_df)
    cg.generate("Convolutional Nueral Network visualization", paths)