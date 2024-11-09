from src.walk_and_encode_examples import WalkAndEncode

if __name__=='__main__':
    wae = WalkAndEncode(api_key="sk-c3ddQQtQ1_M41pX0s9CY3A")
    encodings_df = wae.walk("./data/")
    encodings_df.to_csv("example_mapper_rag_assist.csv", index=False)