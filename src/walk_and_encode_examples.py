from sentence_transformers import SentenceTransformer
import os
import pandas as pd

class WalkAndEncode:
    def __init__(self, model_name=None):
        if model_name is None:
            self.model_name = "jinaai/jina-embeddings-v2-base-code"
        else:
            self.model_name = model_name
        self.model = SentenceTransformer(self.model_name, trust_remote_code=True)
        self.model.max_seq_length = 1024
    
    def read_file_content(self, file_path):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception as e:
            print(f"Error reading file {file_path}: {str(e)}")
            return None
    
    def walk(self, directory_path):
        results = []
        for root, dirs, files in os.walk(directory_path):
            for file in files:
                file_path = os.path.join(root, file)
                content = self.read_file_content(file_path)
                if content is None:
                    continue
                try:
                    encoding = self.model.encode(content)
                    results.append({
                        'file_path': file_path,
                        'encoding': encoding.tolist()
                    })
                    print(f"Processed: {file_path}")
                except Exception as e:
                    print(f"Error encoding {file_path}: {str(e)}")
                    continue
        df = pd.DataFrame(results)
        return df
    
    def encode_single_file(self, file_path):
        content = self.read_file_content(file_path)
        if content is None:
            return None
        try:
            return self.model.encode(content)
        except Exception as e:
            print(f"Error encoding {file_path}: {str(e)}")
            return None