from sentence_transformers import SentenceTransformer
import os
import pandas as pd
from openai import OpenAI

PROXY_ENDPOINT = "https://nova-litellm-proxy.onrender.com/"

class WalkAndEncode:
    def __init__(self, api_key, model_name=None):
        self.api_key = api_key
        self.client = OpenAI(api_key=self.api_key, base_url=PROXY_ENDPOINT)
        if model_name is None:
            self.model_name = "jinaai/jina-embeddings-v2-base-code"
        else:
            self.model_name = model_name
        self.model = SentenceTransformer(self.model_name, trust_remote_code=True)
        self.model.max_seq_length = 1024

    def infer(self, code):
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": f"Observe the given code-script and document, please return a 3-4 sentence description of what the goal of this code is in question format such that we could use it to retrieve code during a future information-retrieval segment. Code:\n\n```{code[:5000]}```"}],
            )
            return response.choices[0].message.content
        except:
            return None, None, None
    
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
                    question = self.infer(content)
                    encoding = self.model.encode(content)
                    question_encoding = self.model.encode(question)
                    results.append({
                        'file_path': file_path,
                        "question": question,
                        "question_encoding": question_encoding.tolist(),
                        'encoding': encoding.tolist(),
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