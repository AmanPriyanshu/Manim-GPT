from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim
from openai import OpenAI
import json
import torch
import numpy as np

PROXY_ENDPOINT = "https://nova-litellm-proxy.onrender.com/"

class CodeGen:
    def __init__(self, api_key, df, model_name=None):
        self.api_key = api_key
        self.df = df
        self.client = OpenAI(api_key=self.api_key, base_url=PROXY_ENDPOINT)
        if model_name is None:
            self.model_name = "jinaai/jina-embeddings-v2-base-code"
        else:
            self.model_name = model_name
        self.model = SentenceTransformer(self.model_name, trust_remote_code=True)
        self.model.max_seq_length = 1024
        self.functions = [
            {
                "name": "manim_story_board",
                "description": "Describe the following <1 minute manim storyboard which is possible with simple code.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "story_board": {
                            "type": "array",
                            "items": {
                                "type": "string",
                            },
                            "description": "An array of story stages.",
                            "minItems": 3,
                            "maxItems": 5,
                        }
                    },
                    "required": ["story_board"]
                }
            }
        ]
        self.paths, self.encodings, self.question_encodings = self.df['file_path'].tolist(), np.array([json.loads(i) for i in self.df['encoding'].tolist()]), np.array([json.loads(i) for i in self.df['question_encoding'].tolist()])
        self.encodings, self.question_encodings = torch.tensor(self.encodings).float(), torch.tensor(self.question_encodings).float()

    def generate(self, aim, path_contexts):
        prompt = f"You are a story-boarding assistant with expertise in visual explanation and in Manim, the Graphical Animation Library. Your goal is to visualize the concept given by the user: \"{aim}\". Give simple steps for creating a 30 second clip, essentially you're story boarding with not more than 3-5 stages and return a string describe the events that take place in each of those stages. (Understand the limitations of Manim and be simplistic). Note: Here are some example files of code we have to take inspiration for some of the panels for:\n\n{path_contexts}"
        response = self.client.chat.completions.create(
            model="openai/gpt-4o",
            messages=[{"role": "system", "content": "You are a research visualization tool. Mathemetical representations etc."}, {"role": "user", "content": prompt}],
            functions=self.functions,
            function_call={"name": "manim_story_board"}
        )
        story_board = json.loads(response.choices[0].message.function_call.arguments)['story_board']
        for story in story_board:
            a = self.model.encode(story)
            print(a.shape)
            cosine_similarity_raw = cos_sim(a, self.encodings)
            cosine_similarity_question = cos_sim(a, self.question_encodings)
            print(cosine_similarity_question, cosine_similarity_raw)
            exit()

        # prompt = """You are a coding assistant with expertise in Manim, Graphical Animation Library. \n
        #     Here is a full set of Manim documentation:  \n ------- \n  {context} \n ------- \n Answer the user
        #     question based on the above provided documentation. Ensure any code you provide can be executed \n
        #     with all required imports and variables defined. Structure your answer with a description of the code solution. \n
        #     Then list the imports. And finally list the functioning code block. Here is the user question:"""