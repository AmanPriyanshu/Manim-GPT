from sentence_transformers import SentenceTransformer
import pandas as pd
from openai import OpenAI
import json

PROXY_ENDPOINT = "https://nova-litellm-proxy.onrender.com/"

class CodeGen:
    def __init__(self, api_key, model_name=None):
        self.api_key = api_key
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

    def generate(self, aim):
        prompt = "You are a planning assistant with expertise in visual explanation and in Manim, the Graphical Animation Library. Your goal is to observe this given objective by the user: \"{aim}\". Give me steps / kinds of visuals to plan for, for creating a 30 second clip, essentially you're story boarding with not more than 3-5 stages and return a string describe the events that take place in each of those stages. (Understand the limitations of Manim and be simplistic)"
        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            functions=self.functions,
            function_call={"name": "manim_story_board"}
        )
        args = json.loads(response.choices[0].message.function_call.arguments)
        print(args)
        # prompt = """You are a coding assistant with expertise in Manim, Graphical Animation Library. \n
        #     Here is a full set of Manim documentation:  \n ------- \n  {context} \n ------- \n Answer the user
        #     question based on the above provided documentation. Ensure any code you provide can be executed \n
        #     with all required imports and variables defined. Structure your answer with a description of the code solution. \n
        #     Then list the imports. And finally list the functioning code block. Here is the user question:"""