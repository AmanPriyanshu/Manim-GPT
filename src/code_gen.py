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
            },
            {
                "name": "code_deployer",
                "description": "Return code which can directly be executed in python, only code-script to be returned.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "code": {
                            "type": "string",
                        }
                    },
                    "required": ["code"]
                }
            },
            {
                "name": "code_combiner",
                "description": "Return code which combines given stage code, to create one unified script including all necessary imports and everything to visaulize the given objective.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "code": {
                            "type": "string",
                        }
                    },
                    "required": ["code"]
                }
            },
            {
                "name": "voiceover_gen",
                "description": "Return voiceover string to put over the following visualization no more than one-three sentences.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "audioscript": {
                            "type": "string",
                        }
                    },
                    "required": ["audioscript"]
                }
            }
        ]
        try:
            self.paths, self.encodings, self.question_encodings = self.df['file_path'].tolist(), np.array([json.loads(i) for i in self.df['encoding'].tolist()]), np.array([json.loads(i) for i in self.df['question_encoding'].tolist()])
        except:
            self.paths, self.encodings, self.question_encodings = self.df['file_path'].tolist(), np.array([i for i in self.df['encoding'].tolist()]), np.array([i for i in self.df['question_encoding'].tolist()])
        self.encodings, self.question_encodings = torch.tensor(self.encodings).float(), torch.tensor(self.question_encodings).float()

    def generate(self, aim, path_contexts, n=3):
        prompt = f"You are a story-boarding assistant with expertise in visual explanation and in Manim, the Graphical Animation Library. Your goal is to visualize the concept given by the user: \"{aim}\". Give simple steps for creating a 30 second clip, essentially you're story boarding with not more than 3-5 stages and return a string describe the events that take place in each of those stages. (Understand the limitations of Manim and be simplistic). Note: Here are some example files of code we have to take inspiration for some of the panels for:\n\n{path_contexts}"
        print("Generating story")
        response = self.client.chat.completions.create(
            model="openai/gpt-4o", #openai/
            messages=[{"role": "system", "content": "You are a research visualization tool. Mathemetical representations etc."}, {"role": "user", "content": prompt}],
            functions=self.functions,
            function_call={"name": "manim_story_board"}
        )
        story_board = json.loads(response.choices[0].message.function_call.arguments)['story_board']
        codes = []
        print("Generating individual code")
        for story in story_board:
            a = self.model.encode(story)
            cosine_similarity_raw = cos_sim(a, self.encodings).numpy().flatten()
            cosine_similarity_question = cos_sim(a, self.question_encodings).numpy().flatten()
            cosine_scores = cosine_similarity_raw+cosine_similarity_question
            top_n_documents = [self.paths[i] for i in np.argsort(cosine_scores[:n])]
            examples = []
            for document_path in top_n_documents:
                with open(document_path, "r") as f:
                    content = f.read()
                    examples.append(document_path+"\n----\n"+content[:5000])
            examples = "\n\n".join(examples)
            response = self.client.chat.completions.create(
                model="openai/gpt-4o", #openai/
                messages=[{"role": "system", "content": "You are a research visualization tool. Mathemetical representations etc. However, no time or complex latex be simple."}, {"role": "user", "content": f"Create code for creating the following visualization stage. The main aim is: {aim} and current stage is {story}.\n\nHere are some examples:{examples}"}],
                functions=self.functions,
                function_call={"name": "code_deployer"}
            )
            code = json.loads(response.choices[0].message.function_call.arguments)['code']
            codes.append(code)
        story_string = '\n'.join(story_board)
        code_string = '\n'.join(codes)
        response = self.client.chat.completions.create(
            model="openai/gpt-4o",  # Assuming 'gpt-4o' was a typo; update accordingly if needed
            messages=[
                {"role": "system", "content": "You are a research visualization tool. Mathematical representations, etc. However, no time or complex latex be simple. At the end of it, combine all scenes. Make sure not to use latex."},
                {
                    "role": "user",
                    "content": f"Create code for creating the following visualization stage. The main aim is: {aim} and the story-board is as follows: {story_string}.\n\nHere are segmented codes: {code_string}"
                }
            ],
            functions=self.functions,
            function_call={"name": "code_combiner"},
        )
        response = self.client.chat.completions.create(
            model="openai/gpt-4o",  # Assuming 'gpt-4o' was a typo; update accordingly if needed
            messages=[
                {"role": "system", "content": "You are a research visualization tool. Mathematical representations, etc. Your specific job is code correction. However, no time or complex latex be simple. At the end of it, combine all scenes into one single scene. Make sure not to use latex."},
                {
                    "role": "user",
                    "content": f"Create code for creating the following visualization stage. The main aim is: {aim} and the story-board is as follows: {story_string}.\n\nHere are segmented codes: {code_string}"
                }
            ],
            functions=self.functions,
            function_call={"name": "code_combiner"},
        )
        code = json.loads(response.choices[0].message.function_call.arguments)['code']
        response = self.client.chat.completions.create(
            model="openai/gpt-4o",  # Updated model if 'gpt-4o' was a typo; adjust accordingly
            messages=[
                {"role": "system", "content": "You are a research visualization tool. Mathematical representations, etc. However, no time or complex latex be simple."},
                {
                    "role": "user",
                    "content": f"Create code for creating the following visualization stage. The main aim is: {aim} and the story-board is as follows: {story_string}.\n\nHere is the visualization code:{code}"
                }
            ],
            functions=self.functions,
            function_call={"name": "voiceover_gen"},
        )
        audioscript = json.loads(response.choices[0].message.function_call.arguments)['audioscript']
        return code, audioscript