from transformers import AutoModel, AutoTokenizer
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from ray import serve
import ray.train.torch
from pydantic import BaseModel

app = FastAPI()

origins = [*]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class CaptionRequest(BaseModel):
    image_path: str

@serve.deployment() # indicate that this is a ray serve deployment
@serve.ingress(app) # this is what tells ray serve that it has to take in this "app"
class YourBigModel:

    def __init__(self):
        # The generic torch.device() will not work here when you deploy across multiple GPUs. 
        # You need to use that specific device where ray deploys a replica on its own.
        self.device = ray.train.torch.get_device()
        self.load_models()
    
    def load_models(self):
        path = 'OpenGVLab/InternVL2-8B'
        self.model = AutoModel.from_pretrained(
            path,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            trust_remote_code=True
          ).eval().to(device) # send to specific device
        self.tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True, use_fast=False)
        self.generation_config = dict(max_new_tokens=1024, do_sample=False)
        self.question = "<image>\nPlease give a detailed caption for the given image that covers all the objects in foreground and background. Please do not start with 'The image'."
    
    @app.post("/caption")
    def generate_caption(self, request:CaptionRequest):
        image_path = request.image_path
        pixel_values = load_image(image_path, max_num=12).to(torch.bfloat16).to(device) # send to device
        caption = self.model.chat(self.tokenizer, pixel_values, self.question, self.generation_config)
        return caption.strip()

# This is the entry point of ray serve. We will use this in the deployment config.
entry = YourBigModel.bind()