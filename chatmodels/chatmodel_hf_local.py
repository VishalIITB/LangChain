
from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline

# Initialize the pipeline
llm = HuggingFacePipeline.from_model_id(
    model_id='MiniLLM/MiniPLM-Qwen-200M',
    task='text-generation',
    
    
    pipeline_kwargs={
        "temperature": 0.5,
        "max_new_tokens": 100
    }
)

# Initialize the ChatHuggingFace model using the pipeline
model = ChatHuggingFace(llm_pipeline=llm)

# Invoke the model
result = model.invoke("What is the capital of India")

print(result.content)
