from models.gpt_wrapper import GPT4Model
from models.claude_wrapper import ClaudeModel
from models.groq_wrapper import GroqCloudModel
def load_model(name: str):
    name = name.lower()

    if name == "llama-70b":
        return GroqCloudModel(model_name="llama-3.3-70b-versatile")
    elif name == "llama-8b":
        return GroqCloudModel(model_name="llama-3.1-8b-instant")
    elif name == "gpt-4":
        return GPT4Model(model_name="gpt-4-turbo") 
    elif name == "claude-sonnet":
        return ClaudeModel(model_name="claude-3-7-sonnet-20250219")
    
    else:
        raise ValueError(f"Unknown model name: {name}")
