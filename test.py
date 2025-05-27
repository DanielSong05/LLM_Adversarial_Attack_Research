from model_factory import load_model

model = load_model("llama-13b")
response = model.query("What are the risks of multi-turn adversarial attacks in dialogue systems?")
print(response)
