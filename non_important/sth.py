import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

def ask_flan_t5(prompt, model_name='google/flan-t5-large', max_length=512):
    # Load pre-trained model and tokenizer
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Encode the input prompt
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids

    # Generate a response
    outputs = model.generate(
        input_ids,
        max_length=max_length,
        num_return_sequences=1,
        do_sample=True,
        temperature=0.7,
        top_k=50,
        top_p=0.95,
    )

    # Decode and return the response
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

# Example usage
prompt = """
Analyze the following cat description and provide a structured output with these keys:
Sex, Age, Race, Nombre, Logement, Zone, Ext, Timide, Calme, Effrayé, Intelligent, Vigilant, Perséverant, Affectueux, Amical, Solitaire, Brutal, Dominant, Agressif, Impulsif, Prévisible, Distrait, PredOiseau, PredMamm.
Use "Unknown" for missing information. Use scales 1-5 for personality traits, where 1 is low and 5 is high. Use 0-4 for PredOiseau and PredMamm.

Cat description: I have a friendly female cat named Whiskers. She's about 3 years old and lives in our apartment in the city. She's very affectionate but can be a bit timid around strangers. Whiskers is an indoor cat and loves to play, but she's not very good at catching the toy mice we give her.

Structured output:
"""

response = ask_flan_t5(prompt)
print("FLAN-T5's response:", response)