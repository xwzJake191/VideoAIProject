from transformers import pipeline

# Create text classification pipeline
text_emo_classifier = pipeline("text-classification", model="SamLowe/roberta-base-go_emotions")

sentences = [
    "I am not having a great day",
    "To enable the following instructions: AVX2 AVX512F AVX512_VNNI FMA, in other operations, rebuild TensorFlow with the appropriate compiler flags.",
    "I had a great time today.",
    "There are multiple ways to use this model in Huggingface Transformers. Possibly the simplest is using a pipeline.",
    "Hi, Is there any codebase or guidance which I follow to finetune roberta on my own dataset?",
    "Thanks @SamLowe, I am basically trying to finetuned it on three class problem. I will give it a shot as well"
]

# Loop through the sentences
for sentence in sentences:
    result = text_emo_classifier(sentence)
    
    # Reformat the results
    formatted_result = {label['label']: f"{label['score'] * 100:.2f}%" for label in result}
    
    # Print the reformatted results
    print(f"Sentence: {sentence},\n"
          f"Recognition result: {formatted_result}""\n")

