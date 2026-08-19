import torch 
from transformers import ( AutoTokenizer, AutoModelForSequenceClassification)

MODEL_PATH = 'models_saved/bert_sentiment'

LABELS = {
    0: "negative",
    1: "neutral",
    2: "positive"
}

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)

def predict(text):
    inputs = tokenizer(text,return_tensors="pt",truncation=True,padding=True,max_length=128)

    with torch.no_grad():

        outputs = model(
            **inputs
        )

    probabilities = torch.softmax(outputs.logits,dim=1)
    predicted_class = torch.argmax(probabilities,dim=1).item()
    confidence = probabilities[0,predicted_class].item()

    return {
        'sentiment': LABELS[predicted_class], 
        'confidence': confidence
    }

if __name__ == "__main__":

    review = """
    The graphics are amazing but
    the servers are terrible.
    """

    result = predict(review)

    print(result)

