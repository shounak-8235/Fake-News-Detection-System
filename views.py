from django.shortcuts import render
from django.http import JsonResponse
from django.db import models
from .models import NewsScan
import os
import pickle
from django.conf import settings

# Load model and vectorizer only once
model_path = os.path.join(settings.BASE_DIR, 'detector/mlModel/fake_news_model.pkl')
vectorizer_path = os.path.join(settings.BASE_DIR, 'detector/mlModel/tfidf_vectorizer.pkl')

model = pickle.load(open(model_path, 'rb'))
vectorizer = pickle.load(open(vectorizer_path, 'rb'))

def home(request):
    return render(request, 'index.html')

def predict(request):
    print("PREDICT FUNCTION CALLED")
    
    if request.method == "POST":
        news_text = request.POST.get('news')
        print("User Input:", news_text)

        # Transform text
        transformed_text = vectorizer.transform([news_text])
        print("Vector Shape:", transformed_text.shape)

        # Predict
        prediction = model.predict(transformed_text)
        print("Prediction Raw:", prediction)

        # Result
        if prediction[0] == 1:
            result = "Fake News "
        else:
            result = "Real News"

        # Save to database
        NewsScan.objects.create(
            news_text=news_text,
            prediction=result
        )

        # Check if it's an AJAX request
        if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
            return JsonResponse({'result': result})

        return render(request, 'index.html', {'result': result})

    return render(request, 'index.html')