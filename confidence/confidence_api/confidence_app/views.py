# confidence_app/views.py

from django.shortcuts import render
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from django.views.decorators.csrf import csrf_exempt
from django.utils.decorators import method_decorator
from .serializers import AudioFileSerializer
from .confident_detection_helper1 import AudioProcessor
import os

@method_decorator(csrf_exempt, name='dispatch')
class AudioUploadView(APIView):
    def get(self, request):
        return render(request, 'audio_upload.html')
    
    def post(self, request):
        audio_file = request.FILES.get('audio_file')
        if audio_file:
            try:
                # Define the path for the model 
                model_path = r'C:\Users\91889\Desktop\zummit\confidence\confidence_api\confidence_app\svm_model.joblib'  # Adjust this path accordingly
                temp_dir = 'temp'
                
                # Ensure the temporary directory exists
                if not os.path.exists(temp_dir):
                    os.makedirs(temp_dir)

                # Save the uploaded file temporarily
                temp_file_path = os.path.join(temp_dir, audio_file.name)
                with open(temp_file_path, 'wb+') as temp_file:
                    for chunk in audio_file.chunks():
                        temp_file.write(chunk)
                
                # Process the file using the audio processor
                audio_processor = AudioProcessor(model_path)
                confidence = audio_processor.process_file(temp_file_path)
                
                # Clean up the temporary file
                os.remove(temp_file_path)
                
                # Determine the confidence text
                confidence_text = 'High Confidence' if confidence == 1 else 'Low Confidence'
                return render(request, 'audio_upload.html', {'confidence': confidence_text})
            except Exception as e:
                return Response({'error': str(e)}, status=status.HTTP_400_BAD_REQUEST)
        else:
            return Response({'error': 'No audio file provided'}, status=status.HTTP_400_BAD_REQUEST)
