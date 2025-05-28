from flask import Flask, render_template, request, jsonify
from services.TTS import TextToSpeech
from services.TextGen import TextGenerator
from services.pipecat_openai_tts import OpenAITTSBlock
import os
import tempfile

app = Flask(__name__, static_folder='static', template_folder='templates')

# Initialize services
text_generator = TextGenerator()
tts = OpenAITTSBlock(voice="nova")

@app.route('/')
def index():
    """Render the main page with the Simli avatar"""
    return render_template('index.html')

@app.route('/api/generate-response', methods=['POST'])
def generate_response():
    """Generate AI response from user input and return speech file URL"""
    if not request.json or 'text' not in request.json:
        return jsonify({'error': 'Missing text parameter'}), 400
    
    user_text = request.json['text']
    
    # Generate AI response
    ai_response = text_generator.generate_response(user_text)
    
    # Generate speech file
    speech_file = tts(ai_response)
    
    # Return response with path to speech file
    speech_url = f"/static/audio/{os.path.basename(speech_file)}"
    
    return jsonify({
        'text': ai_response,
        'audio_url': speech_url
    })

@app.route('/api/simli-config')
def simli_config():
    """Return Simli configuration for the frontend"""
    return jsonify({
        'roomUrl': os.getenv('SIMLI_ROOM_URL', 
                            'https://pc-7efd6f2a87c8db0e8fe4ea108a6e11b6.daily.co/hoHfCDtVcKbk3uy8l2LB'),
        'sessionId': os.getenv('SIMLI_SESSION_ID', ''),
        'token': os.getenv('SIMLI_TOKEN', '')
    })

if __name__ == '__main__':
    # Create static audio directory if it doesn't exist
    os.makedirs('static/audio', exist_ok=True)
    
    # Run the Flask app
    app.run(debug=True, port=5000)