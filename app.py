from services.STT import SpeechToText
from services.TextGen import TextGenerator
from services.TTS import TextToSpeech
from services.pipecat_openai_tts import OpenAITTSBlock
from services.ConversationExtractor import ConversationExtractor
import pipecat  # Import the pipecat module

# Define your own Pipeline class
class Pipeline:
    def __init__(self, blocks):
        self.blocks = blocks
    
    def process(self, input_data):
        current_data = input_data
        for block in self.blocks:
            current_data = block(current_data)
            if current_data is None:
                break
        return current_data

# Add this class to implement branching functionality
class BranchBlock:
    def __init__(self, condition, if_true, if_false):
        self.condition = condition
        self.if_true = if_true
        self.if_false = if_false
    
    def __call__(self, input_data):
        if self.condition(input_data):
            # Change this line to use process() instead of calling directly
            return self.if_true.process(input_data)
        else:
            # Change this line to use process() instead of calling directly
            return self.if_false.process(input_data)

class SpeechToTextBlock:
    def __init__(self):
        self.stt = SpeechToText()
        self.last_text = None
    
    def __call__(self, audio_file):
        if audio_file:
            self.last_text = self.stt.transcribe_audio(audio_file)
            return self.last_text
        return None

class TextGeneratorBlock:
    def __init__(self):
        self.text_gen = TextGenerator()
    
    def __call__(self, text):
        if text:
            return self.text_gen.generate_response(text)
        return None

class AudioRecorderBlock:
    def __init__(self, silence_duration=4, max_record_time=30):
        self.stt = SpeechToText()
        self.silence_duration = silence_duration
        self.max_record_time = max_record_time
    
    def __call__(self, _):
        print("\nListening...")
        return self.stt.record_audio(silence_duration=self.silence_duration, 
                                    max_record_time=self.max_record_time)

class AudioPlayerBlock:
    def __init__(self):
        self.tts = TextToSpeech()
    
    def __call__(self, speech_file):
        if speech_file:
            try:
                self.tts.play_audio(speech_file)
            except Exception as e:
                print(f"Error playing audio: {e}")
                # If there's an error, wait a moment and try to release the file
                import time
                time.sleep(1)
                try:
                    import os
                    if os.path.exists(speech_file):
                        # Just proceed without playing if we can't access it
                        pass
                except:
                    pass
        return speech_file  # Pass through for chaining

class PrintBlock:
    def __init__(self, prefix=""):
        self.prefix = prefix
    
    def __call__(self, text):
        if text:
            print(f"{self.prefix}{text}")
        return text  # Pass through for chaining

class ExitCheckBlock:
    def __init__(self):
        self.should_exit = False
    
    def __call__(self, text):
        if text and text.lower() in ['exit', 'quit', 'goodbye', 'bye']:
            self.should_exit = True
            return "Goodbye! It was nice talking to you."
        return text

class AIVoiceAgentPipeline:
    def __init__(self):
        # Create pipeline blocks
        self.recorder = AudioRecorderBlock()
        self.stt = SpeechToTextBlock()
        self.exit_check = ExitCheckBlock()
        self.text_gen = TextGeneratorBlock()
        self.tts = OpenAITTSBlock(voice="nova")
        self.user_printer = PrintBlock("You: ")
        self.assistant_printer = PrintBlock("Assistant: ")
        self.audio_player = AudioPlayerBlock()
        
        # Add the conversation extractor
        self.extractor = ConversationExtractor()
        
        # Create the exit branch pipeline
        self.exit_pipeline = Pipeline([  # Use Pipeline class with a list of blocks
            self.assistant_printer,
            self.tts,
            self.audio_player
        ])
        
        # Create a custom block factory that captures user input for the extractor
        def create_extractor_block(user_text):
            return self.InformationExtractorBlock(self.extractor, user_text)

        # Create the conversation pipeline with information extraction
        self.conversation_pipeline = Pipeline([
            self.text_gen,
            self.assistant_printer,
            lambda text: create_extractor_block(self.stt.last_text)(text),  # Capture user input
            self.tts,
            self.audio_player
        ])
        
        # Build the main pipeline with our custom branching
        self.pipeline = Pipeline([  # Use Pipeline class with a list of blocks
            self.recorder,
            self.stt,
            self.user_printer,
            self.exit_check,
            BranchBlock(
                condition=lambda text: self.exit_check.should_exit,
                if_true=self.exit_pipeline,
                if_false=self.conversation_pipeline
            )
        ])
    
    # Add a new block to extract information
    class InformationExtractorBlock:
        def __init__(self, extractor, user_text=None):
            self.extractor = extractor
            self.user_text = user_text
            
        def __call__(self, assistant_text):
            if assistant_text and self.user_text:
                extracted_info = self.extractor.extract_information(
                    self.user_text, assistant_text
                )
                # Optionally print extracted information
                if any(extracted_info.values()):
                    print("\n[Important information extracted]")
            return assistant_text  # Pass through for chaining
    
    def start_conversation(self):
        print("AI Voice Agent activated. Speak to interact.")
        print("Say 'exit' or 'quit' to end the conversation.")
        
        # Initial greeting
        initial_greeting = "Hello! I'm your AI voice assistant. How can I help you today?"
        print("Assistant: " + initial_greeting)
        tts = TextToSpeech()
        speech_file = tts.generate_speech(initial_greeting, voice="nova")
        tts.play_audio(speech_file)
        
        # Run the pipeline in a loop
        while not self.exit_check.should_exit:
            try:
                # None is a placeholder input to start the pipeline
                self.pipeline.process(None)
                
                if self.exit_check.should_exit:
                    print("\nExiting AI Voice Agent...")
                    break
                    
            except KeyboardInterrupt:
                print("\nExiting AI Voice Agent...")
                break
            except Exception as e:
                print(f"Error in pipeline: {e}")

if __name__ == "__main__":
    agent = AIVoiceAgentPipeline()
    agent.start_conversation()