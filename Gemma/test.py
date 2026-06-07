import os
import sys
import queue
from unittest.mock import patch, MagicMock

# 1. Ensure Python can find the 'AssistantGlasses' folder in your workspace
WORKSPACE_ROOT = r"c:\Users\32873\.vscode\python"
if WORKSPACE_ROOT not in sys.path:
    sys.path.insert(0, WORKSPACE_ROOT)

# 2. Import your real agent module now that paths are aligned.
# (No sweeping global module mocks, so it will load your real AssistantGlasses package!)
from AssistantGlasses.Gemma import chat

# ==============================================================================
# 3. Interactive Multi-Turn Loop
# ==============================================================================
# We patch stream_audio to prevent ffplay/TTS from launching, keeping it purely text-based.
@patch('AssistantGlasses.Gemma.chat.stream_audio', MagicMock())
def start_terminal_chat():
    destination_queue = queue.Queue()
    speech_queue = queue.Queue()

    # Suppress the background TTS thread so it doesn't try to invoke ffplay
    with patch.object(chat.BaseAgent, '_tts_worker', return_value=None):
        # Initialize your Gemma agent targeting your local llama-server port
        agent = chat.Gemma4Agent(
            destination=destination_queue,
            role="default",
            speech=speech_queue,
            host="http://localhost:8090"
        )
    
    print("\n" + "="*60)
    print("  Gemma4 Smart Glasses Agent - Real Multi-Turn Terminal Session")
    print("  Type your message below. Try: 'Take a photo and tell me what you see.'")
    print("  Type 'exit' or 'quit' to end the session.")
    print("="*60 + "\n")

    while True:
        try:
            user_input = input("\nUser 👤: ").strip()
            if not user_input:
                continue
            if user_input.lower() in ['exit', 'quit']:
                print("Ending agent session. Goodbye!")
                break
            
            # Send input to your agent's live streaming logic
            # This will run your REAL capture_photo tool inside AssistantGlasses.Gemma.tool!
            agent.chat_stream(user_input, img_path=False, tool=True)

            # Clear out the speech queue silently since TTS is disabled
            while not speech_queue.empty():
                speech_queue.get_nowait()
                    
            # Print navigation signals if the model outputs a [&location/...&] tag
            while not destination_queue.empty():
                print(f"\n[Navigation Signal] 📍 Map destination intercepted: {destination_queue.get_nowait()}")
                    
        except KeyboardInterrupt:
            print("\nSession aborted.")
            break
        except Exception as e:
            print(f"\n[Runtime Error]: {e}")

if __name__ == "__main__":
    start_terminal_chat()