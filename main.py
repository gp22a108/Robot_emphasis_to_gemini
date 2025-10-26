import asyncio
import argparse
import Gemini
import subprocess
import sys
import os

if __name__ == "__main__":
    print("main.py is running.")

    # Start Http_realtime.py in the background
    print("Starting Http_realtime.py process...")
    try:
        # Get the absolute path to the directory containing main.py
        script_dir = os.path.dirname(os.path.abspath(__file__))
        http_realtime_path = os.path.join(script_dir, "Http_realtime.py")

        # Use the same python interpreter that is running this script
        python_executable = sys.executable
        
        if not os.path.exists(http_realtime_path):
            raise FileNotFoundError(f"Http_realtime.py not found at {http_realtime_path}")

        subprocess.Popen([python_executable, http_realtime_path])
        print("Http_realtime.py is running in the background.")
    except FileNotFoundError as e:
        print(f"Error: {e}. Make sure Http_realtime.py is in the same directory as main.py.")
    except Exception as e:
        print(f"Failed to start Http_realtime.py: {e}")

    print("Starting Gemini process...")

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        type=str,
        default=Gemini.DEFAULT_MODE,
        help="pixels to stream from",
        choices=["camera", "screen", "none"],
    )
    args = parser.parse_args()
    main = Gemini.AudioLoop(video_mode=args.mode)
    asyncio.run(main.run())
