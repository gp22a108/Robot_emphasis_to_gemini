import subprocess
import sys
import os
import time

if __name__ == "__main__":
    print("main.py is running.")

    script_dir = os.path.dirname(os.path.abspath(__file__))
    python_executable = sys.executable

    processes = []

    # Start Http_realtime.py
    try:
        http_realtime_path = os.path.join(script_dir, "Http_realtime.py")
        if not os.path.exists(http_realtime_path):
            raise FileNotFoundError(f"Http_realtime.py not found at {http_realtime_path}")
        p = subprocess.Popen([python_executable, http_realtime_path])
        processes.append(p)
        print("Http_realtime.py is running in the background.")
    except Exception as e:
        print(f"Failed to start Http_realtime.py: {e}")

    # Start YOLO.py
    try:
        yolo_path = os.path.join(script_dir, "YOLO.py")
        if not os.path.exists(yolo_path):
            raise FileNotFoundError(f"YOLO.py not found at {yolo_path}")
        p = subprocess.Popen([python_executable, yolo_path])
        processes.append(p)
        print("YOLO.py is running in the background.")
    except Exception as e:
        print(f"Failed to start YOLO.py: {e}")

    # Start Gemini.py
    try:
        gemini_path = os.path.join(script_dir, "Gemini.py")
        if not os.path.exists(gemini_path):
            raise FileNotFoundError(f"Gemini.py not found at {gemini_path}")
        # We will modify Gemini.py to watch for a file instead of a threshold argument
        p = subprocess.Popen([python_executable, gemini_path])
        processes.append(p)
        print("Gemini.py is running in the background.")
    except Exception as e:
        print(f"Failed to start Gemini.py: {e}")

    print("\nAll processes started. Monitoring for termination...")

    try:
        # Wait for all processes to complete
        for p in processes:
            p.wait()
    except KeyboardInterrupt:
        print("\nCtrl+C received. Terminating all processes.")
        for p in processes:
            p.terminate()

    print("main.py has finished.")