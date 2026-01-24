import subprocess
import sys
import os
import time
import Logger

if __name__ == "__main__":
    total_start_time = time.perf_counter()
    print("main.py is running.")

    script_dir = os.path.dirname(os.path.abspath(__file__))
    python_executable = sys.executable

    processes = []

    def start_process(script_name):
        start_time = time.perf_counter()
        try:
            script_path = os.path.join(script_dir, script_name)
            if not os.path.exists(script_path):
                raise FileNotFoundError(f"{script_name} not found at {script_path}")
            p = subprocess.Popen([python_executable, script_path])
            processes.append(p)
            elapsed = time.perf_counter() - start_time
            print(f"{script_name} is running in the background. (Startup time: {elapsed:.3f}s)")
        except Exception as e:
            elapsed = time.perf_counter() - start_time
            Logger.log_system_error("プロセス起動", e, message=f"script={script_name}")
            print(f"Failed to start {script_name}: {e} (Attempted for: {elapsed:.3f}s)")

    # 各プロセスを起動
    start_process("Http_realtime.py")
    start_process("Gemini.py")
    start_process("PyView.py")

    total_elapsed = time.perf_counter() - total_start_time
    print(f"\nAll processes started in {total_elapsed:.3f}s. Monitoring for termination...")

    try:
        # Wait for all processes to complete
        for p in processes:
            p.wait()
    except KeyboardInterrupt:
        print("\nCtrl+C received. Terminating all processes.")
        for p in processes:
            p.terminate()

    print("main.py has finished.")
