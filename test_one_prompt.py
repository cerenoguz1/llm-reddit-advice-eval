import subprocess

prompt = input("Enter your advice question:\n> ")

try:
    result = subprocess.run(
        ["llm", "-m", "gemma2", prompt],
        capture_output=True,
        text=True,
        timeout=60
    )
    print("\nModel's Advice:\n")
    print(result.stdout.strip())
except Exception as e:
    print(f"Error: {e}")
