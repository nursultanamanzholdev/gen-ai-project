import argparse
import subprocess
import os

def convert_m4a_to_wav(input_file, output_file):
    if not input_file.lower().endswith('.m4a'):
        print(f"Warning: Input file '{input_file}' does not have .m4a extension. Attempting conversion anyway.")
    
    if not output_file.lower().endswith('.wav'):
        output_file += '.wav'
        print(f"Output file modified to '{output_file}' to ensure .wav extension.")
    
    if not os.path.exists(input_file):
        print(f"Error: Input file '{input_file}' does not exist.")
        return
    
    try:
        subprocess.run(["ffmpeg", "-i", input_file, output_file], check=True)
        print(f"Successfully converted '{input_file}' to '{output_file}'.")
    except subprocess.CalledProcessError as e:
        print(f"Error during conversion: {e}")
    except FileNotFoundError:
        print("Error: ffmpeg is not installed or not found in PATH. Please install ffmpeg.")

def main():
    parser = argparse.ArgumentParser(description="Convert M4A audio files to WAV format using ffmpeg.")
    parser.add_argument("input_file", help="Path to the input M4A file")
    parser.add_argument("output_file", help="Path to the output WAV file")
    args = parser.parse_args()
    
    convert_m4a_to_wav(args.input_file, args.output_file)

if __name__ == "__main__":
    main()