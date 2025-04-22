import librosa
import soundfile as sf
import argparse

def downsample_wav(input_file, output_file, target_sr=24000):
    # Load the audio file with its original sampling rate
    y, sr = librosa.load(input_file, sr=None)
    
    # Check if downsampling is needed
    if sr == target_sr:
        print(f"Input file '{input_file}' is already at {target_sr} Hz. No downsampling needed.")
        return
    
    # Downsample the audio to the target sampling rate
    y_resampled = librosa.resample(y, orig_sr=sr, target_sr=target_sr)
    
    # Save the downsampled audio to a new WAV file
    sf.write(output_file, y_resampled, target_sr)
    print(f"Downsampled '{input_file}' from {sr} Hz to {target_sr} Hz and saved as '{output_file}'.")

def main():
    parser = argparse.ArgumentParser(description="Downsample a WAV file to a target sampling rate.")
    parser.add_argument("input_file", help="Path to the input WAV file")
    parser.add_argument("output_file", help="Path to the output downsampled WAV file")
    parser.add_argument("--target_sr", type=int, default=24000, help="Target sampling rate (default: 24000 Hz)")
    args = parser.parse_args()
    
    downsample_wav(args.input_file, args.output_file, args.target_sr)

if __name__ == "__main__":
    main()