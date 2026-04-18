"""
Download Speech Emotion Recognition Datasets
Supports: RAVDESS, TESS, CREMA-D
"""

import os
import urllib.request
import zipfile
import json
from pathlib import Path

# Zenodo often blocks default Python user agents. Apply a generic browser user agent.
opener = urllib.request.build_opener()
opener.addheaders = [('User-agent', 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36')]
urllib.request.install_opener(opener)

def get_zenodo_file_link(record_id, filename_substring):
    """Fetch the dynamic download URL from Zenodo API"""
    url = f"https://zenodo.org/api/records/{record_id}"
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    try:
        with urllib.request.urlopen(req) as response:
            data = json.loads(response.read().decode())
            for f in data.get('files', []):
                # Using lower case for robust search
                if filename_substring.lower() in f.get('key', '').lower():
                    return f['links']['self']
    except Exception as e:
        print(f"Failed to fetch record API {record_id}: {e}")
    return None

def create_directories():
    """Create necessary directories"""
    dirs = ['data', 'models', 'uploads']
    for dir_name in dirs:
        os.makedirs(dir_name, exist_ok=True)
    print("✓ Created directories: data, models, uploads")

def download_ravdess_dataset():
    """
    Download RAVDESS dataset (Speech only)
    ~630MB
    """
    print("\n📥 Downloading RAVDESS Dataset (~630MB)...")
    
    data_path = Path("data/RAVDESS")
    data_path.mkdir(parents=True, exist_ok=True)
    
    if (data_path / "Actor_01").exists():
        print("✓ RAVDESS already downloaded")
        return True
        
    url = get_zenodo_file_link("1188976", "Audio_Speech_Actors_01-24")
    if not url:
        print("✗ Could not resolve dynamic URL from Zenodo API.")
        return False
        
    file_path = data_path.parent / "ravdess.zip"
    
    try:
        print("Downloading RAVDESS...")
        urllib.request.urlretrieve(url, file_path, reporthook=download_progress)
        print("\n✓ Downloaded RAVDESS")
        
        print("Extracting RAVDESS...")
        with zipfile.ZipFile(file_path, 'r') as zip_ref:
            zip_ref.extractall(data_path)
        print("✓ Extracted RAVDESS")
        
        os.remove(file_path)
        return True
    except Exception as e:
        print(f"\n✗ Error downloading RAVDESS: {str(e)}")
        return False

def download_tess_dataset():
    """
    Download TESS Dataset (Toronto Emotional Speech Set)
    ~1.5GB
    """
    print("\n📥 Downloading TESS Dataset (~1.5GB)...")
    
    data_path = Path("data/TESS")
    data_path.mkdir(parents=True, exist_ok=True)
    
    # Using zenodo metadata search
    if list(data_path.glob("TESS*")) or list(data_path.glob("*TESS*")):
        print("✓ TESS already downloaded")
        return True
        
    url = get_zenodo_file_link("4618484", "tess")
    if not url:
        print("✗ Could not resolve TESS URL from Zenodo API.")
        return False
        
    file_path = data_path.parent / "tess.zip"
    
    try:
        print("Downloading TESS (this may take several minutes)...")
        urllib.request.urlretrieve(url, file_path, reporthook=download_progress)
        print("\n✓ Downloaded TESS")
        
        print("Extracting TESS...")
        with zipfile.ZipFile(file_path, 'r') as zip_ref:
            zip_ref.extractall(data_path)
        print("✓ Extracted TESS")
        
        os.remove(file_path)
        return True
    except Exception as e:
        print(f"\n⚠️  Warning - TESS download failed: {str(e)}")
        print("   You can manually download from https://zenodo.org/record/4618484")
        return False

def download_crema_d_dataset():
    """
    Download CREMA-D Dataset (Crowd-sourced Emotional Multimodal Actors Dataset)
    ~5.8GB
    """
    print("\n📥 Downloading CREMA-D Dataset (~5.8GB)...")
    print("⚠️  Note: CREMA-D is large. Consider downloading manually if connection is unstable.")
    
    data_path = Path("data/CREMA-D")
    data_path.mkdir(parents=True, exist_ok=True)
    
    if (data_path / "AudioWAV").exists() or list(data_path.glob("*.wav")):
        print("✓ CREMA-D already downloaded")
        return True
        
    url = get_zenodo_file_link("3617050", "CremaD_FULL")
    if not url:
        print("✗ Could not resolve CREMA-D URL from Zenodo API.")
        return False
        
    file_path = data_path.parent / "crema_d.zip"
    
    try:
        print("Downloading CREMA-D (this may take 10-20 minutes)...")
        urllib.request.urlretrieve(url, file_path, reporthook=download_progress)
        print("\n✓ Downloaded CREMA-D")
        
        print("Extracting CREMA-D...")
        with zipfile.ZipFile(file_path, 'r') as zip_ref:
            zip_ref.extractall(data_path)
        print("✓ Extracted CREMA-D")
        
        os.remove(file_path)
        return True
    except Exception as e:
        print(f"\n⚠️  Warning - CREMA-D download failed: {str(e)}")
        print("   You can manually download from https://github.com/CheyneyComputerScience/CREMA-D")
        return False

def download_progress(block_num, block_size, total_size):
    """Show download progress"""
    downloaded = block_num * block_size
    if total_size > 0:
        percent = min(downloaded * 100 // total_size, 100)
        print(f"\rProgress: {percent}%", end="")
    else:
        # In case total_size isn't provided by the server
        print(f"\rDownloaded {downloaded / (1024*1024):.2f} MB...", end="")

def organize_dataset():
    """Organize the downloaded datasets"""
    data_path = Path("data")
    
    print("\n📊 Dataset Summary:")
    print("=" * 50)
    
    # Check RAVDESS
    ravdess_actors = list(data_path.glob("RAVDESS/Actor_*"))
    if ravdess_actors:
        print(f"✓ RAVDESS: {len(ravdess_actors)} actors")
    else:
        print("✗ RAVDESS: Not found")
    
    # Check TESS
    # TESS root may vary slightly. Let's look iteratively in TESS directory.
    tess_dirs = []
    if (data_path / "TESS").exists():
        for item in (data_path / "TESS").rglob('*'):
            if item.is_dir() and 'OAF' in item.name or 'YAF' in item.name:
                tess_dirs.append(item)
    if not tess_dirs: # Fallback to generic counting
        tess_dirs = list(data_path.glob("TESS/*/*"))
        if not tess_dirs:
             tess_dirs = list(data_path.glob("TESS/*"))
             if tess_dirs and tess_dirs[0].is_file():
                 tess_dirs = []
             
    if tess_dirs:
        print(f"✓ TESS: {len(tess_dirs)} folders")
    else:
        print("✗ TESS: Not found")
    
    # Check CREMA-D
    crema_audio = list(data_path.glob("CREMA-D/AudioWAV/*.wav"))
    if not crema_audio:
        crema_audio = list(data_path.glob("CREMA-D/*.wav"))
        
    if crema_audio:
        print(f"✓ CREMA-D: {len(crema_audio)} files")
    else:
        print("✗ CREMA-D: Not found")
    
    print("=" * 50)
    
    total_samples = len(ravdess_actors) * 960 + len(tess_dirs) * 200 + len(crema_audio)
    print(f"\nEstimated total samples: {total_samples:,}")
    print("✓ Dataset is ready for training!")
    return True

def main():
    print("=" * 60)
    print("Speech Emotion Recognition - Dataset Downloader")
    print("Datasets: RAVDESS, TESS, CREMA-D")
    print("=" * 60)
    
    # Create directories
    create_directories()
    
    # Download datasets
    ravdess_ok = download_ravdess_dataset()
    tess_ok = download_tess_dataset()
    crema_ok = download_crema_d_dataset()
    
    if not ravdess_ok:
        print("\n⚠️  Failed to download RAVDESS. This is required.")
        print("Try again manually from: https://zenodo.org/record/1188976")
        return
    
    if tess_ok:
        print("✓ TESS available")
    else:
        print("⚠️  TESS not available (optional)")
    
    if crema_ok:
        print("✓ CREMA-D available")
    else:
        print("⚠️  CREMA-D not available (optional)")
    
    # Organize dataset
    organize_dataset()
    
    print("\n" + "=" * 60)
    print("Ready to proceed with model training!")
    print("Run: python train_model.py")
    print("=" * 60)

if __name__ == "__main__":
    main()
