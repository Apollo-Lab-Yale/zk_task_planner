#!/usr/bin/env python3
"""
Smart compression script for robotics raw data directories.
Optimizes compression based on file types and provides progress tracking.
"""

import os
import subprocess
import sys
import time
from pathlib import Path
import shutil
from typing import Dict, List, Tuple

class DataCompressor:
    def __init__(self, source_dir: str, output_dir: str = "compressed_data"):
        self.source_dir = Path(source_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # File type categories for optimal compression
        self.image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
        self.video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.webm'}
        self.json_extensions = {'.json'}
        self.text_extensions = {'.txt', '.log', '.csv', '.xml', '.yaml', '.yml'}
        self.binary_extensions = {'.pkl', '.npy', '.npz', '.h5', '.hdf5'}
        
    def analyze_directory(self) -> Dict[str, Dict]:
        """Analyze directory structure and file types."""
        analysis = {
            'images': {'files': [], 'size': 0},
            'videos': {'files': [], 'size': 0}, 
            'json': {'files': [], 'size': 0},
            'text': {'files': [], 'size': 0},
            'binary': {'files': [], 'size': 0},
            'other': {'files': [], 'size': 0}
        }
        
        print("🔍 Analyzing directory structure...")
        total_files = 0
        total_size = 0
        
        for root, dirs, files in os.walk(self.source_dir):
            for file in files:
                file_path = Path(root) / file
                try:
                    file_size = file_path.stat().st_size
                    total_files += 1
                    total_size += file_size
                    
                    # Categorize by extension
                    ext = file_path.suffix.lower()
                    if ext in self.image_extensions:
                        category = 'images'
                    elif ext in self.video_extensions:
                        category = 'videos'
                    elif ext in self.json_extensions:
                        category = 'json'
                    elif ext in self.text_extensions:
                        category = 'text'
                    elif ext in self.binary_extensions:
                        category = 'binary'
                    else:
                        category = 'other'
                    
                    analysis[category]['files'].append(file_path)
                    analysis[category]['size'] += file_size
                    
                except (OSError, FileNotFoundError):
                    continue
        
        # Print analysis
        print(f"\n📊 Directory Analysis:")
        print(f"Total files: {total_files:,}")
        print(f"Total size: {self._format_size(total_size)}")
        print("-" * 50)
        
        for category, data in analysis.items():
            if data['files']:
                print(f"{category.upper()}: {len(data['files'])} files, {self._format_size(data['size'])}")
        
        return analysis, total_size
    
    def _format_size(self, bytes_size: int) -> str:
        """Format bytes to human readable size."""
        for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
            if bytes_size < 1024.0:
                return f"{bytes_size:.1f} {unit}"
            bytes_size /= 1024.0
        return f"{bytes_size:.1f} PB"
    
    def compress_category(self, category: str, files: List[Path], compression_method: str = "auto") -> Tuple[str, int, int]:
        """Compress files of a specific category."""
        if not files:
            return "", 0, 0
            
        output_file = self.output_dir / f"{category}_data"
        
        # Choose compression method based on category and availability
        if compression_method == "auto":
            if category in ['images', 'videos']:
                # Images/videos are already compressed, use fast compression
                method = "gzip"
            else:
                # Text/JSON/binary benefit from strong compression
                method = "7z" if shutil.which('7z') else "gzip"
        else:
            method = compression_method
        
        print(f"\n🗜️  Compressing {category} ({len(files)} files) using {method}...")
        
        # Create file list for compression
        file_list_path = self.output_dir / f"{category}_files.txt"
        with open(file_list_path, 'w') as f:
            for file_path in files:
                f.write(f"{file_path}\n")
        
        start_time = time.time()
        original_size = sum(f.stat().st_size for f in files if f.exists())
        
        try:
            if method == "7z":
                output_file = output_file.with_suffix('.7z')
                cmd = [
                    '7z', 'a', '-t7z', '-m0=lzma2', '-mx=7',  # Balanced compression
                    str(output_file), f"@{file_list_path}"
                ]
            else:  # gzip/tar
                output_file = output_file.with_suffix('.tar.gz')
                cmd = [
                    'tar', '-czf', str(output_file), 
                    '-T', str(file_list_path), '--no-recursion'
                ]
            
            # Run compression with progress
            process = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
            )
            
            # Simple progress indicator
            while process.poll() is None:
                print(".", end="", flush=True)
                time.sleep(1)
            
            stdout, stderr = process.communicate()
            
            if process.returncode == 0:
                compressed_size = output_file.stat().st_size
                compression_ratio = (1 - compressed_size / original_size) * 100
                elapsed = time.time() - start_time
                
                print(f"\n✅ {category}: {self._format_size(original_size)} → {self._format_size(compressed_size)} "
                      f"({compression_ratio:.1f}% reduction) in {elapsed:.1f}s")
                
                # Clean up file list
                file_list_path.unlink()
                return str(output_file), original_size, compressed_size
            else:
                print(f"\n❌ Failed to compress {category}: {stderr}")
                return "", original_size, original_size
                
        except Exception as e:
            print(f"\n❌ Error compressing {category}: {e}")
            return "", original_size, original_size
    
    def compress_all(self, methods: Dict[str, str] = None) -> None:
        """Compress all categories with optimal methods."""
        if methods is None:
            methods = {
                'images': 'gzip',    # Already compressed
                'videos': 'gzip',    # Already compressed  
                'json': '7z',        # Text benefits from strong compression
                'text': '7z',        # Text benefits from strong compression
                'binary': '7z',      # Binary data varies
                'other': 'gzip'      # Unknown, use safe option
            }
        
        analysis, total_original = self.analyze_directory()
        
        print(f"\n🚀 Starting compression...")
        start_time = time.time()
        
        total_compressed = 0
        results = []
        
        for category, data in analysis.items():
            if data['files']:
                method = methods.get(category, 'gzip')
                output_file, orig_size, comp_size = self.compress_category(
                    category, data['files'], method
                )
                if output_file:
                    results.append((category, output_file, orig_size, comp_size))
                    total_compressed += comp_size
        
        # Summary
        total_time = time.time() - start_time
        overall_ratio = (1 - total_compressed / total_original) * 100 if total_original > 0 else 0
        
        print(f"\n🎉 Compression Complete!")
        print("=" * 60)
        print(f"Original size: {self._format_size(total_original)}")
        print(f"Compressed size: {self._format_size(total_compressed)}")
        print(f"Overall reduction: {overall_ratio:.1f}%")
        print(f"Time taken: {total_time:.1f} seconds")
        print(f"Output directory: {self.output_dir}")
        
        print(f"\n📁 Generated files:")
        for category, output_file, orig, comp in results:
            ratio = (1 - comp / orig) * 100 if orig > 0 else 0
            print(f"  {Path(output_file).name}: {self._format_size(comp)} ({ratio:.1f}% reduction)")
        
        # Create upload script
        self._create_upload_script(results)
    
    def _create_upload_script(self, results: List[Tuple]) -> None:
        """Create a script to help with Google Drive upload."""
        script_path = self.output_dir / "upload_to_gdrive.sh"
        
        with open(script_path, 'w') as f:
            f.write("#!/bin/bash\n")
            f.write("# Upload compressed files to Google Drive\n")
            f.write("# Install: pip install gdrive-cli OR use web interface\n\n")
            
            for category, output_file, orig, comp in results:
                filename = Path(output_file).name
                f.write(f"echo 'Uploading {filename}...'\n")
                f.write(f"# gdrive upload {output_file}\n")
                f.write(f"# OR drag-and-drop {filename} to drive.google.com\n\n")
        
        script_path.chmod(0o755)
        print(f"\n📝 Upload helper script created: {script_path}")

def main():
    if len(sys.argv) < 2:
        print("Usage: python compress_raw_data.py <source_directory> [output_directory]")
        print("Example: python compress_raw_data.py ./raw_data ./compressed")
        sys.exit(1)
    
    source_dir = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "compressed_data"
    
    if not Path(source_dir).exists():
        print(f"❌ Source directory not found: {source_dir}")
        sys.exit(1)
    
    compressor = DataCompressor(source_dir, output_dir)
    compressor.compress_all()

if __name__ == "__main__":
    main()