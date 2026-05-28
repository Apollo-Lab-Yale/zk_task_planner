#!/bin/bash
# Ultra-fast compression script
# Usage: ./quick_compress.sh <directory_to_compress>

if [ $# -eq 0 ]; then
    echo "Usage: $0 <directory_to_compress>"
    echo "Example: $0 ./raw_data"
    exit 1
fi

SOURCE_DIR="$1"
OUTPUT_FILE="${SOURCE_DIR##*/}_compressed.tar.gz"

echo "🚀 Fast compressing $SOURCE_DIR..."
echo "📁 Output: $OUTPUT_FILE"

# Use pigz if available (parallel gzip), otherwise regular gzip
if command -v pigz >/dev/null 2>&1; then
    echo "⚡ Using pigz (parallel compression)"
    tar -cf - "$SOURCE_DIR" | pigz -1 > "$OUTPUT_FILE"
else
    echo "⚡ Using gzip (fast compression level)"
    tar -czf "$OUTPUT_FILE" "$SOURCE_DIR"
fi

# Show results
ORIGINAL_SIZE=$(du -sb "$SOURCE_DIR" | cut -f1)
COMPRESSED_SIZE=$(stat -c%s "$OUTPUT_FILE")
RATIO=$(echo "scale=1; (1 - $COMPRESSED_SIZE / $ORIGINAL_SIZE) * 100" | bc -l 2>/dev/null || echo "N/A")

echo "✅ Done!"
echo "📊 Original: $(numfmt --to=iec $ORIGINAL_SIZE)"
echo "📊 Compressed: $(numfmt --to=iec $COMPRESSED_SIZE)" 
echo "📊 Reduction: ${RATIO}%"
echo "📁 File: $OUTPUT_FILE"