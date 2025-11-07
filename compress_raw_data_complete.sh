#!/bin/bash
# Complete compression of raw_data directory
# Handles large datasets efficiently

SOURCE_DIR="raw_data"
OUTPUT_PREFIX="raw_data_complete"

echo "🚀 Starting complete compression of $SOURCE_DIR (21GB)..."
echo "📊 This will take several minutes..."

# Check available compression tools
if command -v pigz >/dev/null 2>&1; then
    echo "⚡ Using pigz (parallel compression) - FASTEST"
    time tar -cf - "$SOURCE_DIR" | pigz -1 > "${OUTPUT_PREFIX}.tar.gz"
elif command -v 7z >/dev/null 2>&1; then
    echo "⚡ Using 7z (best compression)"
    time 7z a -t7z -m0=lzma2 -mx=3 "${OUTPUT_PREFIX}.7z" "$SOURCE_DIR"
else
    echo "⚡ Using standard gzip"
    time tar -czf "${OUTPUT_PREFIX}.tar.gz" "$SOURCE_DIR"
fi

# Show results
echo ""
echo "✅ Compression complete!"
echo "📁 Output files:"
ls -lh raw_data_complete.*

# Calculate compression ratio
ORIGINAL_SIZE=$(du -sb "$SOURCE_DIR" | cut -f1)
for file in raw_data_complete.*; do
    if [ -f "$file" ]; then
        COMPRESSED_SIZE=$(stat -c%s "$file")
        RATIO=$(echo "scale=1; (1 - $COMPRESSED_SIZE / $ORIGINAL_SIZE) * 100" | bc -l 2>/dev/null || echo "N/A")
        echo "📊 $file: $(numfmt --to=iec $COMPRESSED_SIZE) (${RATIO}% reduction)"
    fi
done

echo ""
echo "🌐 Ready for Google Drive upload!"