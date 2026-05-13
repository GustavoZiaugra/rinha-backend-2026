import gzip
import struct

with gzip.open('data/index.bin.gz', 'rb') as f:
    data = f.read()

print(f'Decompressed size: {len(data)} bytes = {len(data)/1024/1024:.1f} MB')
magic = data[:4]
print(f'Magic: {magic}')

if magic == b'IVF1':
    n = struct.unpack_from('<I', data, 4)[0]
    k = struct.unpack_from('<I', data, 8)[0]
    d = struct.unpack_from('<I', data, 12)[0]
    print(f'n={n}, k={k}, d={d}')
    
    centroids_size = 4 * d * k
    offsets_pos = 16 + centroids_size
    offsets_size = 4 * (k + 1)
    labels_pos = offsets_pos + offsets_size
    
    offsets_data = data[offsets_pos:offsets_pos + offsets_size]
    offsets = struct.unpack_from(f'<{k+1}I', offsets_data)
    padded_n = offsets[k]
    print(f'padded_n = {padded_n}')
    
    total_blocks = padded_n // 8
    blocks_size = total_blocks * d * 8 * 2  # i16 = 2 bytes
    labels_size = padded_n
    
    print(f'Centroids: {centroids_size} bytes ({centroids_size/1024:.1f} KB)')
    print(f'Offsets: {offsets_size} bytes')
    print(f'Labels: {labels_size} bytes ({labels_size/1024:.1f} KB)')
    print(f'Blocks: {blocks_size} bytes ({blocks_size/1024/1024:.1f} MB)')
    print(f'Total computed: {16 + centroids_size + offsets_size + labels_size + blocks_size} bytes')
    print(f'Actual decompressed: {len(data)} bytes')
    
    # Check cluster sizes
    cluster_sizes = [offsets[i+1] - offsets[i] for i in range(k)]
    avg_size = sum(cluster_sizes) / k
    max_size = max(cluster_sizes)
    min_size = min(cluster_sizes)
    print(f'Cluster sizes: avg={avg_size:.1f}, max={max_size}, min={min_size}')
    
    # Count how many non-empty clusters
    non_empty = sum(1 for s in cluster_sizes if s > 0)
    print(f'Non-empty clusters: {non_empty}/{k}')
