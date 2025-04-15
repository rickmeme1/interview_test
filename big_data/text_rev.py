import os
import mmap
import tempfile
import time
import multiprocessing
from concurrent.futures import ThreadPoolExecutor
from functools import partial


def reverse_large_file(input_file, output_file, buffer_size=1024*1024, num_threads=4):
    start_time = time.time()
    with open(input_file, 'rb') as f:
        with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
            encoding = detect_file_encoding_from_mmap(mm)
            total_size = len(mm)
            chunk_ranges = [(i, min(i + buffer_size, total_size))
                            for i in range(0, total_size, buffer_size)]

            with ThreadPoolExecutor(max_workers=num_threads) as executor:
                reverse_func = partial(reverse_chunk_range, mm=mm, encoding=encoding)
                temp_files = list(executor.map(reverse_func, chunk_ranges))

            with open(output_file, 'wb') as out_f:
                for temp_file in reversed(temp_files):
                    with open(temp_file, 'rb') as in_f:
                        while True:
                            data = in_f.read(buffer_size)
                            if not data:
                                break
                            out_f.write(data)
                    os.remove(temp_file)
    elapsed_time = time.time() - start_time
    return elapsed_time


def reverse_large_file_mp(input_file, output_file, buffer_size=1024*1024, num_processes=4):
    start_time = time.time()
    with open(input_file, 'rb') as f:
        with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
            encoding = detect_file_encoding_from_mmap(mm)
            total_size = len(mm)
            chunk_ranges = [(i, min(i + buffer_size, total_size))
                            for i in range(0, total_size, buffer_size)]

            with multiprocessing.Pool(processes=num_processes) as pool:
                reverse_func = partial(reverse_chunk_range_from_file,
                                       mmap_file=input_file, encoding=encoding, chunk_size=buffer_size)
                temp_files = pool.map(reverse_func, chunk_ranges)

            with open(output_file, 'wb') as out_f:
                for temp_file in reversed(temp_files):
                    with open(temp_file, 'rb') as in_f:
                        while True:
                            data = in_f.read(buffer_size)
                            if not data:
                                break
                            out_f.write(data)
                    os.remove(temp_file)
    elapsed_time = time.time() - start_time
    return elapsed_time


def reverse_chunk_range(range_info, mm, encoding):
    start, end = range_info
    chunk = mm[start:end]
    reversed_data = reverse_chunk(chunk, encoding)
    temp_file = tempfile.NamedTemporaryFile(delete=False)
    temp_file.write(reversed_data)
    temp_file.close()
    return temp_file.name


def reverse_chunk_range_from_file(range_info, mmap_file, encoding, chunk_size):
    start, end = range_info
    with open(mmap_file, 'rb') as f:
        f.seek(start)
        chunk = f.read(end - start)
    reversed_data = reverse_chunk(chunk, encoding)
    temp_file = tempfile.NamedTemporaryFile(delete=False)
    temp_file.write(reversed_data)
    temp_file.close()
    return temp_file.name


def detect_file_encoding_from_mmap(mm):
    header = mm[:3]
    if header == b'\xef\xbb\xbf':
        return 'utf-8'
    return 'big5'


def reverse_chunk(chunk, encoding):
    if encoding == 'utf-8':
        reversed_bytes = bytearray()
        i = len(chunk) - 1
        while i >= 0:
            if (chunk[i] & 0x80) == 0:
                reversed_bytes.append(chunk[i])
                i -= 1
            else:
                start = i
                while start >= 0 and (chunk[start] & 0xC0) == 0x80:
                    start -= 1
                reversed_bytes.extend(chunk[start:i+1])
                i = start - 1
        return bytes(reversed_bytes)
    else:
        return chunk[::-1]


def benchmark_all(input_file, base_output_file):
    threads_or_procs = [1, 2, 4, 8]
    buffer_sizes = [1*1024*1024,2*1024*1024, 4*1024*1024, 8*1024*1024]

    results = []

    for n in threads_or_procs:
        for buf in buffer_sizes:
            label = f"T{n}_B{buf // 1024 // 1024}MB"
            out_file_thread = f"{base_output_file}_thread_{label}.txt"
            out_file_mp = f"{base_output_file}_mp_{label}.txt"

            print(f"\n🧵 Threads={n}, Buffer={buf//1024//1024}MB")
            t_time = reverse_large_file(input_file, out_file_thread, buffer_size=buf, num_threads=n)
            print(f"🟦 ThreadPoolExecutor 完成，用時: {t_time:.2f}s")

            print(f"🧠 Processes={n}, Buffer={buf//1024//1024}MB")
            p_time = reverse_large_file_mp(input_file, out_file_mp, buffer_size=buf, num_processes=n)
            print(f"🟥 multiprocessing 完成，用時: {p_time:.2f}s")

            results.append((n, buf, t_time, p_time))

    print("\n📊 總結：")
    print(f"{'Threads/Procs':>14} | {'BufferSize(MB)':>15} | {'Threads(s)':>12} | {'MP(s)':>10}")
    print("-" * 60)
    for n, buf, t_time, p_time in results:
        print(f"{n:>14} | {buf // 1024 // 1024:>15} | {t_time:>12.2f} | {p_time:>10.2f}")


if __name__ == "__main__":
    input_file = "D:/trading_code/py_trade/1gb_text_file.txt" #讀取檔案路徑
    base_output_file = "D:/trading_code/py_trade/reverse_benchmark"#輸出檔案路徑
    benchmark_all(input_file, base_output_file)
