import torch
import time
import numpy as np
from datetime import datetime


def benchmark_matmul(device, size=10000, iterations=5, name="Device"):
    """Benchmark matrix multiplication on a specific device"""
    print(f"\n{'='*60}")
    print(f"Benchmarking on {name}: {device}")
    print(f"{'='*60}")

    # Warm-up
    a = torch.rand(size, size, device=device)
    b = torch.rand(size, size, device=device)
    _ = torch.matmul(a, b)

    if device.type == 'cuda':
        torch.cuda.synchronize(device)

    times = []
    for i in range(iterations):
        a = torch.rand(size, size, device=device)
        b = torch.rand(size, size, device=device)

        if device.type == 'cuda':
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)

            start.record()
            _ = torch.matmul(a, b)
            end.record()

            torch.cuda.synchronize(device)
            elapsed = start.elapsed_time(end) / 1000  # Convert to seconds
        else:
            start = time.time()
            _ = torch.matmul(a, b)
            end = time.time()
            elapsed = end - start

        times.append(elapsed)
        print(f"  Iteration {i+1}/{iterations}: {elapsed:.4f} seconds")

    avg_time = np.mean(times)
    std_time = np.std(times)
    tflops = (2 * size**3) / (avg_time * 1e12)  # Approximate TFLOPS

    print(f"\n  Average time: {avg_time:.4f} ± {std_time:.4f} seconds")
    print(f"  Performance: {tflops:.2f} TFLOPS")

    return avg_time, std_time, tflops


def benchmark_multi_gpu(size=10000, iterations=5):
    """Benchmark on all available GPUs"""
    if not torch.cuda.is_available():
        print("CUDA not available!")
        return

    num_gpus = torch.cuda.device_count()
    print(f"\n{'#'*60}")
    print(f"# Multi-GPU Benchmark - {num_gpus} GPUs Available")
    print(f"# Matrix size: {size}x{size}")
    print(f"# Iterations: {iterations}")
    print(f"{'#'*60}")

    results = {}

    # Benchmark each GPU individually
    for gpu_id in range(num_gpus):
        device = torch.device(f'cuda:{gpu_id}')
        gpu_name = torch.cuda.get_device_name(gpu_id)
        gpu_mem = torch.cuda.get_device_properties(gpu_id).total_memory / 1e9

        print(f"\nGPU {gpu_id}: {gpu_name} ({gpu_mem:.1f} GB)")

        avg_time, std_time, tflops = benchmark_matmul(
            device, size, iterations, f"GPU {gpu_id}"
        )
        results[f'GPU_{gpu_id}'] = {
            'avg_time': avg_time,
            'std_time': std_time,
            'tflops': tflops,
            'name': gpu_name
        }

    # Parallel execution on all GPUs
    print(f"\n{'='*60}")
    print("Parallel execution on ALL GPUs")
    print(f"{'='*60}")

    tensors_a = [torch.rand(size, size, device=f'cuda:{i}') for i in range(num_gpus)]
    tensors_b = [torch.rand(size, size, device=f'cuda:{i}') for i in range(num_gpus)]

    parallel_times = []
    for iteration in range(iterations):
        torch.cuda.synchronize()
        start = time.time()

        # Launch operations on all GPUs
        results_tensors = [
            torch.matmul(tensors_a[i], tensors_b[i])
            for i in range(num_gpus)
        ]

        torch.cuda.synchronize()
        end = time.time()
        elapsed = end - start
        parallel_times.append(elapsed)
        print(f"  Iteration {iteration+1}/{iterations}: {elapsed:.4f} seconds")

    avg_parallel = np.mean(parallel_times)
    std_parallel = np.std(parallel_times)
    parallel_tflops = (2 * size**3 * num_gpus) / (avg_parallel * 1e12)

    print(f"\n  Average time (parallel): {avg_parallel:.4f} ± {std_parallel:.4f} seconds")
    print(f"  Aggregate performance: {parallel_tflops:.2f} TFLOPS")

    return results


def benchmark_cpu(size=1000000, iterations=5, num_threads=None):
    """Benchmark on CPU"""
    print(f"\n{'#'*60}")
    print(f"# CPU Benchmark")
    print(f"# Matrix size: {size}x{size}")
    print(f"# Iterations: {iterations}")
    print(f"{'#'*60}")

    if num_threads:
        torch.set_num_threads(num_threads)
        print(f"Using {num_threads} threads")
    else:
        print(f"Using {torch.get_num_threads()} threads (default)")

    device = torch.device('cpu')
    avg_time, std_time, tflops = benchmark_matmul(
        device, size, iterations, "CPU"
    )

    return {'avg_time': avg_time, 'std_time': std_time, 'tflops': tflops}


def main():
    print("="*60)
    print("HEALPix_BigEarthNet - GPU/CPU Benchmark")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60)
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"Number of CUDA devices: {torch.cuda.device_count()}")

        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            print(f"  GPU {i}: {props.name}")
            print(f"    - Memory: {props.total_memory / 1e9:.1f} GB")
            print(f"    - Compute Capability: {props.major}.{props.minor}")

    # Run benchmarks
    matrix_size = 10000
    num_iterations = 5

    # GPU benchmark (if available)
    if torch.cuda.is_available():
        gpu_results = benchmark_multi_gpu(matrix_size, num_iterations)

    # CPU benchmark
    cpu_results = benchmark_cpu(matrix_size, num_iterations, num_threads=96)

    # Summary
    print(f"\n{'#'*60}")
    print("# BENCHMARK SUMMARY")
    print(f"{'#'*60}")

    if torch.cuda.is_available():
        print("\nGPU Results:")
        for gpu_name, result in gpu_results.items():
            print(f"  {gpu_name}: {result['tflops']:.2f} TFLOPS")

    print(f"\nCPU Results: {cpu_results['tflops']:.2f} TFLOPS")
    print("\n" + "="*60)


if __name__ == "__main__":
    main()