
"""
Simple runner for FlashAttention-2 implementation
"""

import os
import sys
import subprocess

def print_section(title):
    print("\n" + "="*60)
    print(title)
    print("="*60)

def check_dependencies():
    """Check if required packages are installed"""
    print_section("CHECKING DEPENDENCIES")
    
    packages = {
        'mindspore': 'MindSpore Framework',
        'numpy': 'NumPy for numerical operations'
    }
    
    missing = []
    for package, description in packages.items():
        try:
            __import__(package)
            print(f"[OK] {package}: {description}")
        except ImportError:
            print(f"[MISSING] {package}: {description}")
            missing.append(package)
    
    return missing

def run_experiments():
    """Run the experiments"""
    print_section("RUNNING EXPERIMENTS")
    
    if not os.path.exists('experiments.py'):
        print("[ERROR] experiments.py not found")
        return False
    
    try:
        print("Running FlashAttention-2 experiments...")
        result = subprocess.run(
            [sys.executable, 'experiments.py'],
            capture_output=True,
            text=True
        )
        
        print(result.stdout)
        
        if result.returncode != 0:
            print(f"[ERROR] Experiments failed with code {result.returncode}")
            if result.stderr:
                print("Error output:", result.stderr)
            return False
        
        return True
        
    except Exception as e:
        print(f"[ERROR] Running experiments: {e}")
        return False

def show_results():
    """Show results if they exist"""
    print_section("RESULTS SUMMARY")
    
    if os.path.exists('benchmark_results.json'):
        print("[OK] Benchmark results saved to 'benchmark_results.json'")
        
        try:
            import json
            with open('benchmark_results.json', 'r') as f:
                data = json.load(f)
            
            if data:
                print("\nBenchmark Results:")
                print("-" * 50)
                for result in data:
                    print(f"Seq_len {result['seq_len']}: "
                          f"Standard={result['std_time_ms']:.1f}ms, "
                          f"FA2={result['fa2_time_ms']:.1f}ms, "
                          f"Speedup={result['speedup']:.2f}x")
        except:
            print("[INFO] Could not parse results file")
    else:
        print("[INFO] No results file generated")

def main():
    print_section("FLASHATTENTION-2 IMPLEMENTATION")
    print("Running on MindSpore Framework")
    
    # Check dependencies
    missing = check_dependencies()
    if missing:
        print(f"\n[ERROR] Missing packages: {', '.join(missing)}")
        print("Please install with: pip install " + " ".join(missing))
        return
    
    # Check for required files
    required_files = ['flash_attention.py', 'experiments.py']
    for file in required_files:
        if not os.path.exists(file):
            print(f"[ERROR] Required file not found: {file}")
            print("Please create this file in the current directory")
            return
    
    # Run simple test first
    print_section("RUNNING QUICK TEST")
    
    if os.path.exists('minimal_test.py'):
        try:
            result = subprocess.run(
                [sys.executable, 'minimal_test.py'],
                capture_output=True,
                text=True
            )
            print(result.stdout)
            
            if result.returncode != 0:
                print("[WARN] Quick test had issues, but continuing...")
        except:
            print("[INFO] Skipping quick test")
    else:
        print("[INFO] No minimal_test.py found, skipping quick test")
    
    # Run experiments
    if not run_experiments():
        return
    
    # Show results
    show_results()
    
    print_section("COMPLETE")
    print("\n[SUCCESS] FlashAttention-2 implementation completed!")
    print("\nFiles generated:")
    print("  - benchmark_results.json (if benchmarks ran successfully)")
    print("\nTo run experiments again:")
    print("  python experiments.py")
    print("\nTo test basic functionality:")
    print("  python minimal_test.py")

if __name__ == "__main__":
    main()