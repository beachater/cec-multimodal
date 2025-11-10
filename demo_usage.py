#!/usr/bin/env python
"""
Demo script showing run_experiments.py usage examples.
"""

print("=" * 70)
print("AMLP-RS-CMSA-ESII Experiment Runner - Usage Examples")
print("=" * 70)

print("\n📋 INTERACTIVE MODE")
print("-" * 70)
print("Command:")
print("  python run_experiments.py")
print("\nWhat it does:")
print("  • Displays a menu with options")
print("  • Choose to run single/all functions")
print("  • Select custom seed ranges")
print("  • Post-process existing results")
print("  • User-friendly for beginners")

print("\n" + "=" * 70)
print("\n🎯 COMMAND-LINE EXAMPLES")
print("-" * 70)

examples = [
    {
        "title": "Run single function (seeds 1-30)",
        "command": "python run_experiments.py --function 16",
        "description": "Runs DMMOP16 with 30 seeds, auto-generates CSVs"
    },
    {
        "title": "Run all functions (720 experiments)",
        "command": "python run_experiments.py --all",
        "description": "Runs all 24 functions × 30 seeds = 720 total runs"
    },
    {
        "title": "Custom seed range",
        "command": "python run_experiments.py --function 16 --seeds 1 5",
        "description": "Run DMMOP16 with only seeds 1-5 (for quick testing)"
    },
    {
        "title": "Post-process existing results",
        "command": "python run_experiments.py --postprocess-only --function 16",
        "description": "Convert existing .pkl files to CSV without re-running"
    },
    {
        "title": "Post-process all functions",
        "command": "python run_experiments.py --postprocess-only --all",
        "description": "Generate CSVs for all existing result files"
    },
]

for i, example in enumerate(examples, 1):
    print(f"\nExample {i}: {example['title']}")
    print(f"  Command: {example['command']}")
    print(f"  Description: {example['description']}")

print("\n" + "=" * 70)
print("\n📁 OUTPUT FILES")
print("-" * 70)
print("Results are saved in the 'result/' directory:")
print("  • result_F16_D5_seed1.pkl  - Raw experiment data")
print("  • result_F16_D5_seed1.csv  - Processed data for analysis")
print("\nCSV columns:")
print("  • Column 1: Function evaluations")
print("  • Column 2: Best fitness per environment")

print("\n" + "=" * 70)
print("\n✅ WORKFLOW")
print("-" * 70)
print("1. Run experiments:")
print("   python run_experiments.py --function 16")
print("\n2. Results automatically processed to CSV")
print("\n3. Analyze CSV files with your favorite tools")
print("   (Excel, Python pandas, R, MATLAB, etc.)")

print("\n" + "=" * 70)
print("\n⚡ QUICK TIPS")
print("-" * 70)
print("• Test first: Use --seeds 1 5 for quick testing")
print("• Full benchmark: Use --all for complete CEC 2022 benchmark")
print("• Resumable: Interrupt anytime, already-completed runs are saved")
print("• Parallel: Each seed is independent (can run in parallel manually)")

print("\n" + "=" * 70)
print("\n🆘 NEED HELP?")
print("-" * 70)
print("  python run_experiments.py --help")
print("  python main.py --help")
print("\n" + "=" * 70)
print()
