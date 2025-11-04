#!/usr/bin/env python3
"""
Helper script to create .env file from .env.example
"""
import shutil
from pathlib import Path

backend_dir = Path(__file__).parent.resolve()
env_example = backend_dir / ".env.example"
env_file = backend_dir / ".env"

if env_file.exists():
    print(f"⚠️  .env file already exists at {env_file}")
    response = input("Do you want to overwrite it? (y/N): ")
    if response.lower() != 'y':
        print("Aborted.")
        exit(0)

if not env_example.exists():
    print(f"❌ Error: .env.example not found at {env_example}")
    exit(1)

# Copy .env.example to .env
shutil.copy(env_example, env_file)
print(f"✅ Created .env file at {env_file}")
print(f"\n📝 Next steps:")
print(f"   1. Edit {env_file} and add your OPENAI_API_KEY")
print(f"   2. Save the file")
print(f"   3. Run the pipeline: python run_full_pipeline.py ...")

