#!/usr/bin/env python3
"""
Vendor Research Tool - A command-line tool for deep research on vendors.
"""
import argparse
import json
import os
import sys
from dotenv import load_dotenv
from pathlib import Path

# Add the parent directory to the path so we can import from scripts
sys.path.append(str(Path(__file__).parent.absolute()))

from scripts.vendor_research_tool import VendorResearchTool

def load_environment():
    """Load environment variables from .env file if it exists."""
    env_path = Path(__file__).parent / '.env'
    if env_path.exists():
        load_dotenv(dotenv_path=env_path)
    else:
        print("Warning: No .env file found. Using system environment variables.")

def validate_environment():
    """Validate that required environment variables are set."""
    required_vars = [
        'SERPER_API_KEY',
        'DEEPSEEK_API_KEY'
    ]
    
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    
    if missing_vars:
        print(f"Error: The following required environment variables are not set: {', '.join(missing_vars)}")
        print("Please create a .env file or set these environment variables.")
        print("See .env.example for reference.")
        sys.exit(1)

async def run_research(vendor_name: str, pain_point: str, url: str = None) -> str:
    """Run the vendor research asynchronously."""
    research_tool = VendorResearchTool()
    
    # Prepare the input as a dictionary for the tool
    tool_input = {
        'vendor_name': vendor_name,
        'pain_point': pain_point
    }
    
    if url:
        tool_input['url'] = url
    
    # Call the tool asynchronously
    return await research_tool.forward(**tool_input)

def main():
    """Main entry point for the script."""
    # Load environment variables
    load_environment()
    validate_environment()
    
    # Set up argument parsing
    parser = argparse.ArgumentParser(description='Research a vendor and their offerings.')
    parser.add_argument('vendor_name', type=str, help='Name of the vendor to research')
    parser.add_argument('pain_point', type=str, help='Pain point or use case to focus the research')
    parser.add_argument('--url', type=str, help='Optional: Vendor website URL', default=None)
    parser.add_argument('--output', '-o', type=str, help='Output file path (JSON)', default=None)
    parser.add_argument('--pretty', action='store_true', help='Pretty print the JSON output')
    
    args = parser.parse_args()
    
    try:
        # Run the research
        print(f"Researching {args.vendor_name}...")
        
        # Run the async function
        import asyncio
        result = asyncio.run(run_research(args.vendor_name, args.pain_point, args.url))
        
        # Parse the result to pretty print if needed
        try:
            result_json = json.loads(result)
            
            # Save to file if output path is provided
            if args.output:
                with open(args.output, 'w', encoding='utf-8') as f:
                    json.dump(result_json, f, indent=2 if args.pretty else None)
                print(f"Results saved to {args.output}")
            else:
                # Print to console
                indent = 2 if args.pretty else None
                print(json.dumps(result_json, indent=indent, ensure_ascii=False))
                
        except json.JSONDecodeError:
            # If the result isn't valid JSON, print as-is
            print(result)
            
    except Exception as e:
        print(f"An error occurred: {str(e)}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
