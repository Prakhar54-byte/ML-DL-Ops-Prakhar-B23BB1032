import yaml
import glob
import sys

def validate_yaml_files():
    # Find all yaml files in current directory
    files = ["rides_contract.yaml", "orders_contract.yaml", 
             "thermostat_contract.yaml", "fintech_contract.yaml"]
    
    print("🔍 Starting YAML Validation...\n")
    
    all_passed = True
    
    for filename in files:
        try:
            with open(filename, 'r') as f:
                # Attempt to parse the YAML
                yaml.safe_load(f)
                print(f" {filename}: VALID")
        except FileNotFoundError:
            print(f" {filename}: NOT FOUND")
            all_passed = False
        except yaml.YAMLError as e:
            print(f" {filename}: INVALID SYNTAX")
            print(f"   Error: {e}")
            all_passed = False
            
    print("\n" + "="*30)
    if all_passed:
        print("🎉 All files are valid. Ready to submit!")
    else:
        print("⚠️  Fix syntax errors before submitting.")

if __name__ == "__main__":
    validate_yaml_files()