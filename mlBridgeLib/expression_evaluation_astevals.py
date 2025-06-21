from asteval import Interpreter
import re

# Test cases with expected results
verification_cases = {
    # Arithmetic
    "a + b": (8, "a b +"),
    "a * b": (15, "a b *"),
    "a ** b": (125, "a b **"),
    
    # Bitwise
    "a & b": (1, "a b &"),
    "a | b": (7, "a b |"),
    "a ^ b": (6, "a b ^"),
    "~a": (-6, "a ~"),
    "a << b": (40, "a b <<"),
    "a >> b": (0, "a b >>"),
    
    # Logical
    "a > b": (True, "a b >"),
    "not a": (False, "a not"),
    "a and b": (3, "a b and"),
    "a or b": (5, "a b or"),
    
    # Complex expressions
    "a + b * c": (11, "a b c * +"),
    "(a + b) * c": (16, "a b + c *"),
    "a & b | c": (3, "a b & c |"),
    "a ** b ** c": (1953125, "a b c ** **"),
    "1 <= x <= 10": (True, "1 x <= x 10 <= and")
}

def evaluate_expression(expr, values=None):
    """Evaluate expression using asteval."""
    if values is None:
        values = {'a': 5, 'b': 3, 'c': 2, 'd': 4, 'e': 1, 'x': 5, 'y': 3, 'z': 2}
    
    aeval = Interpreter()
    # Add variables to asteval's symbol table
    for name, value in values.items():
        aeval.symtable[name] = value
    
    return aeval.eval(expr)

def run_tests(verification_cases, test_values=None):
    """Run all test cases and return error count."""
    if test_values is None:
        test_values = {'a': 5, 'b': 3, 'c': 2, 'd': 4, 'e': 1, 'x': 5, 'y': 3, 'z': 2}
        
    error_count = 0
    print("\nTesting expressions...")
    
    for infix, (expected_result, expected_postfix) in verification_cases.items():
        try:
            # Evaluate using asteval
            result = evaluate_expression(infix, test_values)
            
            # Print results
            print(f"\nExpression: {infix}")
            print(f"Expected: {expected_result}, Got: {result}")
            
            if result != expected_result:
                print("ERROR: Results don't match expectations!")
                error_count += 1
                
        except Exception as e:
            print(f"\nError evaluating {infix}: {str(e)}")
            error_count += 1
    
    print(f"\nTotal errors: {error_count}")
    return error_count

if __name__ == '__main__':
    run_tests(verification_cases)