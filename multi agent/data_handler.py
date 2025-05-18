import json

def load_puzzle_data(filename="puzzles.txt"):
    """
    Loads puzzle data from a JSON file.

    Args:
        filename (str, optional): The name of the file to load from.
            Defaults to "puzzles.txt".

    Returns:
        dict: A dictionary containing the puzzle data, or None if an error occurs.
    """
    try:
        with open(filename, "r") as f:
            data = json.load(f)
        return data
    except FileNotFoundError:
        print(f"Error: File '{filename}' not found.")
        return None
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON format in '{filename}'.")
        return None
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return None

if __name__ == '__main__':
    # Example usage:
    # Create a dummy puzzles.txt file
    dummy_data = {
        "level1": {
            "description": "Simple addition",
            "problem": "2 + 2 = ?",
            "solution": "4"
        },
        "level2": {
            "description": "Basic subtraction",
            "problem": "5 - 3 = ?",
            "solution": "2"
        }
    }
    with open("puzzles.txt", "w") as f:
        json.dump(dummy_data, f, indent=4)

    puzzle_data = load_puzzle_data()
    if puzzle_data:
        print("Puzzle data loaded successfully:")
        print(puzzle_data)
    else:
        print("Failed to load puzzle data.")