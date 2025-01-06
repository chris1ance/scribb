"""
Using `os.path.join` with `__file__`:
* A common use case is constructing paths relative to the current script’s location:
* See `get_script_loc.py` for an example.
"""

import os


def main():
    # Get the directory where the current script is located
    # NOTE: if you're running code in an interactive Python shell (like IPython or Jupyter notebook), __file__ won't be defined.
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Construct a path to a data file in a 'data' subdirectory
    data_file = os.path.join(script_dir, "data", "dataset.csv")

    print(data_file)


if __name__ == "__main__":
    main()
