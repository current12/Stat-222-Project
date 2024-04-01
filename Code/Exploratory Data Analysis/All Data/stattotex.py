import os

# Function that takes number and associated name from Python and saves into a file that allows for easy-read-in and later formatting in LaTeX
def stattotex(number, number_name, filename, clear_preexisting=True):

    # Clearing the file if specified
    if clear_preexisting:
        # Delete the file if it exists
        if os.path.exists(filename):
            os.remove(filename)

    # Creating the LaTeX command
    command = f"\\newcommand{{\\{number_name}}}{{{number}}}"

    # Writing the command to the file
    with open(filename, "a") as file:
        file.write(command + "\n")
