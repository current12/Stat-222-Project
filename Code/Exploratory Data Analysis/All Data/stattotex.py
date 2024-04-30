import os

def stattotex(number, number_name, filename):
    '''
    Function that takes number and associated name from Python and saves into a file that allows for easy-read-in and later formatting in LaTeX.
    '''

    # Creating the LaTeX command

    # If prior file exists
    # Check for f"\\newcommand{{\\{number_name}}}" in the file
    # If it exists, replace the entire line with the new command
    # If it does not exist, set command = f"\\newcommand{{\\{number_name}}}{{{number}}}"
    if os.path.exists(filename):
        with open(filename, "r") as file:
            if f"\\newcommand{{\\{number_name}}}" in file.read():
                command = f"\\renewcommand{{\\{number_name}}}{{{number}}}"
            else:
                command = f"\\newcommand{{\\{number_name}}}{{{number}}}"
    else:
        command = f"\\newcommand{{\\{number_name}}}{{{number}}}"

    # Writing the command to the file
    with open(filename, "a") as file:
        file.write(command + "\n")
