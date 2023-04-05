import os
import inspect
import importlib.util

# Define the path to the directory containing the subdirectories
directory_path = '.'

with open('output.txt', 'w') as f:
    print("Printing the files from which classes could not be extracted:-")

    # Loop over the subdirectories
    for root, dirs, files in os.walk(directory_path):
        # Loop over the files in the directory
        for file in files:
            # Check if the file is a Python file
            try:
                if file.endswith('.py'):
                    # Get the module name
                    module_name = os.path.splitext(file)[0]
                    # Get the full path to the module
                    module_path = os.path.join(root, file)
                    # Load the module as a spec
                    print(module_path)
                    module_spec = importlib.util.spec_from_file_location(module_name, module_path)
                    module = importlib.util.module_from_spec(module_spec)
                    module_spec.loader.exec_module(module)
                    f.write("----------------------------------------------------------\n")
                    f.write(f"{module_path}\n")
                    f.write("----------------------------------------------------------\n")
                    # Get the classes defined in the module
                    classes = inspect.getmembers(module, inspect.isclass)
                    # Loop over the classes
                    for class_tuple in classes:
                        # Get the class name and object
                        class_name = class_tuple[0]
                        class_obj = class_tuple[1]
                        # Check if the class was defined in the current module
                        if class_obj.__module__ == module_name:
                            # Get the functions in the class
                            functions = inspect.getmembers(class_obj, inspect.isfunction)
                            # Print the class name and associated functions
                            f.write(f'Class: {class_name}\n')
                            for function in functions:
                                f.write(f'{function[0]}, ')
                            f.write("\n")
                    f.write("\n")

                    only_functions = inspect.getmembers(module, inspect.isfunction)
                    f.write("Functions Outside Class: \n")
                    for func in only_functions:
                        
                        if(func[1].__module__==module_name):
                            f.write(f'    {func[0]}\n')
                    f.write("\n")
            except Exception as e:
                print(module_path, e)