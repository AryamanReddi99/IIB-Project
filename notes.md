# Notes

## Common git commands

(adds changes to tracked files, commits with set message, pushes):  
git add -u ; git commit -m "Testing Changes" ; git push

## Virtual Environements

When making a new python project, use a virtual environment. This will create a local copy of a clean python interpreter from elsewhere on your machine. This local copy can then be set as the interpreter for the project, and packages (including custom packages) should be installed with this interpreter instead of the global one. This keeps things neat as only the requires add-ons for each project are installed.  
To make a new virtual environment, use "py venv \<name\>" where py (or python) points to the clean interpreter you want to be copied. The virtual environment can be activated with "\<name\>/scripts/activate" and deactivated with "deactivate". The terminal should indicate when we're in a virtual environment. Once here, use py -m pip install \<pkg_name\>. To use the virtual environment when running files, use ctrl+shift+p, "Python: Select Interpreter", choose local interpreter copy. 

## Custom packages

Working on a project will sometimes require designing a custom package with special functions, classes, and resources. To be accessible by the rest of the project, this package will need to be on the python path. This can be achieved by simply placing it within the virtual environment of the project. However, for some reason VSCode doesn't like running the debugger for python files on the path - this makes debugging custom packes tedious. A hacky solution to this is to simply symlink the custom package one level up - that is, on the same level of the file heirarchy as the virtual environment - which allows the debugger to run AND retains the custom package on the path for use. To do this automatically
