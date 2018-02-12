import tkinter
from tkinter import ttk, messagebox
from tkinter.scrolledtext import ScrolledText
import os
import sys
import subprocess
import threading

root = tkinter.Tk()

root.wm_title("Pyrgos")
root.resizable(width='TRUE', height='TRUE')
root.wm_geometry("%dx%d%+d%+d" % (720, 480, 0, 0))

tree = ttk.Treeview(root)

tree["columns"] = "type"
tree.pack(expand=tkinter.YES, fill=tkinter.BOTH)

tree.column("type", width=200)
tree.heading("type", text="program type")

elems = []


def remove_string_from_end(source_string, end_string):
    return source_string.replace(end_string, '')


def make_filepath_from_dir_and_file(dirname, filename, offset="../", sep="/"):
    fname_as_is = offset + dirname + sep + filename
    return fname_as_is


def replace_spacebar_for_console_input(filepath):
    return filepath.replace(' ', '\ ')

for dname in [dir_name for dir_name in os.listdir("../") if os.path.isdir("../" + dir_name)]:
    elems.append(tree.insert("", 1, str(dname), text=str(dname)))
    for fname in os.listdir("../" + dname):
        full_fname = make_filepath_from_dir_and_file(dname, fname)
        if os.path.isfile(full_fname):
            tree.insert(elems[len(elems)-1], 2, full_fname, text=str(fname))

tree.pack()

nb = ttk.Notebook(root)

# adding Frames as pages for the ttk.Notebook
# first page, which would get widgets gridded into it
page1 = ttk.Frame(nb)

# second page
text = ScrolledText(page1)
text.pack(expand=1, fill="both")
nb.add(page1, text='Output')

def run_button_click():
    """
    On button click we can receive a string from tree which contains
    all strings concatenated together

    We get rid of child string to get parent, then put them together to
    make a filepath.

    We launch a filepath afterwards
    :return:
    """
    cur_item = tree.focus()
    text.delete('1.0', tkinter.END)

    def launch_script():
        try:
            launcher = ''
            if cur_item.endswith('.py'):
                launcher = 'python3'

            res = subprocess.Popen([launcher, cur_item], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            output, error = res.communicate()
            if output:
                text.insert(tkinter.END, output)
            if error:
                text.insert(tkinter.END, error)

                # for windows
                # result = subprocess.check_output(["cmd.exe", batcmd])
        except PermissionError:
            error_message = "Cannot launch executable, permission error. Check configs. Use chmod u+x or equivalent. \n"
            text.insert(tkinter.END, error_message)
        except OSError as e:
            text.insert(tkinter.END, "OSError > ", e.strerror)
        except:
            text.insert(tkinter.END, "Cannot launch executable")

    launcher_thread = threading.Thread(target=launch_script())
    if os.path.isfile(cur_item):
        launcher_thread.start()

button = ttk.Button(root, command=run_button_click, text="run selected")

button.pack()
nb.pack()

def about():
    pyrgos_description = \
"""
The solution can be used for market calibration, risk management and option pricing.

The code and algorithms provided are products of the research supported by RFBR grant (project 15-32-01390).

For license information see "license" file.

Authors: A. Grechko, O. Kudryavtsev and V. Rodochenko.

Inwise-Systems, 2015-2017.
"""
    messagebox.showinfo("Pyrgos", pyrgos_description)


def help():
    readme_text = \
"""
The form you use is a GUI for Pyrgos system.

The files are being launched by your OS default launcher.

The tree structure updated every time you launch the program.

For detailed information on each module see the respective "readme"

Authors: A. Grechko, O. Kudryavtsev and V. Rodochenko.

Inwise-Systems, 2015-2017.
"""
    messagebox.showinfo("Help", readme_text)

menubar = tkinter.Menu(root)

# create a pulldown menu, and add it to the menu bar
helpmenu = tkinter.Menu(menubar, tearoff=0)
helpmenu.add_command(label="Help", command=help)
helpmenu.add_command(label="About", command=about)
helpmenu.add_separator()
helpmenu.add_command(label="Exit", command=root.quit)
menubar.add_cascade(label="Help", menu=helpmenu)


# display the menu
root.config(menu=menubar)

root.mainloop()