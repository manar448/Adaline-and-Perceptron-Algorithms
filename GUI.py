import tkinter
from tkinter import *
from tkinter import ttk
from tkinter import messagebox

gui_data = []


def submit():
    if not first_feature.get() or not second_feature.get() or not first_class.get() or not second_class.get() or \
            not learning_rate.get() or not epochs.get() or not mse.get() or not algo_var.get():
        tkinter.messagebox.showwarning(title="Error", message="Please select a value for each field.")
    first_feature_retrived = first_feature.get()
    second_feature_retrived = second_feature.get()
    first_class_retrived = first_class.get()
    second_class_retrived = second_class.get()
    if first_feature.get() == second_feature_retrived or first_class.get() == second_class.get():
        tkinter.messagebox.showwarning(title="Error",
                                       message="Please select two different feature and two  different classes.")
    else:
        learning_rate_retrived = learning_rate.get()
        epochs_retrived = epochs.get()
        mse_retrived = mse.get()
        model_bais = bias_retrived.get()
        if algo_var.get() == 1:
            algorithm_retrived = "Perceptron"
        elif algo_var.get() == 2:
            algorithm_retrived = "Adaline"

        gui_data.append(str(first_feature_retrived))
        gui_data.append(str(second_feature_retrived))
        gui_data.append(str(first_class_retrived))
        gui_data.append(str(second_class_retrived))
        gui_data.append(str(learning_rate_retrived))
        gui_data.append(str(epochs_retrived))
        gui_data.append(str(mse_retrived))
        gui_data.append(str(model_bais))
        gui_data.append(str(algorithm_retrived))
        tkinter.messagebox.showinfo(title="Error",
                                    message="Submit Successfully")
    '''
    print("first_feature_retrived : " + first_feature_retrived)
    print("second_feature_retrived : " + second_feature_retrived)
    print("first_class_retrived : " + first_class_retrived)
    print("second_class_retrived : " + second_class_retrived)
    print("learning_rate_retrived : " + learning_rate_retrived)
    print("epochs_retrived : " + epochs_retrived)
    print("mse_retrived : " + mse_retrived)
    print("bias_retrived : " + model_bais)
    print("Algorithm : " + algorithm_retrived)
    '''
    return gui_data


# text_color=#6a9d62
text_color = "#2f452b"
header_font = "18"
text_font = "6"

# Create Form
form = Tk()
form.geometry("600x600")
form.title("Task 1 - GUI")

# Create Frame 1 Features
#########################
frame1 = Frame(form)
frame1.pack()
model_features = LabelFrame(frame1, text="Features", font=header_font, fg=text_color)
model_features.grid(row=0, column=0, padx=10, pady=10)

lb1 = Label(model_features, text="First Feature", font=text_font, fg=text_color)
lb1.grid(row=0, column=0)
features = ["Area", "Perimeter", "MajorAxisLength", "MinorAxisLength", "roundnes"]
first_feature = ttk.Combobox(model_features, values=features, width="30")
first_feature.grid(row=1, column=0)

lb2 = Label(model_features, text="Second Feature", font=text_font, fg=text_color)
lb2.grid(row=0, column=1)
second_feature = ttk.Combobox(model_features, values=features, width="30")
second_feature.grid(row=1, column=1)

for widget in model_features.winfo_children():
    widget.grid_configure(padx=35, pady=10)

# Create Frame 2 Classes
#########################
frame2 = Frame(form)
frame2.pack()
model_classes = LabelFrame(frame2, text="Classes ", font=header_font, fg=text_color)
model_classes.grid(row=1, column=0, pady=10)

lb3 = Label(model_classes, text="First Class", font=text_font, fg=text_color)
lb3.grid(row=1, column=0)
classes = ["BOMBAY", "CALI", "SIRA"]
first_class = ttk.Combobox(model_classes, values=classes, width="30")
first_class.grid(row=2, column=0)

lb4 = Label(model_classes, text="Second Class", font=text_font, fg=text_color)
lb4.grid(row=1, column=1)
second_class = ttk.Combobox(model_classes, values=classes, width="30")
second_class.grid(row=2, column=1)

for widget in model_classes.winfo_children():
    widget.grid_configure(padx=35, pady=10)

# Create Frame 3 Parameters
###########################
frame3 = Frame(form)
frame3.pack()
model_parameter = LabelFrame(frame3, text="Parameters", font=header_font, fg=text_color)
model_parameter.grid(row=2, column=0, pady=10)

lb5 = Label(model_parameter, text="Learning rate", font=text_font, fg=text_color)
lb5.grid(row=2, column=0)
learning_rate = Entry(model_parameter, width="26")
learning_rate.grid(row=3, column=0)

lb6 = Label(model_parameter, text="Epochs", font=text_font, fg=text_color)
lb6.grid(row=2, column=1)
epochs = Spinbox(model_parameter, from_=0, to="infinity", width="25")
epochs.grid(row=3, column=1)

lb7 = Label(model_parameter, text="MSE", font=text_font, fg=text_color)
lb7.grid(row=2, column=2)
mse = Entry(model_parameter, width="26")
mse.grid(row=3, column=2)

for widget in model_parameter.winfo_children():
    widget.grid_configure(padx=10, pady=10)

# Create Frame 3 Algorithm
###########################
frame4 = Frame(form)
frame4.pack()
algo = LabelFrame(frame4, text="Algorithm", font=header_font, fg=text_color)
algo.grid(row=3, column=0, pady=10)

bias_retrived = tkinter.StringVar(value="False")
bias = Checkbutton(algo, text="Add bias", variable=bias_retrived, onvalue="True", offvalue="False", font=text_font,
                   fg=text_color)
bias.grid(row=3, column=0)

algo_var = IntVar()
perceptron = Radiobutton(algo, text="Perceptron", font=text_font, variable=algo_var, value=1, fg=text_color)
perceptron.grid(row=4, column=0)
adaline = Radiobutton(algo, text="Adaline", font=text_font, variable=algo_var, value=2, fg=text_color)
adaline.grid(row=4, column=1)

for widget in algo.winfo_children():
    widget.grid_configure(padx=85, pady=0)

btn1 = Button(frame4, command=submit, text="Submit", width="25", height="1", font=text_font, bg="#b5ceb1",
              fg=text_color)
btn1.grid(row=4, column=0, padx=20, pady=10)
form.mainloop()
