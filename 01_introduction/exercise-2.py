def print_list(list):
    for item in list:
        print(item)
    print("")

writers = ["Agatha Christie", "J. K. Rowling", "Dr. Seuss", "J. R. R. Tolkien", "Shakespeare"]
print_list(writers)

writers.append("J. D. Salinger")
print_list(writers)

writers.remove("J. K. Rowling")
print_list(writers)

len = len(writers)
print("There are " + str(len) + " writers in the list. \n")

writers.reverse()
print_list(writers)
