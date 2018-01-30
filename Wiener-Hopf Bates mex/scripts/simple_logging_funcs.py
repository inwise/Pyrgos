def tree_to_csv_file(tree, fname):
    """ logging output of trees"""
    treefile = open(fname, 'w')
    treefile.write('time scale' + '; ')
    for col_num in range(len(tree[0])):
        # writing colnames
        treefile.write('k-state' + str(col_num) + '; ')
    treefile.write('\n')
    for row_num in range(len(tree)):
        treefile.write('moment ' + str(row_num) + ": ")
        for col_num in range(len(tree[row_num])):
            treefile.write(str(tree[row_num][col_num]) + ';')
        treefile.write('\n')
    treefile.close()


def array_to_csv_file(array_to_write, fname):
    """simply writing an array to a file in a line, with no additional info"""
    array_file = open(fname, 'w')
    for elem in array_to_write:
        array_file.write(str(elem))
        array_file.write('\n')
    array_file.close()
