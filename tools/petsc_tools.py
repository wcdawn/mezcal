import numpy as np
import scipy.sparse
import re

def to_coord(fname):
    file_data = open(fname, 'r').readlines()

    row = re.compile('^\s*row\s+(\d+)')
    record = re.compile('\((\d+),\s*([+-]?([0-9]+([.][0-9]*)?|[.][0-9]+)([eE][+-]?[0-9]+)?)\)')

    row_ind = []
    col_ind = []
    data = []

    last_row = 0

    for line in file_data:
        entry = row.match(line)
        if (entry is not None) :
            this_row = int(entry.group(1))
            last_row = this_row
            records = record.finditer(line)
            for rec in records:
                this_col = int(rec.group(1))
                this_val = float(rec.group(2))
                row_ind.append(this_row)
                col_ind.append(this_col)
                data.append(this_val)

    shape = (last_row+1, max(col_ind)+1)

    return row_ind, col_ind, data, shape

def read_matrix(fname):
    row_ind, col_ind, data, shape = petsc_to_coord(fname)
    return scipy.sparse.csr_matrix((data, (row_ind, col_ind)), shape=shape)
