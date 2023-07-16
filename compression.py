import math
def compress_elem(elem, sv_d, ns = 2, one_hot = True):
    # print("The elem is " + str(elem) + "The svd is " + str(sv_d))
    sv_q = math.ceil(elem/sv_d)
    # get the reminder from the division and increase by 1 to avoid 0
    sv_r = (elem % sv_d)
    if one_hot:
        return sv_q, sv_r
    else:
        return sv_q + 1, sv_r + 1

def compress_elem_ns(elem, sv_d, ns = 2, one_hot = True):
    i = 0
    compressed_elems = []
    number_to_compress = elem
    while i < (ns - 1):
        sv_q, sv_r = compress_elem(number_to_compress, sv_d, one_hot = one_hot)
        compressed_elems.append(sv_r)
        number_to_compress = sv_q
        i += 1
    compressed_elems.append(sv_q)
    return compressed_elems