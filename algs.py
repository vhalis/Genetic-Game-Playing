import operator


def add_at_idx(t, idx, val, typ=tuple, op=operator.__add__):
    """
    add_at_idx(t, idx, val, typ=tuple, op=operator.__add__) ->
        t with t[idx] = op(t[idx], val)
    """
    return t[:idx] + typ((op(t[idx], val),)) + t[idx+1:]


def add_at_l_idx(t, idx, val, op=operator.__add__):
    """See add_at_idx but uses typ=list"""
    return add_at_idx(t, idx, val, typ=list, op=op)


def get_class_hierarchy_attrs(klass, attr_name, output_type=list, sub_attr=None):
    """Works with dictionaries and lists as output type, not tuples"""

    multiple_attrs = False
    if (not isinstance(attr_name, basestring) and
            hasattr(attr_name, '__iter__')):
        multiple_attrs = True
    if multiple_attrs:
        output = list(output_type() for _ in xrange(len(attr_name)))
    else:
        output = output_type()

    output_is_dict = isinstance(output, dict)
    def add_to_output(kls, name, idx=None):
        o = getattr(kls, name, output_type())
        if idx is not None:
            out = output[idx]
        else:
            out = output
        if output_is_dict:
            for k, v in o.iteritems():
                out[k] = v
        else:
            if (isinstance(o, basestring) or
                    not hasattr(o, '__iter__')):
                out.append(o)
            else:
                for k in o:
                    out.append(k)

    for kls in klass.__mro__:
        if sub_attr:
            if hasattr(kls, sub_attr):
                kls = getattr(kls, sub_attr)
            else:
                continue
        if multiple_attrs:
            for idx, name in enumerate(attr_name):
                add_to_output(kls, name, idx)
        else:
            add_to_output(kls, attr_name)
    return output


def remove_at_idx(t, idx):
    """Removes the entry at index idx from a sliceable"""
    return t[:idx] + t[idx+1:]


def n_cmp(t_1, t_2):
    """Compare two iterables with the same length item by item"""
    for a, b in zip(t_1, t_2):
        c = cmp(a, b)
        if c != 0:
            return c
    return 0


def weighted_idx_cmp(idx):
    """
    Returns a function that compares the index provided after all other
    indices have been compared
    """
    def w_cmp(t_1, t_2):
        c_1 = remove_at_idx(t_1, idx)
        c_2 = remove_at_idx(t_2, idx)
        v = n_cmp(c_1, c_2)
        if v == 0:
            return cmp(t_1[idx], t_2[idx])
        else:
            return v
    return w_cmp


def sort_quickest_ascending(sortable, idx, reverse=False):
    return sorted(sortable, weighted_idx_cmp(idx), reverse=reverse)
