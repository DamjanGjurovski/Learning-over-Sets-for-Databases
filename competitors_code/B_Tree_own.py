import bisect
import math
import sys


def flatten(l):
    return [y for x in l for y in x]

# return bytes
def size_calculator(elem, gso):
    if gso:
        return sys.getsizeof(elem)
    # print(type(elem))
    if isinstance(elem, int):
        return 4
    elif isinstance(elem, list):
        total_elems = 0
        for i in range(len(elem)):
            elem_i = elem[i]
            if isinstance(elem_i, int):
                total_elems += 1
            else:
                total_elems += len(elem_i)
        return total_elems * 4
    else:
        elem_type = type(elem)
        print(elem_type)
        exit(1)






class Leaf:
    def __init__(self, previous_leaf, next_leaf, parent, b_factor):
        self.previous = previous_leaf
        self.next = next_leaf
        self.parent = parent
        self.b_factor = b_factor
        self.a_factor = math.ceil(b_factor/2)
        self.keys = []
        self.children = []

    def get_size(self, gso):
        # print("Going here")
        # total_size = sys.getsizeof(self.previous) + sys.getsizeof(self.next) + sys.getsizeof(self.parent) + sys.getsizeof(self.b_factor) + sys.getsizeof(self.a_factor) + sys.getsizeof(self.keys) + sys.getsizeof(self.children)
        total_size = size_calculator(self.b_factor, gso) + size_calculator(self.a_factor, gso) + size_calculator(
            self.keys, gso) + size_calculator(self.children, gso)
        return total_size

    @property
    def is_root(self):
        return self.parent is None

    def insert(self, key, value):
        index = bisect.bisect_left(self.keys, key)
        if index < len(self.keys) and self.keys[index] == key:
            self.children[index].append(value)
        else:
            self.keys.insert(index, key)
            self.children.insert(index, [value])
            if len(self.keys) > self.b_factor:
                split_index = math.ceil(self.b_factor/2)
                self.split(split_index)

    def get(self, key):
        index = bisect.bisect_left(self.keys, key)
        if index < len(self.keys) and self.keys[index] == key:
            return self.children[index]
        else:
            return None

    def split(self, index):
        new_leaf_node = Leaf(self, self.next, self.parent, self.b_factor)
        new_leaf_node.keys = self.keys[index:]
        new_leaf_node.children = self.children[index:]
        self.keys = self.keys[:index]
        self.children = self.children[:index]
        if self.next is not None:
            self.next.previous = new_leaf_node
        self.next = new_leaf_node
        if self.is_root:
            self.parent = Node(None, None, [new_leaf_node.keys[0]], [self, self.next], b_factor=self.b_factor, parent=None)
            self.next.parent = self.parent
        else:
            self.parent.add_child(self.next.keys[0], self.next)

    def find_left(self, key, include_key=True):
        items = []
        index = bisect.bisect_right(self.keys, key) - 1
        if index == -1:
            items = []
        else:
            if include_key:
                items = self.children[:index+1]
            else:
                if key == self.keys[index]:
                    index -= 1
                items = self.children[:index+1]
        return self.left_items() + flatten(items)

    def find_right(self, key, include_key=True):
        items = []
        index = bisect.bisect_left(self.keys, key)
        if index == len(self.keys):
            items = []
        else:
            if include_key:
                items = self.children[index:]
            else:
                if key == self.keys[index]:
                    index += 1
                items = self.children[index:]
        return flatten(items) + self.right_items()

    def left_items(self):
        items = []
        node = self
        while node.previous is not None:
            node = node.previous
        while node != self:
            for elem in node.children:
                if type(elem) == list:
                    items.extend(elem)
                else:
                    items.append(elem)
            node = node.next
        return items

    def right_items(self):
        items = []
        node = self.next
        while node is not None:
            for elem in node.children:
                if type(elem) == list:
                    items.extend(elem)
                else:
                    items.append(elem)
            node = node.next
        return items

    def items(self):
        return zip(self.keys, self.children)

class Node:
    def __init__(self, previous_node, next_node, keys, children, b_factor, parent=None):
        self.previous = previous_node
        self.next = next_node
        self.keys = keys
        self.children = children
        self.b_factor = b_factor
        self.a_factor = math.ceil(b_factor / 2)
        self.parent = parent

    def get_size(self, gso):
        # total_size = sys.getsizeof(self.previous) + sys.getsizeof(self.next) + sys.getsizeof(self.keys) + sys.getsizeof(self.children) + sys.getsizeof(self.b_factor)+ sys.getsizeof(self.a_factor) + sys.getsizeof(self.parent)
        total_size = size_calculator(self.keys, gso) + size_calculator(self.b_factor, gso) + size_calculator(self.a_factor, gso)

        return total_size
        # if self.next_node is not None:
        #     total_size += sys.getsizeof(self.next_node)

    @property
    def degree(self):
        return len(self.children)

    @property
    def is_root(self):
        return self.parent is None

    def insert(self, key, value):
        index = bisect.bisect_right(self.keys, key)
        node = self.children[index]
        node.insert(key, value)

    def get(self, key):
        index = bisect.bisect_right(self.keys, key)
        return self.children[index].get(key)

    def find_left(self, key, include_key=True):
        index = bisect.bisect_right(self.keys, key)
        return self.children[index].find_left(key, include_key)

    def find_right(self, key, include_key=True):
        index = bisect.bisect_right(self.keys, key)
        return self.children[index].find_right(key, include_key)

    def add_child(self, key, child):
        index = bisect.bisect_right(self.keys, key)
        self.keys.insert(index, key)
        self.children.insert(index+1, child)
        if self.degree > self.b_factor:
            split_index = math.floor(self.b_factor / 2)
            self.split(split_index)

    def split(self, index):
        split_key = self.keys[index]
        new_node = Node(self, self.next, self.keys[index+1:], self.children[index+1:], self.b_factor, self.parent)
        for node in self.children[index+1:]:
            node.parent = new_node
        self.keys = self.keys[:index]
        self.children = self.children[:index+1]

        if self.next is not None:
            self.next.previous = new_node
        self.next = new_node
        if self.is_root:
            self.parent = Node(None, None, [split_key], [self, self.next], b_factor=self.b_factor, parent=None)
            self.next.parent = self.parent
        else:
            self.parent.add_child(split_key, self.next)




class BPlusTree:
    def __init__(self, b_factor=32):
        self.b_factor = b_factor
        self.root = Leaf(None, None, None, b_factor)
        self.size = 0

    def get(self, key):
        return self.root.get(key)

    def __getitem__(self, key):
        return self.get(key)

    def __len__(self):
        return self.size

    def insert(self, key, value):
        self.root.insert(key, value)
        self.size += 1
        if self.root.parent is not None:
            self.root = self.root.parent

    def range_search(self, notation, cmp_key):
        notation = notation.strip()
        if notation not in [">", "<", ">=", "<="]:
            raise Exception("Nonsupport notation: {}. Only '>' '<' '>=' '<=' are supported".format(notation))
        if notation == '>':
            return self.root.find_right(cmp_key, False)
        if notation == '>=':
            return self.root.find_right(cmp_key, True)
        if notation == '<':
            return self.root.find_left(cmp_key, False)
        if notation == '<=':
            return self.root.find_left(cmp_key, True)

    def search(self, notation, cmp_key):
        notation = notation.strip()
        if notation not in [">", "<", ">=", "<=", "==", "!="]:
            raise Exception("Nonsupport notation: {}. Only '>' '<' '>=' '<=' '==' '!=' are supported".format(notation))
        if notation == '==':
            res = self.get(cmp_key)
            if res is None:
                return []
            else:
                return res
        if notation == '!=':
            return self.root.find_left(cmp_key, False) + self.root.find_right(cmp_key, False)
        return self.range_search(notation, cmp_key)

    def show(self):
        tree_size = 0
        layer = 0
        node = self.root
        while node is not None:
            tree_size += len(node.keys)
            print("Layer: {}".format(layer))
            inner_node = node
            while inner_node is not None:
                print(inner_node.keys, end=' ')
                inner_node = inner_node.next
                if inner_node is not None:
                    tree_size += len(inner_node.keys)
            print('')
            node = node.children[0]
            layer += 1
            if type(node) != Leaf and type(node) != Node:
                break
        print(tree_size)


    def get_size(self, gso = True):
        tree_size = 0
        layer = 0
        node = self.root
        while node is not None:
            tree_size += node.get_size(gso)
            # print("Layer: {}".format(layer))
            inner_node = node
            while inner_node is not None:
                # print(inner_node.keys, end=' ')
                inner_node = inner_node.next
                if inner_node is not None:
                    tree_size += inner_node.get_size(gso)
            # print('')
            node = node.children[0]
            layer += 1
            if type(node) != Leaf and type(node) != Node:
                break
        # print(tree_size)
        return tree_size


    def leftmost_leaf(self):
        leaf = self.root
        while type(leaf) != Leaf:
            leaf = leaf.children[0]
        return leaf

    def items(self):
        leaf = self.leftmost_leaf()
        items = []
        while leaf is not None:
            pairs = list(leaf.items())
            items.extend(pairs)
            leaf = leaf.next
        return items

    def keys(self):
        leaf = self.leftmost_leaf()
        ks = []
        while leaf is not None:
            ks.extend(leaf.keys)
            leaf = leaf.next
        return ks

    def values(self):
        leaf = self.leftmost_leaf()
        vals = []
        while leaf is not None:
            for elem in leaf.children:
                if type(elem) == list:
                    vals.extend(elem)
                else:
                    vals.append(elem)
            leaf = leaf.next
        return vals

    def height(self):
        node = self.root
        height = 0
        while type(node) != Leaf:
            height += 1
            node = node.children[0]
        return height


if __name__ == '__main__':
    t = BPlusTree(32)
    nums = [55,44,65,16,80,74,14,19,95,36,2,90,74,94,27,89,85]
    for x in nums:
        t.insert(x, x)
    print(t.items())